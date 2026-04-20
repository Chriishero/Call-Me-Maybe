from pydantic import BaseModel, PrivateAttr
from typing import Any
from llm_sdk import Small_LLM_Model
import json
from enum import Enum
import numpy as np
from .ConstrainingDecoder import ConstrainingDecoder


class GenerationState(Enum):
    function_name = "function_name"
    parameters = "parameters"


class GenerationPipeline(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }
    model: Small_LLM_Model
    prompt: str
    functions_def: list[dict[str, str | dict[Any, Any]]]

    _output: dict[str, Any] = PrivateAttr(default_factory=dict)
    _vocabulary: list[str] = PrivateAttr(default_factory=list)
    _parameters_types: list[Any] = PrivateAttr(default_factory=list)
    _generated_function: dict[Any, Any] = PrivateAttr(default_factory=dict)

    @property
    def output(self) -> dict[str, Any]:
        return self._output

    def generate_output(self) -> None:
        self.load_vocabulary()
        decoder = ConstrainingDecoder(
            prompt=self.prompt,
            vocabulary=self._vocabulary,
            functions_def=self.functions_def
        )
        self.output['prompt'] = self.prompt
        system_prompt = self.get_system_prompt()
        output_str = system_prompt
        fun_name_state = False
        params_state = False
        generated_function_name = ""
        while fun_name_state is False:
            logits = self.llm_pipeline(output_str)
            constrained_logits = decoder.constrain_function(
                logits=logits, current_function_name=generated_function_name
            )
            selected_token = self.token_selection(constrained_logits)
            generated_function_name += selected_token
            output_str += selected_token
            for function in self.functions_def:
                if str(function['name']) in generated_function_name:
                    fun_name_state = True
                    self._generated_function = function
        self.output['name'] = self._generated_function['name']
        self.output['parameters'] = {}
        while params_state is False:
            for name, d_type in self._generated_function['parameters'].items():
                p_type = d_type['type']
                output_str += (f"\nParameter '{name}' of "
                               f"type {p_type}: ")
                generated_param = ""
                selected_token = ""
                while generated_param + selected_token in self.prompt:
                    generated_param += selected_token
                    output_str += selected_token
                    logits = self.llm_pipeline(output_str)
                    constrained_logits = decoder.constrain_params(
                        logits=logits,
                        current_params=str(generated_param),
                        parameter_type=p_type
                    )
                    selected_token = self.token_selection(constrained_logits)
                self.output['parameters'][name] = (
                    float(generated_param)if p_type == "NUMBERS"
                    else generated_param
                    )
            params_state = True

    def get_system_prompt(self) -> str:
        return (
            "You are a specialized agent in function calling.\n"
            "Your goal is, from some available functions, "
            "to give the right function name with the right "
            "parameters to solve my prompts.\n"
            "You must first generate the function name, then"
            "the parameters with the right type.\n"
            "The available functions are:"
            f"{self.functions_def}\n"
            f"and the prompt is {self.prompt}.\n")

    def llm_pipeline(self, text: str) -> list[float]:
        input_ids = self.model.encode(text)
        input_ids_list = input_ids[0].tolist()
        logits: list[float] = self.model.get_logits_from_input_ids(
            input_ids_list
        )
        return logits

    def token_selection(self, logits: list[float]) -> str:
        trimmed = logits[:len(self._vocabulary)]
        best_index = np.argmax(trimmed)
        return self._vocabulary[best_index]

    def load_vocabulary(self) -> None:
        try:
            with open(self.model.get_path_to_vocab_file(), 'r') as f:
                vocab_dict: dict[str, int] = json.loads(f.read())
            self._vocabulary = list(vocab_dict.keys())
        except FileNotFoundError as e:
            raise ValueError(f"Missing vocabulary file: {e}")
