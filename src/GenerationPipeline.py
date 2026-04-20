from pydantic import BaseModel, PrivateAttr
from typing import Any, Optional
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
    _output_str: str = PrivateAttr(default_factory=str)
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
        self._output_str = system_prompt
        self.generate_function_name(decoder)
        self.generate_parameters(decoder)

    def generate_function_name(self, decoder: ConstrainingDecoder) -> None:
        fun_name_state = False
        generated_function_name = ""
        while fun_name_state is False:
            logits = self.llm_pipeline(self._output_str)
            constrained_logits = decoder.constrain_function(
                logits=logits, current_function_name=generated_function_name
            )
            selected_token = self.token_selection(constrained_logits)
            if selected_token is None:
                break
            generated_function_name += selected_token
            self._output_str += selected_token
            for function in self.functions_def:
                if str(function['name']) in generated_function_name:
                    fun_name_state = True
                    self._generated_function = function
        self.output['name'] = self._generated_function['name']

    def generate_parameters(self, decoder: ConstrainingDecoder) -> None:
        self.output['parameters'] = {}
        params_state = False
        while params_state is False:
            for name, d_type in self._generated_function['parameters'].items():
                p_type = d_type['type']
                self._output_str += (f"\nParameter '{name}': ")
                generated_param = ""
                logits = self.llm_pipeline(self._output_str)
                constrained_logits = logits
                selected_token = ""
                while True:
                    constrained_logits = decoder.constrain_params(
                        logits=logits,
                        current_param=str(generated_param),
                        parameter=(name, p_type)
                    )
                    selected_token = self.token_selection(constrained_logits)
                    if selected_token is None:
                        break
                    selected_token = selected_token.replace("Ġ", " ")
                    if decoder.parameters_condition(
                        parameter=(name, p_type),
                        new_parameter=generated_param + selected_token
                            ) is False:
                        break
                    generated_param += selected_token
                    self._output_str += selected_token
                    logits = self.llm_pipeline(self._output_str)
                self.output['parameters'][name] = (
                    float(generated_param) if p_type == "number"
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

    def token_selection(self, logits: list[float]) -> Optional[str]:
        trimmed = logits[:len(self._vocabulary)]
        if all(x == float("-inf") for x in trimmed):
            return None
        best_index = np.argmax(trimmed)
        return self._vocabulary[best_index]

    def load_vocabulary(self) -> None:
        try:
            with open(self.model.get_path_to_vocab_file(), 'r') as f:
                vocab_dict: dict[str, int] = json.loads(f.read())
            self._vocabulary = list(vocab_dict.keys())
        except FileNotFoundError as e:
            raise ValueError(f"Missing vocabulary file: {e}")
