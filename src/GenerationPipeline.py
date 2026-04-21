from pydantic import BaseModel, PrivateAttr, Field
from typing import Any, Optional
from llm_sdk import Small_LLM_Model
import json
import numpy as np
from .ConstrainingDecoder import ConstrainingDecoder


class GenerationPipeline(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }
    model: Small_LLM_Model
    prompt: str
    functions_def: list[dict[str, Any]]
    max_fn_tokens: int = Field(default=32)
    max_param_tokens: int = Field(default=64)

    _output: dict[str, Any] = PrivateAttr(default_factory=dict)
    _new_prompt: str = PrivateAttr(default_factory=str)
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
        self._output = {"prompt": self.prompt, "name": "", "parameters": {}}
        self._new_prompt = self.get_system_prompt()
        self.generate_function_name(decoder)
        self.generate_parameters(decoder)

    def generate_function_name(self, decoder: ConstrainingDecoder) -> None:
        token_count = 0
        while token_count < self.max_fn_tokens:
            logits = self.llm_pipeline(self._new_prompt)
            constrained_logits = decoder.constrain_function(
                logits=logits,
                current_function_name=self.output['name']
            )
            token = self.token_selection(constrained_logits)
            if token is None:
                break
            print(token)
            self.output['name'] += token
            self._new_prompt += token
            token_count += 1
            for function in self.functions_def:
                if str(function['name']) == self.output['name']:
                    self._generated_function = function
                    break
        if not self._generated_function:
            raise ValueError(
                f"Generated function name: {self.output['name']} does not "
                "match any available function.")

    def generate_parameters(self, decoder: ConstrainingDecoder) -> None:
        for p_name, type_dict \
                in self._generated_function['parameters'].items():
            p_type = type_dict['type']
            self.output['parameters'][p_name] = ""
            print(self._generated_function['parameters'])
            token_count = 0
            while token_count < self.max_param_tokens:
                logits = self.llm_pipeline(self._new_prompt)
                constrained_logits = decoder.constrain_parameter(
                    logits=logits,
                    current_param=self.output['parameters'][p_name],
                    parameter=(p_name, p_type)
                )
                token = self.token_selection(constrained_logits)
                if token is None:
                    break
                if p_type == "string":
                    token = token.replace("Ġ", " ")
                else:
                    token = token.replace("Ġ", "")
                print(token)
                if decoder.parameters_condition(
                    parameter=(p_name, p_type),
                    next_value=self.output['parameters'][p_name] + token
                        ) is False:
                    break
                self.output['parameters'][p_name] += token
                self._new_prompt += token
                token_count += 1
            try:
                if p_type == "number":
                    self.output['parameters'][p_name] = float(
                        self.output['parameters'][p_name]
                    )
            except Exception:
                self.output['parameters'][p_name] = None

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
