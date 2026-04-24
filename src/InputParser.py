from pydantic import BaseModel, model_validator, PrivateAttr
from .Input import Type, Parameter, Function, Prompt
import json
from typing import Any


class InputParser(BaseModel):
    """Use pydantic and the json module to check the input's
    files format."""
    model_config = {
        "arbitrary_types_allowed": True
    }
    fun_call_tests_path: str
    funs_def_path: str

    _function_calling_tests_json: list[dict[str, str]] = PrivateAttr(
        default_factory=list)
    _functions_definition_json: list[dict[str, Any]] = PrivateAttr(
        default_factory=list)
    _functions: list[Function] = PrivateAttr(default_factory=list)
    _prompts: list[Prompt] = PrivateAttr(default_factory=list)

    @model_validator(mode='after')
    def validation(self) -> "InputParser":
        """Check if the input files are readable and are in
        an adequate JSON format."""
        try:
            with open(self.fun_call_tests_path, 'r') as f:
                raw_tests = f.read()
            with open(self.funs_def_path, 'r') as f:
                raw_defs = f.read()
        except Exception as e:
            raise ValueError(
                f"Error reading input files: {e}")
        try:
            self._function_calling_tests_json = json.loads(raw_tests)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in functions definition: {e}")
        try:
            self._functions_definition_json = json.loads(raw_defs)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in functions definition: {e}")
        return self

    @property
    def functions(self) -> list[Function]:
        return self._functions

    @property
    def prompts(self) -> list[Prompt]:
        return self._prompts

    def parse(self) -> None:
        self.parse_functions()
        self.parse_prompts()

    def parse_functions(self) -> None:
        for function_def in self._functions_definition_json:
            params = list()
            for p_name, dict_type in function_def['parameters'].items():
                p_type = dict_type['type']
                try:
                    e_type = Type[p_type]
                except Exception as e:
                    raise ValueError(
                        f"Invalid type '{p_type}': {e}"
                    )
                parameter = Parameter(
                    name=p_name,
                    type=e_type
                )
                params.append(parameter)
            function = Function(
                name=function_def['name'],
                description=function_def['description'],
                parameters=params
            )
            self._functions.append(function)

    def parse_prompts(self) -> None:
        for calling_test in self._function_calling_tests_json:
            prompt = Prompt(prompt=calling_test['prompt'])
            self._prompts.append(prompt)
