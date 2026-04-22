from pydantic import BaseModel, model_validator, PrivateAttr
import json
from typing import Any


class InputParser(BaseModel):
    """Use pydantic and the json module to check the input's
    files format."""
    fun_call_tests_path: str
    funs_def_path: str

    _function_calling_tests: list[dict[str, str]] = PrivateAttr()
    _functions_definition: list[dict[str, Any]] = PrivateAttr()

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
            self._function_calling_tests = json.loads(raw_tests)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in functions definition: {e}")
        try:
            self._functions_definition = json.loads(raw_defs)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in functions definition: {e}")
        return self

    @property
    def function_calling_tests(self) -> list[dict[str, str]]:
        """Getter for the class attribute '_function_calling_tests'"""
        return self._function_calling_tests

    @property
    def functions_definition(self) -> list[dict[str, str]]:
        """Getter for the class attribute '_functions_definition'"""
        return self._functions_definition
