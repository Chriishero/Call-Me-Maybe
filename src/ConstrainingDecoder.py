from pydantic import BaseModel
from typing import Any


class ConstrainingDecoder(BaseModel):
    prompt: str
    vocabulary: list[str]
    functions_def: list[dict[str, str | dict[Any, Any]]]

    def constrain_function(
            self, logits: list[float], current_function_name: str
            ) -> list[float]:
        right_prefix = False
        for i in range(len(self.vocabulary)):
            for function in self.functions_def:
                if str(function['name']).startswith(
                    current_function_name + self.vocabulary[i]
                ):
                    right_prefix = True
            if right_prefix is False:
                logits[i] = float("-inf")
            right_prefix = False
        return logits

    def constrain_params(
            self, logits: list[float], current_params: str,
            parameter_type: str
    ) -> list[float]:
        for i in range(len(self.vocabulary)):
            if parameter_type == "number":
                try:
                    float(self.vocabulary[i])
                    float(current_params + self.vocabulary[i])
                except Exception:
                    logits[i] = float("-inf")
            elif parameter_type == "string":
                in_a_word = False
                for word in self.prompt.split(' '):
                    word = word.strip("'\".,?!")
                    if current_params + self.vocabulary[i] in word:
                        in_a_word = True
                if in_a_word is False:
                    logits[i] = float("-inf")
        return logits
