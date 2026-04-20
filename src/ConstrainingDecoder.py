from pydantic import BaseModel
from typing import Any
import re


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

    def parameters_condition(
            self, parameter: tuple[str, str], new_parameter: str
            ) -> bool:
        parameter_name = parameter[0]
        parameter_type = parameter[1]
        if parameter_type == "number":
            nums = re.findall(r"\d+", self.prompt)
            if any(n.startswith(new_parameter) for n in nums):
                return True
        elif parameter_type == "string":
            if parameter_name == "source_string":
                self.load_quotes()
                if any(quote.startswith(new_parameter)
                       for quote in self._quotes):
                    return True
            elif parameter_name == "regex":
                self.load_regex_candidates()
                if any(regex.startswith(new_parameter)
                       for regex in self._regex_candidates):
                    return True
            elif parameter_name == "replacement":
                self.load_replacement_candidates()
                if any(rep.strip("'").strip('"').startswith(new_parameter)
                       for rep in self._replacements_candidates):
                    return True
            else:
                if any(word.strip("'").strip('"').startswith(new_parameter)
                        for word in self.prompt.split(' ')):
                    return True
        return False

    def constrain_params(
            self, logits: list[float], current_param: str,
            parameter: tuple[str, str]
    ) -> list[float]:
        parameter_name = parameter[0]
        parameter_type = parameter[1]
        for i in range(len(self.vocabulary)):
            new_parameter = current_param + self.vocabulary[i]
            if parameter_type == "number":
                logits[i] = self.constrain_number(
                    self.vocabulary[i], current_param, logits[i]
                )
            elif parameter_type == "string":
                logits[i] = self.constrain_string(
                    parameter_name=parameter_name,
                    new_parameter=new_parameter,
                    logit=logits[i]
                )
        return logits

    def constrain_number(
            self, token: str, current_param: str, logit: float
            ) -> float:
        try:
            float(token)
            float(current_param + token)
        except Exception:
            logit = float("-inf")
        return logit

    def constrain_string(
            self, parameter_name: str,
            new_parameter: str, logit: float,
            ) -> float:
        new_parameter = new_parameter.replace("Ġ", " ")
        if parameter_name == "source_string":
            self.load_quotes()
            for quote in self._quotes:
                if quote.startswith(new_parameter):
                    return logit
        elif parameter_name == "regex":
            self.load_regex_candidates()
            for regex in self._regex_candidates:
                if regex.startswith(new_parameter):
                    return logit
        elif parameter_name == "replacement":
            self.load_replacement_candidates()
            for rep in self._replacements_candidates:
                if rep.startswith(new_parameter):
                    return logit
        else:
            for word in self.prompt.split(' '):
                word = word.strip("'\".,?!")
                if word.startswith(new_parameter):
                    return logit
        return float("-inf")

    def load_quotes(self) -> None:
        if hasattr(self, "_quotes"):
            return
        self._quotes = set()
        quotes = re.findall(r"'.*?'|\".*?\"", self.prompt)
        for quote in quotes:
            value = quote[1:-1]
            if value:
                self._quotes.add(fr"{quote}")

    def load_regex_candidates(self) -> None:
        if hasattr(self, "_regex_candidates"):
            return
        self._regex_candidates = set()
        self._regex_candidates.update([
            r"\d+", r"\d", r"\w+", r"[A-Za-z]+", r"\s+"
            r"[aeiouAEIOU]+", r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]+"
            ])
        words = re.findall(r"\b\[A-Za-z]\b", self.prompt)
        for word in words:
            self._regex_candidates.add(fr"{word}")
        self.load_quotes()
        self._regex_candidates.update(self._quotes)

    def load_replacement_candidates(self) -> None:
        if hasattr(self, "_replacement_candidates"):
            return
        self._replacements_candidates = set()
        words_next_with = re.findall(
            r"\bwith\s+(?:\"([^\"]+)\"|'([^']+)'|(\S+))", self.prompt
        )
        for w1, w2, w3 in words_next_with:
            value = w1 or w2 or w3
            if value:
                self._replacements_candidates.add(fr"{value.strip()}")
        quoted_words = re.findall(
            r"'\w+'|\"w+\"", self.prompt
        )
        for quote in quoted_words:
            value = quote[0] or quote[1]
            if value:
                self._replacements_candidates.add(fr"{value.strip()}")
        upper_case_words = re.findall(
            r"[A-Z]{2,}", self.prompt
        )
        for word in upper_case_words:
            self._replacements_candidates.add(fr"{word}")
