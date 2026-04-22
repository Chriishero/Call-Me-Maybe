from pydantic import BaseModel, PrivateAttr
from typing import Any
import re


class ConstrainingDecoder(BaseModel):
    prompt: str
    vocabulary: list[str]
    functions_def: list[dict[str, Any]]

    _source_string: str = PrivateAttr(default="")
    _regex: str = PrivateAttr(default="")
    _replacement: str = PrivateAttr(default="")
    _s: str = PrivateAttr(default="")

    def constrain_function(
            self, logits: list[float], current_function_name: str
            ) -> list[float]:
        self.load_functions_quandidates()
        for i in range(len(self.vocabulary)):
            right_prefix = False
            for function in self._function_quandidates:
                if function.startswith(
                    current_function_name + self.vocabulary[i]
                ):
                    right_prefix = True
            if right_prefix is False:
                logits[i] = float("-inf")
        return logits

    def parameters_condition(
            self, parameter: tuple[str, str], next_value: str
            ) -> bool:
        parameter_name = parameter[0]
        parameter_type = parameter[1]
        if parameter_type == "number":
            nums = re.findall(r"\d+", self.prompt)
            if next_value == "":
                return True
            try:
                if parameter_name == 'a':
                    if nums[0].startswith(next_value):
                        return True
                elif parameter_name == 'b':
                    if nums[1].startswith(next_value):
                        return True
            except IndexError:
                return False
        elif parameter_type == "string":
            if parameter_name == "source_string":
                self.load_quotes()
                if any(quote.startswith(next_value)
                       for quote in self._quotes):
                    self._source_string = next_value
                    return True
            elif parameter_name == "regex":
                self.load_regex_candidates()
                if any(regex.startswith(next_value)
                       for regex in self._regex_candidates):
                    self._regex = next_value
                    return True
            elif parameter_name == "replacement":
                self.load_replacement_candidates()
                if any(rep.strip("'").strip('"').startswith(next_value)
                       for rep in self._replacements_candidates):
                    self._replacement = next_value
                    return True
            elif parameter_name == "s":
                self.load_reverse_candidates()
                if any(c.strip("'").strip('"').startswith(next_value)
                        for c in self._reverse_candidates):
                    self._s = next_value
                    return True
            else:
                if any(word.strip('"').strip("'").startswith(next_value)
                       for word in self.prompt.split(' ')):
                    return True
        return False

    def constrain_parameter(
            self, logits: list[float], current_param: str,
            parameter: tuple[str, str]
    ) -> list[float]:
        parameter_name = parameter[0]
        parameter_type = parameter[1]
        for i in range(len(self.vocabulary)):
            new_parameter = current_param + self.vocabulary[i]
            if parameter_type == "number":
                logits[i] = self.constrain_number(
                    parameter_name=parameter_name,
                    current_param=current_param,
                    token=self.vocabulary[i],
                    logit=logits[i]
                )
            elif parameter_type == "string":
                logits[i] = self.constrain_string(
                    parameter_name=parameter_name,
                    new_parameter=new_parameter,
                    logit=logits[i]
                )
        return logits

    def constrain_number(
            self, parameter_name: str, current_param: str,
            token: str, logit: float
            ) -> float:
        token = token.replace("Ġ", "")
        if not token.isdigit():
            return float("-inf")
        nums = re.findall(r"\d+", self.prompt)
        try:
            if parameter_name == 'a':
                if nums[0].startswith(current_param + token):
                    return logit
            elif parameter_name == 'b':
                if nums[1].startswith(current_param + token):
                    return logit
        except IndexError:
            return float("-inf")
        return float("-inf")

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
        elif parameter_name == "s":
            self.load_reverse_candidates()
            for word in self._reverse_candidates:
                word = word.strip("'\".,?!")
                if word.startswith(new_parameter):
                    return logit
        else:
            for word in self.prompt.split(' '):
                word = word.strip("'\".,?!")
                if word.startswith(new_parameter):
                    return logit
        return float("-inf")

    def load_functions_quandidates(self) -> None:
        if hasattr(self, "_function_quandidates"):
            return
        self._function_quandidates = set()
        numbers = re.findall(r"\d+", self.prompt)
        for function in self.functions_def:
            params = function["parameters"]
            has_number_param = any(
                p["type"] == "number" for p in params.values()
            )
            if has_number_param and not numbers:
                continue
            self._function_quandidates.add(function["name"])

    def load_quotes(self) -> None:
        if hasattr(self, "_quotes"):
            return
        self._quotes = set()
        quotes = re.findall(r"'.*?'|\".*?\"", self.prompt)
        for quote in quotes:
            self._quotes.add(fr"{quote[1:-1]}")

    def load_regex_candidates(self) -> None:
        if hasattr(self, "_regex_candidates"):
            return
        self._regex_candidates = set()
        prompt_lower = self.prompt.lower()
        if "number" in prompt_lower:
            self._regex_candidates.add(r"\d+")
        if "digit" in prompt_lower:
            self._regex_candidates.add(r"[0-9]")
        if "words" in prompt_lower:
            self._regex_candidates.add(r"\w+")
        if "letter" in prompt_lower:
            self._regex_candidates.add(r"[a-zA-z]")
        if any(w in prompt_lower for w in ("space", "indent", "line")):
            self._regex_candidates.add(r"\s")
        if "vowel" in prompt_lower:
            self._regex_candidates.add(r"[aeiouAEIOU]")
        if "consonant" in prompt_lower:
            self._regex_candidates.add(
                r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]")
        self.load_quotes()
        self.load_replacement_candidates()
        for quote in self._quotes:
            if quote not in self._replacements_candidates \
                    and quote != self._source_string:
                self._regex_candidates.add(quote)

    def load_replacement_candidates(self) -> None:
        if hasattr(self, "_replacement_candidates"):
            return
        self._replacements_candidates = set()
        words_next = re.findall(
            r"\b(?:with|by|to)\s+(\S+)", self.prompt
        )
        for w in words_next:
            if w != self._regex and w != self._source_string:
                self._replacements_candidates.add(
                    fr"{w.strip().strip("'").strip('"')}")

    def load_reverse_candidates(self) -> None:
        if hasattr(self, "_reverse_candidates"):
            return
        self._reverse_candidates = set()
        self.load_quotes()
        self._reverse_candidates.update(self._quotes)
        prompt_lower = self.prompt.lower()
        if "word" in prompt_lower:
            match = re.search(
                r"reverse ?(?:the )?(?:word )?(\S+)",
                prompt_lower)
        else:
            match = re.search(
                r"reverse ?(?:the )?(?:string |phrase)?(.+?)",
                prompt_lower)
        if match:
            candidate = match.group(1).strip(" .!?")
            pos = prompt_lower.find(candidate)
            self._reverse_candidates.add(self.prompt[pos:len(candidate)])
