from pydantic import BaseModel
from typing import Any
import re


class ConstrainingDecoder(BaseModel):
    """
    Decoder applying token-level constraints to ensure that
    generated function names and parameters are adapted to
    the prompt.
    """
    prompt: str
    vocabulary: list[str]
    functions_def: list[dict[str, Any]]

    def constrain_function(
            self, logits: list[float], current_function_name: str
            ) -> list[float]:
        """Apply constrained decoding for functions name and
        return the updated logit list.
        """
        self.load_functions_candidates()
        for i in range(len(self.vocabulary)):
            right_prefix = False
            for function in self._function_candidates:
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
        """Condition of every parameters to stop, or continue,
        the token generation.

        Keyword arguments:
        parameter -- tuple with the parameter name and its type
        next_value -- concatenation of the current generated parameter
                      and the POTENTIALLY next token
        """
        parameter_name = parameter[0]
        parameter_type = parameter[1]
        if parameter_type == "number":
            nums = re.findall(r"\d+", self.prompt)
            if next_value == "":
                return True
            try:
                if parameter_name == 'a':
                    return bool(nums[0].startswith(next_value))
                elif parameter_name == 'b':
                    return bool(nums[1].startswith(next_value))
                else:
                    return any(n.startswith(next_value) for n in nums)
            except IndexError:
                return False
        elif parameter_type == "string":
            candidates = self._get_candidates(parameter_name)
            if not candidates:
                self._load_prompt_words()
                words = self._cached_prompt_words
                return any(
                    w.startswith(next_value) for w in words
                )
            return any(c.startswith(next_value) for c in candidates)
        return False

    def constrain_parameter(
            self, logits: list[float], current_param: str,
            parameter: tuple[str, str]
    ) -> list[float]:
        """Apply constrained decoding for parameter depending
        of its type. Return the updated logit list.
        """
        parameter_name = parameter[0]
        parameter_type = parameter[1]
        for i in range(len(self.vocabulary)):
            token = self.vocabulary[i]
            new_parameter = current_param + token
            if parameter_type == "number":
                logits[i] = self.constrain_number(
                    parameter_name=parameter_name,
                    current_param=current_param,
                    token=token,
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
        """Apply the constrained decoding specifically on the parameters
        of type 'number'. Return the update logit.
        """
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
            else:
                if any(n.startswith(current_param + token) for n in nums):
                    return logit
        except IndexError:
            return float("-inf")
        return float("-inf")

    def constrain_string(
            self, parameter_name: str,
            new_parameter: str, logit: float,
            ) -> float:
        """Apply the constrained decoding specifically on the parameters
        of type 'string'. Return the update logit.
        """
        new_parameter = new_parameter.replace("Ġ", " ")
        candidates = self._get_candidates(parameter_name)
        if candidates:
            for c in candidates:
                if c.startswith(new_parameter):
                    return logit
            return float("-inf")
        else:
            self._load_prompt_words()
            for word in self._cached_prompt_words:
                if word.startswith(new_parameter):
                    return logit
            return float("-inf")

    def _get_candidates(self, parameter_name: str) -> set[str]:
        """Call the functions that load the candidate values for a specific
        parameter. Return the candidates set.
        """
        if parameter_name == "source_string":
            self._load_source_candidates()
            return self._source_candidates
        elif parameter_name == "regex":
            self._load_regex_candidates()
            return self._regex_candidates
        elif parameter_name == "replacement":
            self._load_replacement_candidates()
            return self._replacement_candidates
        elif parameter_name == "s":
            self._load_reverse_candidates()
            return self._reverse_candidates
        else:
            self._load_prompt_words()
            return self._cached_prompt_words

    def _load_prompt_words(self) -> None:
        """Load all the words of the prompt in a class attribute."""
        if hasattr(self, "_cached_prompt_words"):
            return
        self._cached_prompt_words = set()
        self._cached_prompt_words.update(
            [w.strip("'\".,?!:;") for w in self.prompt.split()
             if w.strip("'\".,?!:;")])

    def load_functions_candidates(self) -> None:
        """Load all the potential function names depending of the prompt"""
        if hasattr(self, "_function_candidates"):
            return
        self._function_candidates = set()
        numbers = re.findall(r"\d+", self.prompt)
        for function in self.functions_def:
            params = function["parameters"]
            has_number_param = any(
                p["type"] == "number" for p in params.values()
            )
            if has_number_param and not numbers:
                continue
            self._function_candidates.add(function["name"])

    def _load_source_candidates(self) -> None:
        """Load all the potentially candidates for parameters
        of name 'source_string'.
        """
        if hasattr(self, "_source_candidates"):
            return
        self._source_candidates = set()
        prompt_lower = self.prompt.lower()
        match = re.search(
            r"in ?(?:the )?(?:string |phrase )?(\".+?\"|'.+?')",
            prompt_lower)
        if match:
            candidate = match.group(1).strip("'\"")
            pos = prompt_lower.find(candidate)
            if pos != -1:
                self._source_candidates.add(
                    self.prompt[pos:pos + len(candidate)])
        if not self._source_candidates:
            quotes = re.findall(r"'([^']*)'|\"([^\"]*)\"", self.prompt)
            values = [q[0] or q[1] for q in quotes if q[0] or q[1]]
            if values:
                self._source_candidates.add(max(values, key=len))

    def _load_regex_candidates(self) -> None:
        """Load all the potentially candidates for parameters
        of name 'regex'.
        """
        if hasattr(self, "_regex_candidates"):
            return
        self._regex_candidates = set()
        prompt_lower = self.prompt.lower()
        semantic = [
            (r'\bnumber[s]?\b',     r'\d+'),
            (r'\bdigit[s]?\b',      r'[0-9]'),
            (r'\bword[s]?\b',       r'\w+'),
            (r'\bletter[s]?\b',     r'[a-zA-Z]'),
            (r'\bspace[s]?\b',      r'\s'),
            (r'\bwhitespace\b',     r'\s'),
            (r'\bvowel[s]?\b',      r'[aeiouAEIOU]'),
            (r'\bconsonant[s]?\b',
             r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]'),
        ]
        for pattern, regex_val in semantic:
            if re.search(pattern, prompt_lower):
                self._regex_candidates.add(regex_val)
        self._load_replacement_candidates()
        quoted_words = re.findall(r"'(\w+)'|\"(\w+)\"", self.prompt)
        for quote in quoted_words:
            value = quote[0] or quote[1]
            if value and value not in self._replacement_candidates:
                self._regex_candidates.add(value)

    def _load_replacement_candidates(self) -> None:
        """Load all the potentially candidates for parameters
        of name 'replacement'.
        """
        if hasattr(self, "_replacement_candidates"):
            return
        self._replacement_candidates = set()
        words_next = re.findall(
            r"\b(?:with|by|to)\s+(\S+)", self.prompt
        )
        natural_to_symbol = {
            'underscores': '_',
            'asterisks':   '*',
            'hashes':      '#',
            'hash':        '#',
            'stars':       '*',
            'dashes':      '-',
            'dots':        '.',
        }
        for w in words_next:
            clean = w.strip().strip("'\"")
            translated = natural_to_symbol.get(clean.lower(), clean)
            self._replacement_candidates.add(translated)
            if translated != clean:
                self._replacement_candidates.add(clean)

    def _load_reverse_candidates(self) -> None:
        """Load all the potentially candidates for parameters
        of name 's'.
        """
        if hasattr(self, "_reverse_candidates"):
            return
        self._reverse_candidates = set()
        self._load_source_candidates()
        self._reverse_candidates.update(self._source_candidates)

        prompt_lower = self.prompt.lower()
        if "word" in prompt_lower:
            match = re.search(r"reverse ?(?:the )?(?:word )?(\S+)",
                              prompt_lower)
        else:
            match = re.search(
                r"reverse ?(?:the )?(?:string |phrase )?('.+'|\".+\"|.+)",
                prompt_lower)
        if match:
            candidate = match.group(1).strip("'\"")
            pos = prompt_lower.find(candidate)
            if pos != -1:
                self._reverse_candidates.add(
                    self.prompt[pos:pos + len(candidate)])
