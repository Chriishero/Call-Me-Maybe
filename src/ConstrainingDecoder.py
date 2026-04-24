from .Input import Function, Type
from llm_sdk import Small_LLM_Model
from pydantic import BaseModel
from typing import Any
import re


class ConstrainingDecoder(BaseModel):
    """
    Decoder applying token-level constraints to ensure that
    generated function names and parameters are adapted to
    the prompt.
    """
    model_config = {
        "arbitrary_types_allowed": True
    }
    model: Small_LLM_Model
    prompt: str
    functions: list[Function]
    vocabulary: dict[int, str]
    tokenizer: dict[str, Any]
    max_tokens: int

    def constrain_function(self, logits: list[float]) -> Function:
        available_tokens_id = {
            k for k, v in self.vocabulary.items()
            if v in [str(i) for i in range(len(self.functions))]
        }
        masked_logits = [
            logit if idx in available_tokens_id else float("-inf")
            for idx, logit in enumerate(logits)
        ]
        predicted_fun_idx = int(
            self.vocabulary[
                masked_logits.index(max(masked_logits))
            ]
        )
        return self.functions[predicted_fun_idx]

    def constrain_parameter(
            self,
            tokens: list[int],
            param_type: Type,
            eos_token_id: int
    ) -> tuple[str | int | float | bool | None, list[int]]:
        if param_type == Type.string:
            return self.generate_string(
                tokens=tokens,
                eos_token_id=eos_token_id
            )
        elif param_type == Type.number \
                or param_type == Type.integer:
            return self.generate_number(
                tokens=tokens,
                param_type=param_type,
                eos_token_id=eos_token_id
            )
        elif param_type == Type.boolean:
            return self.generate_boolean(
                tokens=tokens
            )
        return None, tokens

    def generate_string(
            self,
            tokens: list[int],
            eos_token_id: int
    ) -> tuple[str, list[int]]:
        prompt_len = len(tokens)
        for _ in range(100):
            logits = self.model.get_logits_from_input_ids(tokens)
            next_token = logits.index(max(logits))
            if next_token == eos_token_id:
                break
            tokens.append(next_token)
            generated = self.model.decode(tokens[prompt_len:])
            if generated.count('"') >= 2 or generated.count("'") >= 2:
                quote = '"' if generated.count('"') >= 2 else "'"
                start = generated.index(quote) + 1
                end = generated.index(quote, start)
                value = generated[start:end].strip()
                value = value.replace("\\\\", "\\")
                matches = list(
                    re.finditer(re.escape(value), self.prompt, re.IGNORECASE)
                )
                if matches:
                    value = max(
                        matches,
                        key=lambda m: sum(c.isupper()for c in m.group(0))
                    ).group(0)
                return value, tokens
        return "", tokens

    def generate_number(
            self,
            tokens: list[int],
            param_type: Type,
            eos_token_id: int
    ) -> tuple[int | float, list[int]]:
        allowed_tokens = set("0123456789.-+eE")
        valid_tokens = {
            k for k, v in self.vocabulary.items()
            if v and all(c in allowed_tokens for c in v)
        }
        value_str = ""
        for _ in range(self.max_tokens):
            logits = self.model.get_logits_from_input_ids(tokens)
            next_token = logits.index(max(logits))
            if next_token not in valid_tokens or next_token == eos_token_id:
                break
            tokens.append(next_token)
            value_str += self.vocabulary[next_token]
        try:
            value = (
                int(float(value_str))
                if param_type == Type.integer
                else float(value_str)
            )
            if str(value) not in self.prompt:
                candidates = re.findall(r"\d+(?:\.\d+)?", self.prompt)
                for candidate in candidates:
                    candidate_value = (
                        int(float(candidate))
                        if param_type == Type.integer
                        else float(value_str)
                    )
                    if candidate_value != value:
                        return candidate_value, tokens
            return value, tokens
        except ValueError:
            return 0, tokens

    def generate_boolean(
            self,
            tokens: list[int],
    ) -> tuple[bool, list[int]]:
        true_tokens = [
            k for k, v in self.vocabulary.items() if v == "true"
        ]
        false_tokens = [
            k for k, v in self.vocabulary.items() if v == "false"
        ]
        logits = self.model.get_logits_from_input_ids(tokens)
        true_score = max(
            (logits[k] for k in true_tokens), default=float("-inf")
        )
        false_score = max(
            (logits[k] for k in false_tokens), default=float("-inf")
        )
        value = true_score >= false_score
        tokens += self.model.encode(str(value).lower()).int().tolist()[0]
        return value, tokens
