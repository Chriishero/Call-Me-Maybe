from .ConstrainingDecoder import ConstrainingDecoder
from .Input import Function, Prompt
from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, PrivateAttr, Field
from typing import Any, Optional
import json


class GenerationPipeline(BaseModel):
    """Runs the LLM pipeline and apply constrained decoding
    before token selection from the given prompt, functions definition
    and model.
    Then save the function-calling output in an attribute."""
    model_config = {
        "arbitrary_types_allowed": True
    }
    model: Small_LLM_Model
    prompt: Prompt
    functions_definition: list[Function]
    max_tokens: int = Field(default=64)

    _output: dict[str, Any] = PrivateAttr(default_factory=dict)
    _function: Optional[Function] = PrivateAttr(default=None)
    _vocabulary: dict[int, str] = PrivateAttr(default_factory=dict)
    _tokenizer: dict[str, Any] = PrivateAttr(default_factory=dict)
    _parameters_types: list[Any] = PrivateAttr(default_factory=list)

    @property
    def output(self) -> dict[str, Any]:
        """Getter for the class attribute '_output'"""
        return self._output

    def generate_output(self) -> None:
        """Generate the function-calling output by calling
        the two adequates functions for function name and parameters."""
        self.load_vocabulary()
        self.load_tokenizer_file()
        decoder = ConstrainingDecoder(
            model=self.model,
            prompt=self.prompt.prompt,
            functions=self.functions_definition,
            vocabulary=self._vocabulary,
            tokenizer=self._tokenizer,
            max_tokens=self.max_tokens
        )
        self._output = {
            "prompt": self.prompt.prompt,
            "name": "",
            "parameters": {}
        }
        self.generate_function_name(decoder)
        self.generate_parameters(decoder)

    def generate_function_name(self, decoder: ConstrainingDecoder) -> None:
        """Execute the LLM pipeline and apply constrained decoding before
        token selection, to generate the function name."""
        f_prompt = self.get_function_system_prompt()
        tokens = self.model.encode(f_prompt)[0].tolist()
        logits: list[float] = self.model.get_logits_from_input_ids(
            tokens
        )
        self._function = decoder.constrain_function(logits)
        self._output['name'] = self._function.name

    def generate_parameters(self, decoder: ConstrainingDecoder) -> None:
        """Execute the LLM pipeline and apply constrained decoding to
        generate the parameters of the generated function."""
        if self._function is None:
            return
        p_prompt = self.get_parameters_system_prompt()
        tokens = self.llm_encode(p_prompt)
        eos_token_id = self._tokenizer["added_tokens"][-1]["id"]
        for i, parameter in enumerate(self._function.parameters):
            tokens += self.llm_encode(f'"{parameter.name}": ')
            value, tokens = decoder.constrain_parameter(
                tokens=tokens,
                param_type=parameter.type,
                eos_token_id=eos_token_id,
            )
            self._output['parameters'][parameter.name] = value
            if i < len(self._function.parameters) - 1:
                tokens += self.llm_encode(", ")

    def get_function_system_prompt(self) -> str:
        """Return the function system prompt."""
        functions_str = "\n".join(
            f"[{i}] {fun.name}: {fun.description}"
            for i, fun in enumerate(self.functions_definition)
        )
        examples_str = "\n".join(
            f"Request: '{fun.name} example' -> [{i}]"
            for i, fun in enumerate(self.functions_definition)
        )
        return (f"Task: Match the request to the correct function index.\n\n"
                f"Functions:\n{functions_str}\n\n"
                f"Examples:\n{examples_str}\n\n"
                f"Request: {self.prompt.prompt}\n\n"
                f"The index of the most appropriate function is ["
                )

    def get_parameters_system_prompt(self) -> str:
        """Return the parameters system prompt."""
        if self._function is None:
            return ""
        params_str = "\n".join(
            f"{param.name}: {param.type}"
            for param in self._function.parameters
        )
        return (f"Task: Extract the parameters from the request.\n\n"
                f"Function: {self._function.name}\n"
                f"Parameters:\n{params_str}\n\n"
                f"Request: {self.prompt.prompt}\n\n"
                f"Rules:\n"
                f"- Preserve ALL characters from the request exactly,"
                f" including quotes.\n"
                f"- For regex parameters: use character classes like [0-9] or "
                f"[aeiouAEIOU] instead of literal values or alternations.\n"
                f'- Escaped quotes in the request like \\"word\\" must stay'
                f' as \\"word\\" in the output.\n\n'
                f"Output ONLY a valid JSON with the parameter values.\n"
                f"JSON: {{"
                )

    def llm_encode(self, text: str) -> Any:
        return self.model.encode(text).int().tolist()[0]

    def load_vocabulary(self) -> None:
        """Load the vocabulary using by the LLM in '_vocabulary'."""
        try:
            with open(self.model.get_path_to_vocab_file(), 'r') as f:
                vocab = json.loads(f.read())
                self._vocabulary: dict[int, str] = {
                    v: k for k, v in vocab.items()
                }
        except FileNotFoundError as e:
            raise ValueError(f"Missing vocabulary file: {e}")

    def load_tokenizer_file(self) -> None:
        """Load the tokenizer file using by te LLM in '_tokenizer'."""
        try:
            with open(self.model.get_path_to_tokenizer_file()) as f:
                self._tokenizer = json.loads(f.read())
        except Exception as e:
            raise ValueError(f"Missing tokenizer file: {e}")
