from pydantic import BaseModel, PrivateAttr
from typing import Any
from llm_sdk import Small_LLM_Model
from .GenerationPipeline import GenerationPipeline
import json
import os


class PipelineManager(BaseModel):
    model_name: str
    prompts: list[dict[str, str]]
    functions_def: list[dict[str, Any]]

    _model: Small_LLM_Model = PrivateAttr(default_factory=Small_LLM_Model)
    _output: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    @property
    def output(self) -> list[dict[str, Any]]:
        return self._output

    def load_model(self) -> None:
        self._model = Small_LLM_Model(model_name=self.model_name)

    def generate_outputs(self, output_path: str) -> None:
        self.load_model()
        i = 1
        with open(output_path, 'w') as f:
            f.write('[]')
        for i in range(len(self.prompts)):
            print(f"[{i + 1}/{len(self.prompts)}] Generating response for "
                  f"\"{self.prompts[i]['prompt']}\" ...")
            gen = GenerationPipeline(
                model=self._model,
                prompt=self.prompts[i]['prompt'],
                functions_def=self.functions_def
            )
            gen.generate_output()
            self.output.append(gen.output)
            self.save_output(output_path=output_path,
                             to_save=gen.output)

    def save_output(
            self,
            output_path: str = "data/output/function_calling_results.json",
            to_save: dict[str, Any] = {}
            ) -> None:
        output_dir = os.path.dirname(output_path)
        if output_dir != '' and not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                raise ValueError(
                    f"Failed create directory '{output_dir}': {e}"
                    )
        try:
            with open(output_path, 'r') as f:
                content = list(json.loads(f.read()))
        except (FileNotFoundError, json.JSONDecodeError):
            content = []
        try:
            with open(output_path, 'w') as f:
                content.append(to_save)
                f.write(json.dumps(content, indent=2))
                print(f"Response saved in '{output_path}'")
        except Exception as e:
            raise ValueError(
                f"Failed to write the output in {output_path}: {e}")
