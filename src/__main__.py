from .InputParser import InputParser
from .PipelineManager import PipelineManager
from pydantic import ValidationError
import sys


def is_path(path: str) -> bool:
    """Check if the argument is a path or not."""
    return not path.startswith("--")


def get_all_path() -> tuple[str, str, str]:
    """Get input and output files path from the arguments."""
    argv = sys.argv[1:]
    function_calling_tests = "data/input/function_calling_tests.json"
    functions_definition = "data/input/functions_definition.json"
    function_calling_results = "data/output/function_calling_results.json"
    for i in range(0, len(argv), 1):
        if argv[i] == "--input":
            function_calling_tests = (argv[i + 1] if is_path(argv[i + 1])
                                      else function_calling_tests)
        elif argv[i] == "--functions_definition":
            functions_definition = (argv[i + 1] if is_path(argv[i + 1])
                                    else functions_definition)
        elif argv[i] == "--output":
            function_calling_results = (argv[i + 1] if is_path(argv[i + 1])
                                        else function_calling_results)
    return (
        function_calling_tests, functions_definition, function_calling_results
        )


def main() -> None:
    """Main function"""
    input_path, funs_def, output_path = get_all_path()
    try:
        parser = InputParser(
            fun_call_tests_path=input_path,
            funs_def_path=funs_def
        )
        manager = PipelineManager(
            model_name="Qwen/Qwen3-0.6B",
            prompts=parser.function_calling_tests,
            functions_def=parser.functions_definition
        )
        manager.generate_outputs(
            output_path=output_path
        )
    except ValidationError as e:
        print(e.errors()[0]["msg"].removeprefix("Value error, "))


if __name__ == "__main__":
    main()
