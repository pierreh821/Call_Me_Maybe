import json
from typing import Callable, Optional
from tqdm import tqdm

from .models import FunctionCallResult
from .prompt import build_prompt
from .function_definition import FunctionDefinition
from .generator import JsonGenerator
from .validation import normalize_call


class FunctionCaller:
    def __init__(self, functions: list[FunctionDefinition]) -> None:
        self.functions = functions
        self.generator = JsonGenerator()

    def run(self,
            prompts: list[str],
            on_result: Optional[Callable[[list[FunctionCallResult]], None]]
            = None
            ) -> tuple[list[FunctionCallResult], dict[str, list[str]]]:

        results: list[FunctionCallResult] = []
        errors: dict[str, list[str]] = {}

        for prompt in tqdm(prompts, desc="Prompt processing"):
            errors[prompt] = []
            raw_str = self.generator.generate(build_prompt(
                self.functions, prompt))

            try:
                raw = json.loads(raw_str)
            except json.decoder.JSONDecodeError:
                errors[prompt].append(
                    "Invalid JSON, its result will be omitted.")
                continue

            result, call_errors = normalize_call(raw, self.functions, prompt)
            errors[prompt] += call_errors

            if result is not None:
                results.append(result)
                if on_result:
                    on_result(results)

        return results, errors
