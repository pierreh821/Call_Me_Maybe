from typing import Callable, Optional, Any
from tqdm import tqdm  # type: ignore
from functools import partial

from llm_sdk import Small_LLM_Model  # type: ignore

from .constraints import (TokenConstraint, ChoiceConstraint,
                          NumericalConstraint, RawTextConstraint)
from .models import FunctionCallResult
from .prompt import build_name_prompt, build_param_prompt
from .functions import FunctionDef
from .generator import Generator
from .vocab import Vocab


class FunctionCaller:
    """Generate and validate function calls for a collection of prompts."""

    def __init__(self,
                 model: Small_LLM_Model,
                 vocab: Vocab,
                 functions: list[FunctionDef]) -> None:
        """Initialize a caller with the functions available to the model.

        Args:
            functions: Function definitions that may be selected.
        """
        self.vocab = vocab
        self.functions = functions
        self.generator = Generator(model, vocab)
        self._constraint_factories: dict[type, Callable[[], TokenConstraint]] \
            = {
            float: partial(NumericalConstraint, vocab, allow_float=True),
            int: partial(NumericalConstraint, vocab, allow_float=False),
            str: partial(RawTextConstraint, vocab),
            }

    def run(self,
            prompts: list[str],
            on_result: Optional[Callable[[list[FunctionCallResult]], None]]
            = None
            ) -> tuple[list[FunctionCallResult], dict[str, list[str]]]:
        """Process prompts and return valid calls together with errors.

        Args:
            prompts: Natural-language prompts to process.
            on_result: Optional callback invoked after each valid result.

        Returns:
            A tuple containing valid calls and errors grouped by prompt.
        """

        results: list[FunctionCallResult] = []
        errors: dict[str, list[str]] = {}

        for prompt in tqdm(prompts, desc="Prompt processing"):
            errors[prompt] = []
            name_prompt = build_name_prompt(self.functions, prompt)

            name = self.generator.generate(
                name_prompt, ChoiceConstraint(
                    self.vocab, [fn.name for fn in self.functions]))

            fn = next((f for f in self.functions if f.name == name), None)
            if fn is None:
                errors[prompt].append(f"Unknown function name: {name!r}.")
                continue

            parameters: dict[str, Any] = {}
            param_errors: list[str] = []

            for p_name, p_type in fn.parameters.items():
                constraint = self._constraint_for(p_type)
                if constraint is None:
                    param_errors.append(
                        f"Using {fn.name}, unsupported type for '{p_name}': "
                        f"{p_type.__name__}.")
                    continue

                value_prompt = build_param_prompt(fn, p_name, p_type, prompt,
                                                  parameters)
                raw_value = self.generator.generate(value_prompt, constraint)
                if p_type is str:
                    raw_value = self._strip_quotes(raw_value)

                try:
                    parameters[p_name] = self._convert(raw_value, p_type)
                except (ValueError, TypeError):
                    param_errors.append(
                        f"Using {fn.name}, cannot convert parameter '{p_name}'"
                        f" value: {raw_value!r} to {p_type.__name__}.")
                    parameters[p_name] = "" if p_type is str else 0

            if param_errors:
                errors[prompt] += param_errors

            result = FunctionCallResult(prompt=prompt, name=fn.name,
                                        parameters=parameters)
            results.append(result)
            if on_result:
                on_result(results)

        return results, errors

    def _constraint_for(self, spec: type) -> TokenConstraint | None:
        factory = self._constraint_factories.get(spec)
        return factory() if factory else None

    @staticmethod
    def _strip_quotes(raw: str) -> str:
        raw = raw.strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
            return raw[1:-1]
        return raw

    @staticmethod
    def _convert(raw_value: str, p_type: type) -> Any:
        cleaned = raw_value.strip()
        if not cleaned:
            raise ValueError("empty generated value")

        if p_type is int:
            as_float = float(cleaned)
            if not as_float.is_integer():
                raise ValueError(f"{cleaned!r} is not an integer")
            return int(as_float)

        if p_type is float:
            return float(cleaned)

        return p_type(cleaned)
