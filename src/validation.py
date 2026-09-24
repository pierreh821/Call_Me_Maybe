from typing import Any

from .function_definition import FunctionDefinition
from .models import FunctionCallResult


def _find_function(ref: str | None, functions: list[FunctionDefinition]
                   ) -> FunctionDefinition | None:
    """Find a function by its exact name or optional ``fn_`` prefix.

    Args:
        ref: Function name returned by the model.
        functions: Available function definitions.

    Returns:
        The matching definition, or ``None`` when no match exists.
    """
    if not isinstance(ref, str):
        return None

    for fn in functions:
        if fn.name in (ref, 'fn_' + ref):
            return fn

    return None


def _coerce(value: Any, expected: type) -> Any:
    """Convert value to the expected type or raise ValueError/TypeError."""

    if expected is int:
        if isinstance(value, int):
            return value

        as_float = float(value)
        if not as_float.is_integer():
            raise ValueError("not an integer")
        return int(as_float)

    return expected(value)


def normalize_call(raw: Any, functions: list[FunctionDefinition],
                   prompt: str
                   ) -> tuple[FunctionCallResult | None, list[str]]:
    """Validate and normalize one raw model function call.

    Args:
        raw: JSON-decoded model output.
        functions: Available function definitions.
        prompt: Original user prompt associated with the output.

    Returns:
        A validated result and an empty error list, or ``None`` and errors.
    """

    if not isinstance(raw, dict):
        return None, ["The model output is not a JSON object."]

    fn = _find_function(raw.get("name"), functions)
    if fn is None:
        return None, [f"Unknown function name: {raw.get('name')}."]

    raw_params = raw.get("parameters")
    if not isinstance(raw_params, dict):
        return None, [f"Using {fn.name}, 'parameters' must be an object."]

    parameters: dict[str, Any] = {}
    errors: list[str] = []

    for p_name, p_type in fn.parameters.items():
        if p_name not in raw_params:
            errors.append(f"Using {fn.name}, missing parameter '{p_name}'.")
            continue

        p_val = raw_params[p_name]
        try:
            parameters[p_name] = _coerce(p_val, p_type)
        except (ValueError, TypeError):
            errors.append(
                f"Using {fn.name}, cannot convert parameter '{p_name}' "
                f"value: '{p_val}' ({type(p_val).__name__}, "
                f"expected {p_type.__name__}). Maybe check your prompt?")

    if errors:
        return None, errors

    return FunctionCallResult(
        prompt=prompt, name=fn.name, parameters=parameters), []
