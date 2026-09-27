from typing import Any

from .functions import FunctionDef


def build_name_prompt(functions: list[FunctionDef],
                      query: str) -> str:
    """Build a prompt asking the model to choose an available function.

    Args:
        functions: Function definitions to present as candidates.
        query: User request the selected function should satisfy.

    Returns:
        A text prompt ending with the function-name generation prefix.
    """
    fn_list = ""
    for fn in functions:
        parameters = [f"{p_name}: {p_type.__name__}"
                      for p_name, p_type in fn.parameters.items()]

        param_str = ", ".join(parameters)
        fn_list += f"- {fn.name}({param_str}): {fn.description}\n"

    return (
        f"Available functions: \n{fn_list}"
        "You are a function calling assistant.\n"
        "Choose the most appropriate function from the provided options based "
        "on the user query.\n"
        f"User query: {query}.\n"
        "Function name: "
    )


def build_param_prompt(fn: FunctionDef, p_name: str, p_type: type,
                       query: str, filled: dict[str, Any]) -> str:
    """Build a prompt for extracting one function parameter.

    Args:
        fn: Selected function whose description guides extraction.
        p_name: Name of the parameter currently being generated.
        p_type: Python type declared for the parameter.
        query: Original user request.
        filled: Parameter values already generated for this function call.

    Returns:
        A prompt containing the request, relevant hints, and partial call.
    """
    call_so_far = f"{fn.name}("
    call_so_far += ", ".join(
        f'{k}={v!r}' if isinstance(v, str) else f"{k}={v}"
        for k, v in filled.items())
    if filled:
        call_so_far += ", "
    call_so_far += f"{p_name}="
    if p_type is str:
        call_so_far += '"'

    regex_hint = ""
    if p_name == "regex" or "regex" in fn.description.lower():
        regex_hint = (
            "# Common regex building blocks: \\d+ (digits), \\s+ (spaces), "
            "\\w+ (word chars), [abc] (character set), \\bword\\b "
            "(whole word)\n"
        )

    numbers_hint = ""
    if p_type in (int, float):
        numbers_hint = (
            "# Add two numbers together.\n"
            "# Query: \"What is the sum of -5 and 10?\"\n"
            "fn_add_numbers(a=-5, b=10\n\n"
            "# Add two numbers together.\n"
            "# Query: \"What is the sum of -3 and -8?\"\n"
            "fn_add_numbers(a=-3, b=-8\n\n"
        )

    extra_hint = ""
    name_lower = p_name.lower()
    if "path" in name_lower or "file" in name_lower:
        extra_hint = (
            "# Example: query mentions /var/log/app.log -> "
            "path=\"/var/log/app.log\n"
            "# Copy the full path exactly as written, slashes and drive "
            "prefixes included.\n"
        )

    return (
        f"# {fn.description}\n"
        f"{regex_hint}"
        f"{numbers_hint}"
        f"{extra_hint}"
        "# A minus sign directly before a digit means a negative number, "
        "not subtraction or punctuation.\n"
        f"# Query: \"{query}\"\n"
        f"{call_so_far}"
    )
