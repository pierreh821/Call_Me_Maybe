from typing import Any

from .functions import FunctionDef


def build_name_prompt(functions: list[FunctionDef],
                      query: str) -> str:
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
        f"{extra_hint}"
        f"# Query: \"{query}\"\n"
        f"{call_so_far}"
    )
