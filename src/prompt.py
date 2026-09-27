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


EXAMPLE = (
    "Examples:\n"
    "Function: fn_reverse_string(s: str): Reverse a string.\n"
    "User query: Reverse the string 'abc'\n"
    "Value of 's' (str): abc\n\n"
    "Function: fn_greet(name: str): Greet a person by name.\n"
    "User query: Greet alice\n"
    "Value of 'name' (str): alice\n\n"
    "Function: fn_substitute_string_with_regex(source_string: str, regex: str,"
    " replacement: str): Replace regex matches in a string.\n"
    "User query: Replace all numbers in \"abc 123 def\" with X\n"
    "Value of 'source_string' (str): abc 123 def\n"
    "Value of 'regex' (str): \\d+\n"
    "Value of 'replacement' (str): X\n\n"
    "User query: Replace all vowels in 'sky' with *\n"
    "Value of 'regex' (str): [aeiouAEIOU]\n\n"
    "User query: Substitute the word 'foo' with 'bar' in 'foo is foo'\n"
    "Value of 'regex' (str): \\bfoo\\b\n\n"
)


def build_param_prompt(fn: FunctionDef,
                       p_name: str,
                       p_type: type,
                       query: str,
                       filled: dict[str, Any]) -> str:

    parameters = [f"{p_name}: {p_type.__name__}"
                  for p_name, p_type in fn.parameters.items()]

    filled_str = ""
    if filled:
        filled_lines = "\n".join(f"  {k} = {v!r}" for k, v in filled.items())
        filled_str = f"Already determined:\n{filled_lines}\n"

    regex_hint = ""
    if p_name == "regex":
        regex_hint = (
            "This value must be a regular expression pattern (using "
            "syntax like \\d, \\s, [abc], \\b), never a literal word or "
            "number copied from the query.\n"
        )

    return (
        "You are a function calling assistant.\n"
        f"{EXAMPLE}"
        f"The chosen function is '{fn.name}: {fn.description}'.\n"
        f"The function needs these parameters: {', '.join(parameters)}.\n"
        f"{filled_str}"
        f"{regex_hint}"
        "Give the value of the remaining parameter below, matching its "
        "role exactly. Answer with the raw value only — no quotes, no "
        "explanation.\n"
        f"User query: {query}\n"
        f"Value of '{p_name}' ({p_type.__name__}): "
    )
