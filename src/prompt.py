from .functions import FunctionDef


def build_prompt(functions: list[FunctionDef], prompt: str) -> str:
    """Build the model prompt from available functions and a user query.

    Args:
        functions: Function definitions to present to the model.
        prompt: User's natural-language request.

    Returns:
        A prompt requesting a structured function call.
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
        "Produce the output as a JSON with a top-level key called 'name' "
        "and a second-level key named 'parameters' like this:\n"
        '{"name": "<function_name>", "parameters": {<args>}}\n'
        "Preserve ALL exact quotes, punctuation, and characters from the "
        "user query verbatim in string parameters.\n"
        "If the user query contains quotes inside text, escape them "
        "properly with \\\" inside the JSON string values.\n"
        "Do not use parameters that are not given in the available "
        "functions.\n"
        "The output ends when the JSON is properly closed.\n"
        f"User query: {prompt}\n"
        "Clean JSON output: {"
    )
