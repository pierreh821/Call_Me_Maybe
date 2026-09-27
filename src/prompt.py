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
        f"User query: '{query}'.\n"
        "Function name: "
    )


def build_param_prompt(fn: FunctionDef,
                       p_name: str,
                       p_type: type,
                       query: str) -> str:

    parameters = [f"{p_name}: {p_type.__name__}"
                  for p_name, p_type in fn.parameters.items()]

    return (
        "You are a function calling assistant.\n"
        f"The chosen function is '{fn.name}: {fn.description}'.\n"
        f"The function needs these parameters: {", ".join(parameters)}.\n"
        "Given the user query, give the value of the parameter: "
        f"'{p_name}'. Respect its type: '{p_type.__name__}'.\n"
        f"User query: {query}.\n"
        f"'{p_name}'s value: "
    )
