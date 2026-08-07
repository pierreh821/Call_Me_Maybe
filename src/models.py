class TypeHandler:
    types = {
        "number": float,
        "string": str
        }


class Tool:
    def __init__(
            self,
            name: str,
            description: str,
            parameters: dict[str, dict[str, str]],
            returns: dict[str, str]) -> None:
        self.name = name
        self.description = description

        type_str = returns.get("type")
        if type_str is not None:
            self.returns: type | None = TypeHandler.types.get(type_str)
        else:
            raise SyntaxError("Type not specified in function returns")

        self.parameters: dict[str, type | None] = {}
        for item in parameters.items():
            param_name = item[0]
            type_str = item[1].get("type")
            if type_str is not None:
                type_cln = TypeHandler.types.get(type_str)
            else:
                raise SyntaxError("Type not specified in function returns")
            self.parameters[param_name] = type_cln

    @staticmethod
    def is_valid(raw: dict) -> bool:
        if not (
                "name" in raw.keys()
                and "description" in raw.keys()
                and "parameters" in raw.keys()
                and "returns" in raw.keys()):
            return False

        if not (
            isinstance(raw.get("parameters"), dict)
            and isinstance(raw.get("returns"), dict)
        ):
            return False

        if len(raw.get("parameters", {})) > 0:
            for opt in raw.get("parameters", {}).items():
                if not isinstance(opt[1], dict):
                    return False

        if len(raw.get("returns", {})) > 0:
            for opt in raw.get("returns", {}).items():
                if not isinstance(opt[1], str):
                    return False

        return True

    @staticmethod
    def print_list(func_list: list["Tool"]) -> None:
        for fn in func_list:
            print(f"\n{fn.name}: {fn.description} Returns "
                  f"{str(fn.returns).split("'")[1]}")
            print("Parameters:")
            for p in fn.parameters.items():
                print(f"{p[0]}: {str(p[1]).split("'")[1]}")
