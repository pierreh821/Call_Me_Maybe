import json


class Function:
    def __init__(
            self,
            name: str,
            description: str,
            parameters: dict[str, str],
            returns: dict[str, str]) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.returns = returns

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
                if not isinstance(opt, dict):
                    return False

        if len(raw.get("returns", {})) > 0:
            for opt in raw.get("returns", {}).items():
                if not isinstance(opt, str):
                    return False

        return True


class Parser:
    @staticmethod
    def parse(file: str) -> list[Function]:
        with open(file) as f:
            raw_func_list = json.load(f)

        if not isinstance(raw_func_list, list):
            raise SyntaxError("JSON error")

        func_list: list[Function] = []
        for raw_func in raw_func_list:
            # if not Function.is_valid(raw_func):
            #     raise SyntaxError("JSON Error")

            func_list.append(Function(
                raw_func["name"],
                raw_func["description"],
                raw_func["parameters"],
                raw_func["returns"]
            ))

        return func_list
