import json
from .function import Function


class Parser:
    @staticmethod
    def parse(file: str) -> list[Function]:
        with open(file) as f:
            raw_func_list = json.load(f)

        if not isinstance(raw_func_list, list):
            raise SyntaxError("JSON error")

        func_list: list[Function] = []
        for raw_func in raw_func_list:
            if not Function.is_valid(raw_func):
                raise SyntaxError("JSON Error")

            func_list.append(Function(
                raw_func["name"],
                raw_func["description"],
                raw_func["parameters"],
                raw_func["returns"]
            ))

        return func_list
