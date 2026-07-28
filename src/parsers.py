from abc import ABC
from typing import Any
import json

from .models import Function


class Parser(ABC):
    @staticmethod
    def load_json(file: str) -> Any:
        with open(file) as f:
            return json.load(f)


class FunctionParser(Parser):
    @classmethod
    def parse(cls, file: str) -> list[Function]:
        raw_func_list = cls.load_json(file)
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


class PromptParser(Parser):
    @classmethod
    def parse(cls, file: str) -> list[str]:
        raw_prompt_list = cls.load_json(file)

        prompt_list: list[str] = []
        for raw_prompt in raw_prompt_list:
            if "prompt" not in raw_prompt.keys():
                raise SyntaxError("Prompt not provided")

            if not isinstance(raw_prompt["prompt"], str):
                raise SyntaxError("Prompt must be a string")

            prompt_list.append(raw_prompt["prompt"])

        return prompt_list
