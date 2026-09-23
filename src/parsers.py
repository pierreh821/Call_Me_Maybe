from abc import ABC, abstractmethod
from typing import Any
import json

from .tools import Tool


class Parser(ABC):
    @abstractmethod
    def parse(cls, file: str) -> list[Tool] | list[str]:
        pass

    @staticmethod
    def _load_json(file: str) -> Any:
        with open(file) as f:
            return json.load(f)


class ToolParser(Parser):
    @classmethod
    def parse(cls, file: str) -> list[Tool]:
        raw_func_list = cls._load_json(file)
        if not isinstance(raw_func_list, list):
            raise SyntaxError("JSON error")

        func_list: list[Tool] = []
        for raw_func in raw_func_list:
            if not Tool.is_valid(raw_func):
                raise SyntaxError("JSON Error")

            func_list.append(Tool(
                raw_func["name"],
                raw_func["description"],
                raw_func["parameters"],
                raw_func["returns"]
            ))

        return func_list


class PromptParser(Parser):
    @classmethod
    def parse(cls, file: str) -> list[str]:
        raw_prompt_list = cls._load_json(file)

        prompt_list: list[str] = []
        for raw_prompt in raw_prompt_list:
            if "prompt" not in raw_prompt.keys():
                raise SyntaxError("Prompt not provided")

            if not isinstance(raw_prompt["prompt"], str):
                raise SyntaxError("Prompt must be a string")

            prompt_list.append(raw_prompt["prompt"])

        return prompt_list
