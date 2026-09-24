from abc import ABC, abstractmethod
from pydantic import BaseModel, ValidationError
from typing import Any
import json

from .function_definition import FunctionDefinition


class PromptModel(BaseModel):
    prompt: str


class Parser(ABC):
    @abstractmethod
    def parse(cls, file: str) -> list[FunctionDefinition] | list[str]:
        pass

    @staticmethod
    def _load_json(file: str) -> Any:
        with open(file) as f:
            return json.load(f)


class ToolParser(Parser):
    @classmethod
    def parse(cls, file: str) -> list[FunctionDefinition]:
        try:
            raw_func_list = cls._load_json(file)
        except json.decoder.JSONDecodeError as e:
            raise SyntaxError(
                f"Invalid JSON syntax on functions definitions input:\n  {e}")

        if not isinstance(raw_func_list, list):
            raise SyntaxError("JSON error: root element must be a list")

        func_list: list[FunctionDefinition] = []
        for raw_func in raw_func_list:
            try:
                tool = FunctionDefinition(**raw_func)
                func_list.append(tool)
            except (ValidationError, ValueError, TypeError) as e:
                raise SyntaxError(f"JSON Error: invalid tool structure -> {e}")

        return func_list


class PromptParser(Parser):
    @classmethod
    def parse(cls, file: str) -> list[str]:
        try:
            raw_prompt_list = cls._load_json(file)
        except json.decoder.JSONDecodeError as e:
            raise SyntaxError(f"Invalid JSON syntax on prompts input:\n  {e}")

        if not isinstance(raw_prompt_list, list):
            raise SyntaxError("JSON Error: prompts root must be a list")

        prompt_list: list[str] = []
        for raw_prompt in raw_prompt_list:
            try:
                item = PromptModel(**raw_prompt)
                prompt_list.append(item.prompt)
            except (ValidationError, TypeError):
                raise SyntaxError("Prompt not provided or invalid format")

        return prompt_list
