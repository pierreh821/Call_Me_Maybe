from pydantic import ValidationError
from typing import Any
from pathlib import Path
import json

from .function_definition import FunctionDefinition
from .models import PromptInput, FunctionCallResult


class InputError(Exception):
    pass


def load_json(file: Path) -> Any:
    try:
        with open(file, encoding="utf-8") as f:
            return json.load(f)
    except OSError as e:
        raise InputError(f"Cannot read '{file}': {e.strerror}") from e
    except UnicodeDecodeError as e:
        raise InputError(f"'{file}' is not valid UTF-8 text.") from e
    except json.JSONDecodeError as e:
        raise InputError(f"Invalid JSON syntax in '{file}':\n  {e}") from e


def load_prompts(file: Path) -> list[str]:
    try:
        raw_prompt_list = load_json(file)
    except json.decoder.JSONDecodeError as e:
        raise InputError(f"Invalid JSON syntax on prompts input:\n  {e}")

    if not isinstance(raw_prompt_list, list):
        raise InputError("JSON Error: prompts root must be a list")

    prompt_list: list[str] = []
    for raw_prompt in raw_prompt_list:
        try:
            item = PromptInput(**raw_prompt)
            prompt_list.append(item.prompt)
        except (ValidationError, TypeError):
            raise InputError("Prompt not provided or invalid format")

    return prompt_list


def load_functions(file: Path) -> list[FunctionDefinition]:
    try:
        raw_func_list = load_json(file)
    except json.decoder.JSONDecodeError as e:
        raise InputError(
            f"Invalid JSON syntax on functions definitions input:\n  {e}")

    if not isinstance(raw_func_list, list):
        raise InputError("JSON error: root element must be a list")

    func_list: list[FunctionDefinition] = []
    for raw_func in raw_func_list:
        try:
            tool = FunctionDefinition(**raw_func)
            func_list.append(tool)
        except (ValidationError, ValueError, TypeError) as e:
            raise InputError(f"JSON Error: invalid tool structure -> {e}")

    return func_list


def save_result(file: Path, output: list[FunctionCallResult]) -> None:
    file.parent.mkdir(exist_ok=True, parents=True)
    data = [item.model_dump()for item in output]
    file.write_text(json.dumps(data, indent=4), encoding="utf-8")
