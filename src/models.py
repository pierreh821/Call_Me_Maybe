from pydantic import BaseModel
from typing import Any
from functions import FunctionDef


class PromptInput(BaseModel):
    """Validated input item containing one natural-language prompt."""

    prompt: str


class FunctionCallResult(BaseModel):
    """Validated function-call result written to the output JSON file."""

    prompt: str
    function: FunctionDef
    parameters: dict[str, Any]
