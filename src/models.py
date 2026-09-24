from pydantic import BaseModel
from typing import Any


class PromptInput(BaseModel):
    """Validated input item containing one natural-language prompt."""

    prompt: str


class FunctionCallResult(BaseModel):
    """Validated function-call result written to the output JSON file."""

    prompt: str
    name: str
    parameters: dict[str, Any]
