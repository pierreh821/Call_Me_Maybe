from pydantic import BaseModel
from typing import Any


class PromptInput(BaseModel):
    prompt: str


class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
