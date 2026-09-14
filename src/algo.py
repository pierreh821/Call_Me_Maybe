import json
import math

from llm_sdk import Small_LLM_Model
from torch import Tensor
from enum import Enum, auto
from .models import Tool
import re


def decode_token_string(s: str) -> str:
    return s.replace(u"\u2581", ' ').replace('Ġ', ' ')


def format_prompt(prompt: str, tools: list[Tool]) -> str:
    res = "Available functions:\n"

    for tool in tools:
        param_str = ", ".join([f'''{p_name}: {str(p_type).split("'")[1]}''' for p_name, p_type in tool.parameters.items()])
        res += f"- {tool.name}({param_str}): {tool.description}\n"

    res += (
        "\nYou are a function calling assistant who can only speak JSON."
        "You must respond ONLY with a JSON object matching this structure: "
        '{"name": "<function_name>", "parameters": {<args>}}\n'
    )

    res += (
        f"\nUser query: {prompt}\n"
        "Find the appropriate function giving the user query, give its name and parameters as required by the JSON syntax.\n"
        "Do not say anything outside of the JSON output.\n"
        "Output JSON: {"
        )
    print(f"prompt: {res}")
    return res

def tensor_to_token(tensor: Tensor) -> list[int]:
    res = tensor.tolist() if hasattr(tensor, "tolist") else list(tensor)

    while isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
        res = res[0]

    return res


def algo(prompt: str, tools: list[Tool]) -> str:
    model = Small_LLM_Model()

    prompt = format_prompt(prompt, tools)

    current_input_ids = tensor_to_token(model.encode(prompt))
    init_ids = tensor_to_token(model.encode('{'))

    generated_tokens: list[int] = list(init_ids)
    max_tokens = 100

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(current_input_ids)
        current_text = model.decode(generated_tokens)

        next_token_id = int(max(range(len(logits)), key=lambda i: logits[i]))

        current_input_ids.append(next_token_id)
        generated_tokens.append(next_token_id)

        if current_text.endswith("}}"):
            break

    return test_end_json("".join([model.decode(t) for t in generated_tokens]))

def test_end_json2(response: str):
    braces = 0
    for i, char in enumerate(response):
        braces += (char == '{')
        braces -= (char == '}')

        if i > 0 and braces == 0:
            break

    return response[:i+1]

def test_end_json(response: str):
    opened = response.count('{')
    closed = 0
    for i, char in enumerate(response):
        if char == '}':
            closed += 1
        if opened == closed:
            return response[:i+1]