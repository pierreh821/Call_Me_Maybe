import json
import math

from llm_sdk import Small_LLM_Model
from .models import Tool
import re


def decode_token_string(s: str) -> str:
    return s.replace(u"\u0120", ' ').replace(u"\u2581", ' ')


def is_valid_number_part(token_str: str) -> bool:
    """
    Vérifie si le token contient uniquement des caractères
    valides pour composer un nombre (chiffres, point décimal, signe moins,
    espaces).
    """
    # Si le token est vide, on l'ignore ou le rejette
    if not token_str:
        return False

    # On vérifie que le token ne contient QUE des chiffres, '.', '-' ou espaces
    return bool(re.match(r'^[ -]?[0-9.]+$', token_str))


def format_prompt(prompt: str, tools: list[Tool]) -> str:
    res = (
        "You are a function calling assistant."
        "You must respond ONLY with a JSON object matching this structure:"
        "{\"name\": \"<function_name>\", \"parameters\": {<args>}}"
        "Available functions:"
        )

    for tool in tools:
        param_str = ""
        for p_name, p_type in tool.parameters.items():
            param_str += f"{p_name} :{p_name}, "
        res += f" - {tool.name}({param_str}): {tool.description}"

    res += (
        f"User query: {prompt}"
        "Output JSON: {")

    return res


def algo(prompt: str, tools: list[Tool]) -> str:
    model = Small_LLM_Model()
    vocab_path = model.get_path_to_vocab_file()

    prompt = format_prompt(prompt, tools)
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw_vocab: dict[str, int] = json.load(f)

    id_to_token: dict[int, str] = {
        token_id: token_str for token_str, token_id in raw_vocab.items()}

    prompt_tensor = model.encode(prompt)

    if hasattr(prompt_tensor, "tolist"):
        current_input_ids: list[int] = prompt_tensor.tolist()
    else:
        current_input_ids = list(prompt_tensor)

    if isinstance(current_input_ids[0], list):
        current_input_ids = current_input_ids[0]

    generated_tokens: list[int] = []
    max_tokens = 150

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(current_input_ids)

        for token_id, token_str in id_to_token.items():
            clean_str = decode_token_string(token_str)

        next_token_id = int(max(range(len(logits)), key=lambda i: logits[i]))
        current_input_ids.append(next_token_id)
        generated_tokens.append(next_token_id)

    if hasattr(model, "decode"):
        return model.decode(generated_tokens)

    return "".join([decode_token_string(id_to_token[t]) for t in generated_tokens])
