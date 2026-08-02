import json
import math

from llm_sdk import Small_LLM_Model


def decode_token_string(s: str) -> str:
    return s.replace(u"\u0120", ' ').replace(u"\u2581", ' ')

import re

def is_valid_number_part(token_str: str) -> bool:
    """
    Vérifie si le token contient uniquement des caractères
    valides pour composer un nombre (chiffres, point décimal, signe moins, espaces).
    """
    # Si le token est vide, on l'ignore ou le rejette
    if not token_str:
        return False

    # On vérifie que le token ne contient QUE des chiffres, '.', '-' ou espaces
    return bool(re.match(r'^[ -]?[0-9.]+$', token_str))


def idk() -> None:
    model = Small_LLM_Model()
    vocab_path = model.get_path_to_vocab_file()

    with open(vocab_path, "r", encoding="utf-8") as f:
        raw_vocab: dict[str, int] = json.load(f)

    id_to_token: dict[int, str] = {token_id: token_str for token_str, token_id in raw_vocab.items()}

    logits = model.get_logits_from_input_ids(current_input_ids)

    for token_id, token_str in id_to_token.items():
        clean_str = decode_token_string(token_str)

        if not is_valid_number_part(clean_str):
            logits[token_id] = -math.inf
