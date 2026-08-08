import json
import math

from llm_sdk import Small_LLM_Model
from torch import Tensor
from enum import Enum, auto
from .models import Tool
import re


class State(Enum):
    EXPECT_NAME_KEY = auto()
    EXPECT_COLON_1 = auto()
    EXPECT_FUNC_NAME = auto()
    EXPECT_COMMA_1 = auto()
    EXPECT_PARAMS_KEY = auto()
    EXPECT_COLON_2 = auto()
    EXPECT_START_PARAMS = auto()
    INSIDE_PARAMS = auto()
    EXPECT_END_OBJ = auto()


class FunctionCallFSM:
    def __init__(self,
                 tools: list[Tool],
                 vocab: Vocab,
                 id_to_token: dict[int, str]) -> None:
        self.state = State.EXPECT_NAME_KEY
        self.vocab = vocab
        self.id_to_token = id_to_token
        self.valid_func_names = {f.name for f in tools}
        self.param_depth = 0

    def get_allowed_tokens(self, current_text: str) -> set[int] | None:
        clean_text = current_text.replace(' ', '').replace('\n', '')

        match self.state:
            case State.EXPECT_NAME_KEY:
                if '"name"' in clean_text:
                    self.state = State.EXPECT_COLON_1

            case State.EXPECT_COLON_1:
                if '"name":' in clean_text:
                    self.state = State.EXPECT_FUNC_NAME

            case State.EXPECT_FUNC_NAME:
                for func_name in self.valid_func_names:
                    if f'"name":"{func_name}"' in clean_text:
                        self.state = State.EXPECT_COMMA_1
                        break

            case State.EXPECT_COMMA_1:
                if ',"parameters"' in clean_text:
                    self.state = State.EXPECT_COLON_2
                elif ',' in clean_text[clean_text.rfind(func_name) if 'func_name' in locals() else 0:]:
                    self.state = State.EXPECT_PARAMS_KEY

            case State.EXPECT_PARAMS_KEY:
                if '"parameters"' in clean_text:
                    self.state = State.EXPECT_COLON_2

            case State.EXPECT_COLON_2:
                if '"parameters":' in clean_text:
                    self.state = State.EXPECT_START_PARAMS

            case State.EXPECT_START_PARAMS:
                if '"parameters":{' in clean_text:
                    self.state = State.INSIDE_PARAMS
                    self.param_depth = 1

        return self._get_mask_for_state()

    def _get_mask_for_state(self) -> set[int] | None:
        if self.state == State.EXPECT_FUNC_NAME:
            allowed: set = set()
            for token_tuple in self.vocab.function_name_tokens:
                allowed.update(token_tuple)
            allowed.update({t for t, s in self.id_to_token.items()
                            if '"' in s})
            return allowed

        if self.state in (State.EXPECT_COLON_1, State.EXPECT_COLON_2):
            return {t for t, s in self.id_to_token.items() if ':' in s}

        if self.state == State.EXPECT_COMMA_1:
            return {t for t, s in self.id_to_token.items() if ',' in s}

        if self.state == State.EXPECT_START_PARAMS:
            return {t for t, s in self.id_to_token.items() if '{' in s}

        if self.state == State.INSIDE_PARAMS:
            return (self.vocab.alpha_tokens
                    | self.vocab.digits_tokens
                    | self.vocab.punct_tokens)

        return None


def decode_token_string(s: str) -> str:
    return s.replace(u"\u2581", ' ').replace('Ġ', ' ')




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


class Vocab:
    def __init__(self) -> None:
        self.digits_tokens: set[int] = set()
        self.alpha_tokens: set[int] = set()
        self.punct_tokens: set[int] = set()
        self.function_name_tokens: set[tuple[int, ...]] = set()


def sort_vocab(id_to_token: dict[int, str]) -> Vocab:
    vocab = Vocab()

    for token_id, token_str in id_to_token.items():
        clean = decode_token_string(token_str).strip()

        if clean.isnumeric():
            vocab.digits_tokens.add(token_id)

        if clean.isalpha():
            vocab.alpha_tokens.add(token_id)

    return vocab


def tensor_to_token(tensor: Tensor) -> list[int]:
    res = tensor.tolist() if hasattr(tensor, "tolist") else list(tensor)

    while isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
        res = res[0]

    return res


def algo(prompt: str, tools: list[Tool]) -> str:
    model = Small_LLM_Model()
    vocab_path = model.get_path_to_vocab_file()

    prompt = format_prompt(prompt, tools)
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw_vocab: dict[str, int] = json.load(f)

    id_to_token: dict[int, str] = {
        token_id: token_str for token_str, token_id in raw_vocab.items()}

    vocab = sort_vocab(id_to_token)
    vocab.punct_tokens = set()
    for char in ('{', '}', ',', ':', '"'):
        tokens = tensor_to_token(model.encode(char))
        if tokens:
            vocab.punct_tokens.add(tokens[0])

    for func in tools:
        tokens = tensor_to_token(model.encode(func.name))
        if tokens:
            vocab.function_name_tokens.add(tuple(tokens))

    current_input_ids = tensor_to_token(model.encode(prompt))
    init_ids = tensor_to_token(model.encode('{'))

    fsm = FunctionCallFSM(tools, vocab, id_to_token)
    generated_tokens: list[int] = list(init_ids)
    max_tokens = 100

    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(current_input_ids)
        current_text = model.decode(generated_tokens)

        allowed_tokens = fsm.get_allowed_tokens(current_text)

        if allowed_tokens is not None:
            mask = [-math.inf] * len(logits)
            for token_id in allowed_tokens:
                mask[token_id] = logits[token_id]
            logits = mask

        next_token_id = int(max(range(len(logits)), key=lambda i: logits[i]))

        current_input_ids.append(next_token_id)
        generated_tokens.append(next_token_id)

        if current_text.endswith("}}"):
            break

    return "".join([decode_token_string(id_to_token[t])
                    for t in generated_tokens])
