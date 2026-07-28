#!/usr/bin/env python3
import json

# Utilise des imports relatifs au sein du package src
from .parsers import FunctionParser, PromptParser
from .models import Function

from llm_sdk import Small_LLM_Model

if __name__ == "__main__":
    func_list = FunctionParser.parse("data/input/functions_definition.json")
    Function.print_list(func_list)
    print()

    prompt_list = PromptParser.parse("data/input/function_calling_tests.json")
    for p in prompt_list:
        print(p)

    model = Small_LLM_Model()
    vocab_path = model.get_path_to_vocab_file()

    with open(vocab_path, "r", encoding="utf-8") as f:
        raw_vocab: dict[str, int] = json.load(f)

    print(raw_vocab)
