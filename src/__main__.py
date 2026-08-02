#!/usr/bin/env python3

from .parsers import FunctionParser, PromptParser
from .models import Function
from .idk import idk

if __name__ == "__main__":
    func_list = FunctionParser.parse("data/input/functions_definition.json")
    Function.print_list(func_list)
    print()

    prompt_list = PromptParser.parse("data/input/function_calling_tests.json")
    for p in prompt_list:
        print(p)

    idk()
