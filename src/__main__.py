#!/usr/bin/env python3

from .parsers import ToolParser, PromptParser
from .models import Tool
from .algo import algo

if __name__ == "__main__":
    func_list = ToolParser.parse("data/input/functions_definition.json")
    Tool.print_list(func_list)
    print()

    prompt_list = PromptParser.parse("data/input/function_calling_tests.json")
    for p in prompt_list:
        print(p)

    print(algo("What is the sum of 2 and 3?", func_list))
