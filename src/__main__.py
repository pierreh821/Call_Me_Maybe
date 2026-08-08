#!/usr/bin/env python3

from .parsers import ToolParser, PromptParser
from .models import Tool
from .algo import algo

if __name__ == "__main__":
    func_list = ToolParser.parse("data/input/functions_definition.json")
    print(algo("What is the sum of 2 and 3?", func_list))
