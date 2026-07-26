#!/usr/bin/env python3

from config import Parser, Function

if __name__ == "__main__":
    func_list = Parser.parse("data/input/functions_definition.json")
    Function.print_list(func_list)
