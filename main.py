#!/usr/bin/env python3

from config import Parser

if __name__ == "__main__":
    func_list = Parser.parse("src/data/input/functions_definition.json")

    for fn in func_list:
        print(f"\n{fn.name}: {fn.description} Returns {str(fn.returns).split("'")[1]}")
        print("Parameters:")
        for p in fn.parameters.items():
            print(f"{p[0]}: {str(p[1]).split("'")[1]}")
