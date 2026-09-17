#!/usr/bin/env python3

from argparse import ArgumentParser

from .parsers import ToolParser
from .algo import algo


def main() -> None:
    input_parser = ArgumentParser()

    input_parser.add_argument(
        '-f', '--functions_definition',
        default='data/input/functions_definition.json',
        help='Path to the functions definition JSON',
        required=False
    )
    input_parser.add_argument(
        '-i', '--input',
        default='data/input/function_calling_tests_json',
        help='Path to the prompt list JSON',
        required=False
    )
    input_parser.add_argument(
        '-o', '--output',
        default='data/output.json',
        help='Path to the output JSON',
        required=False
    )

    args = input_parser.parse_args()
    print(args.input)

    func_list = ToolParser.parse("data/input/functions_definition.json")
    print(algo("Greet Shrek", func_list))


if __name__ == "__main__":
    main()
