#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace
from pathlib import Path
import json

from .parsers import ToolParser, PromptParser
from .algo import FunctionCalling


def parse_args() -> Namespace:
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

    return input_parser.parse_args()


def save_output(file: Path, output: list[dict]) -> None:
    file.parent.mkdir(exist_ok=True, parents=True)
    file.write_text(json.dumps(output, indent=4, sort_keys=True))


def main() -> None:
    args = parse_args()
    tools = ToolParser.parse(args.functions_definition)
    prompts = PromptParser.parse(args.input)

    fc = FunctionCalling(tools)
    output = fc.run_prompts(prompts)
    save_output(Path(args.output), output)


if __name__ == "__main__":
    main()
