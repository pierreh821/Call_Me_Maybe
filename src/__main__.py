#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import sys

from .parsers import ToolParser, PromptParser
from .algo import FunctionCalling


class Colors:
    RS = "\033[0m"
    B_RD = "\033[1;31m"


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
        default='data/input/function_calling_tests.json',
        help='Path to the prompt list JSON',
        required=False
    )
    input_parser.add_argument(
        '-o', '--output',
        default='data/output/function_calling_results.json',
        help='Path to the output JSON',
        required=False
    )

    return input_parser.parse_args()


def save_output(file: Path, output: list[dict]) -> None:
    file.parent.mkdir(exist_ok=True, parents=True)
    file.write_text(json.dumps(output, indent=4, sort_keys=True))


def print_errors(errors: dict[str, list[str]]) -> None:
    found_error = False

    for prompt, err in errors.items():
        if len(err) == 0:
            continue
        elif not found_error:
            print(
                Colors.B_RD + "\n=== Errors detected ===\n" + Colors.RS +
                "Pay attention to the following errors, they may or may "
                "not be problematic."
                )
            found_error = True

        print(f'- On prompt "{prompt}"', end='')
        print('\n  - '.join([''] + err))


if __name__ == "__main__":
    args = parse_args()
    try:
        tools = ToolParser.parse(args.functions_definition)
        prompts = PromptParser.parse(args.input)
    except SyntaxError as e:
        print(e)
        sys.exit(0)

    fc = FunctionCalling(tools)

    def incremental_save(output_data: list[dict]) -> None:
        save_output(Path(args.output), output_data)

    _, errors = fc.run_prompts(prompts, on_result=incremental_save)
    print_errors(errors)
