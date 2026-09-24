from argparse import ArgumentParser, Namespace
from sys import stderr
from pathlib import Path

from .loaders import InputError, load_functions, load_prompts, save_result
from .caller import FunctionCaller
from .models import FunctionCallResult


class Colors:
    """ANSI escape sequences used by the command-line interface."""

    RS = "\033[0m"
    B_RD = "\033[1;31m"


def print_errors(errors: dict[str, list[str]]) -> None:
    """Print non-empty prompt errors in a readable format.

    Args:
        errors: Error messages grouped by their original prompt.
    """
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


def parse_args() -> Namespace:
    """Parse command-line paths for definitions, prompts, and output.

    Returns:
        Parsed command-line arguments.
    """
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


def main() -> int:
    """Load inputs, generate results, and save the output file.

    Returns:
        Zero on success, or one when an input file cannot be loaded.
    """
    args = parse_args()

    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
    except InputError as e:
        print(e, file=stderr)
        return (1)

    def incremental_save(output_data: list[FunctionCallResult]) -> None:
        """Persist results incrementally while prompts are processed.

        Args:
            output_data: Results generated up to the current prompt.
        """
        save_result(Path(args.output), output_data)

    output, errors = FunctionCaller(functions).run(
        prompts, on_result=incremental_save)

    save_result(Path(args.output), output)
    print_errors(errors)

    return (0)
