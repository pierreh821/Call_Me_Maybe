*This project has been created as part of the 42 curriculum by phenry.*

# Call Me Maybe

## Description

Call Me Maybe is a Python tool that translates natural-language requests into
structured function calls. It loads a list of available functions and a list of
prompts, asks the provided small language model to select a function and extract
its arguments, validates the response, and writes JSON results.

The project uses the provided `llm_sdk` package and is designed for Python 3.10
or later. Function definitions are loaded at runtime, so the implementation is
not tied to the example functions in the repository.

## Instructions

Install the dependencies with `uv`:

```sh
uv sync
```

Run the default example:

```sh
uv run python -m src
```

Run with explicit paths:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

The input files are JSON arrays. A prompt item has the form
`{"prompt": "..."}`. A function definition contains `name`, `description`,
`parameters`, and `returns`. Supported parameter types are `number`, `string`,
and `integer`.

Useful Makefile commands:

```sh
make install
make run
make debug
make lint
make clean
```

`make lint` runs flake8 and mypy with the flags required by the subject. The
output directory is created automatically when results are saved.

## Algorithm explanation

The generation pipeline is:

1. `load_functions` and `load_prompts` parse the input files and validate their
   Pydantic models.
2. `build_prompt` serializes the available function names, descriptions, and
   parameter types into a model prompt.
3. `JsonGenerator` encodes the prompt and repeatedly requests logits from the
   SDK. At each step it selects the highest-logit token and stops when the
   generated object is closed or the token limit is reached.
4. The generated text is parsed as JSON.
5. `normalize_call` checks the function name, requires every declared
   parameter, converts values to the declared Python types, and rejects invalid
   calls.
6. Valid `FunctionCallResult` models are serialized as a JSON array.

The generator currently uses greedy generation plus structural stopping and
post-generation validation. A complete constrained-decoding implementation
would additionally mask invalid vocabulary tokens at every generation step
according to the JSON grammar and active function schema. The validation layer
still prevents malformed or incorrectly typed model output from being written
as a successful result.

## Design Decisions

- Pydantic models provide a single validation boundary for input definitions
  and output records.
- Function definitions are converted from JSON type names to Python types once,
  which keeps argument normalization independent of the example data.
- Input and model errors are collected and reported instead of terminating the
  whole prompt-processing loop.
- Results are saved incrementally through a callback, reducing the amount of
  work lost if a later prompt fails.
- The SDK is used only through its public encoding, decoding, and logits APIs.

## Performance analysis

The model is initialized once per run and prompts are processed sequentially.
Generation is bounded by `MAX_TOKENS`, and each prompt requires at most one
bounded generation pass. This keeps runtime predictable, although model
inference remains the dominant cost.

JSON parsing, function-name checks, required-parameter checks, and type
conversion provide deterministic output validation. Missing files, invalid UTF-8,
invalid JSON, and invalid input structures are reported as clear `InputError`
messages.

## Challenges faced

The main challenge is extracting exact string values while asking a small model
to produce JSON. The prompt emphasizes preserving quotes and punctuation, while
`normalize_call` provides a deterministic second validation boundary. Another
challenge is supporting changing function sets; this is handled by constructing
the prompt and validation rules directly from the input definitions.

## Testing strategy

The example input files cover arithmetic, greetings, string reversal, square
roots, and regular-expression substitutions. The moulinette input set covers
multiple parameters, integer and boolean values, SQL text, file paths, and
quoted strings.

Static and style checks are run with:

```sh
make lint
```

Manual end-to-end testing is run with:

```sh
make run
python3 -m json.tool data/output/function_calling_results.json
```

Additional checks should include malformed JSON, missing files, empty strings,
large numbers, special characters, unknown functions, missing parameters, and
wrong parameter types.

## Example Usage

Given a prompt such as:

```json
{"prompt": "What is the sum of 2 and 3?"}
```

and a matching definition for `fn_add_numbers`, the output has this shape:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

## Resources

- The project subject, `subject_1.5.pdf`, supplied with the assignment.
- https://www.vellum.ai/blog/we-dont-speak-json
- https://fr.wikipedia.org/wiki/Automate_fini

AI was used to help inspect the repository, identify undocumented functions
and classes, draft documentation structure, and check the consistency of the
README with the subject requirements. The implementation and documentation
were reviewed against the source code and validated with the repository lint
commands.
