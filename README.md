*This project has been created as part of the 42 curriculum by phenry.*

# Call Me Maybe

## Description

Call Me Maybe turns natural-language requests into structured function-call records. It reads prompts and function definitions from JSON, uses the bundled `llm_sdk` model (Qwen/Qwen3-0.6B by default) to select a function and generate each argument, then writes the results as a JSON array.

The implementation is written for Python 3.10 or later. Function names and parameter rules are loaded from the input definitions rather than hardcoded to the sample functions. The main application is in `src/`; the bundled SDK is in `llm_sdk/`.

## Instructions

Install the project dependencies from the repository root with `uv`:

```sh
uv sync
```

Run using the default inputs and output path:

```sh
uv run python -m src
```

The command-line interface accepts paths for all three files:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

The prompt input is a JSON array of objects such as `{"prompt": "..."}`. The function-definition input is a JSON array with function names, descriptions, parameter declarations, and (in the subject's format) return declarations. This implementation extracts and uses the name, description, and parameter types; it supports `number`, `integer`, and `string` parameters. Input examples are in `data/input/`.

Common Makefile targets are:

```sh
make install
make run
make debug
make lint
make clean
```

`make run` uses the default paths. `make debug` starts the program under `pdb`. `make lint` runs the subject's Flake8 and Mypy checks. The output parent directory is created when results are saved; generated output is ignored by Git.

If you are running the program from a 42 computer, I recommend using:

```sh
export GOINFRE=/goinfre/$USER
export UV_CACHE_DIR="$GOINFRE/uv_storage/cache"
export PIP_CACHE_DIR="$GOINFRE/pip_cache"
export npm_config_cache="$GOINFRE/npm_cache"
export CARGO_HOME="$GOINFRE/cargo"
```

This avoids using too much space from your personal directory.

## Algorithm Explanation

The current implementation uses constrained, greedy token selection for individual function names and argument values; it does **not** generate a complete JSON document token by token. Its pipeline is:

1. `load_prompts` and `load_functions` read JSON and validate prompt and function-definition data. `FunctionDef` maps supported JSON parameter type names to Python types.
2. `build_name_prompt` presents the available functions and the user request. `ChoiceConstraint` filters the vocabulary to tokens that continue at least one candidate function name.
3. For each declared parameter of the selected function, `build_param_prompt` provides the query, function description, prior arguments, and applicable hints. The caller chooses a type-specific token constraint.
4. `Generator` encodes the prompt and requests next-token logits repeatedly. It selects the highest-logit token among the constraint's allowed IDs and stops at a resolved value, delimiter, empty allowed set, or the per-value limit of 30 generated tokens.
5. `NumericalConstraint` permits numeric characters (with a leading minus and, for numbers, one decimal point) and recognizes delimiters. `RawTextConstraint` permits vocabulary tokens for free text and stops at an un-nested quote or newline while tracking common opening/closing delimiters. The resulting values are converted to their declared Python types.
6. A `FunctionCallResult` is assembled and `save_result` writes the records with Python's JSON serializer.

This narrows the model's choices for function names and numeric values, but the string constraint is intentionally permissive and the decoder does not enforce the complete JSON grammar or the full function schema at every token. JSON syntax in the output file is provided by `json.dumps`; schema correctness is not guaranteed for unsupported parameter types or failed conversions. Such cases are reported, but conversion failures may still produce fallback values. This distinction is important: the current design does not claim the subject's 100% schema-constrained decoding target.

## Design Decisions

- Pydantic models validate prompt entries, function definitions, and the shape of result records. A field validator translates the supported parameter type names once, at input time.
- Function selection is performed by the language model, with `ChoiceConstraint` limiting generation to supplied function names rather than selecting by hand-coded query heuristics.
- Parameter extraction uses separate prompts and lightweight type-specific constraints. This keeps each generation step small and makes the constraints independent of the example function set.
- The implementation uses the SDK's public encoding, vocabulary-path, and logits APIs.
- Results are saved incrementally after successful function selection. Errors are collected per prompt and printed at the end; generated defaults for failed conversions are an acknowledged limitation rather than a strict rejection policy.

## Performance Analysis

The model and vocabulary are initialized once per execution. Prompts and parameters are processed sequentially. Each generated function name and parameter value is bounded to 30 tokens, so the number of model steps per prompt scales with the number of parameters as well as the selected function's arguments. Model inference is repeated for each generated token and is expected to dominate runtime; there is no batching or logits cache.

No representative accuracy or latency benchmark has been recorded, so a numerical accuracy or under-five-minute performance claim cannot be made. The main reliability benefit is that output is serialized as syntactically valid JSON and input-file/definition errors are surfaced as readable messages. Function/parameter correctness remains dependent on model generation and the limited constraints described above; it is not formally validated against the complete schema before writing.

## Challenges Faced

Small models can omit, alter, or add punctuation to string values, while free-form text cannot be safely constrained by a simple numeric grammar. The implementation addresses this with query-focused prompts, path/regex hints, quote and newline stopping, and a conversion step. These measures reduce some format errors but are not a substitute for complete schema-aware decoding. Another challenge is allowing the function set to change between runs; the available names and parameter types are therefore read from the supplied JSON instead of being embedded in the source.

## Testing Strategy

The committed demonstration inputs exercise arithmetic, greetings, string reversal, square roots, and regular-expression substitutions. There is not yet a dedicated automated `pytest`/`unittest` suite in the main application. For a manual end-to-end check, run the default inputs, then parse and inspect the generated output:

```sh
make run
uv run python -m json.tool data/output/function_calling_results.json
make lint
```

The subject recommends testing malformed/missing files, empty strings, large numbers, special characters, ambiguous requests, multiple parameters, unknown functions, and type mismatches. Those cases should be added to a repeatable test suite; the example assets alone do not establish accuracy or edge-case reliability.

## Example Usage

For example, a prompt entry can be:

```json
{"prompt": "What is the sum of 2 and 3?"}
```

With a matching `fn_add_numbers` definition, a successful output record has this shape (the actual result depends on model inference):

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

Run with a separate prompt, definitions, and destination using the command in the Instructions section; the flags `--input`, `--functions_definition`, and `--output` accept custom paths.

## Repository Structure

- `src/`: command-line entry point, loaders, prompt construction, constraints, decoding, and result models.
- `llm_sdk/`: bundled model SDK used by the application.
- `data/input/`: default prompt and function-definition examples.
- `docs/subject_1.5.pdf`: project requirements.
- `pyproject.toml`: dependency declaration.
- `Makefile`: install, run, debug, clean, and lint tasks.

## Resources

- The project subject, `subject_1.5.pdf`, supplied with the assignment.
- https://www.vellum.ai/blog/we-dont-speak-json
- https://fr.wikipedia.org/wiki/Automate_fini

AI assistance was used to compare the source against the subject, audit and write PEP 257/Google-style docstrings and applicable type annotations in the application, and draft this README. It was also used to review the Makefile and repository layout against the stated requirements. AI was not used by the program to choose functions at runtime; that decision is made by the configured language model. Documentation and code changes should be checked with the commands in the Testing Strategy section.
