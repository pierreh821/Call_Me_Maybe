import llm_sdk
import json
from typing import Callable, Optional
from tqdm import tqdm  # type: ignore
from torch import Tensor

from .function_definition import FunctionDefinition

MAX_TOKENS = 100


class FunctionCaller:
    def __init__(self, tools: list[FunctionDefinition]):
        self.tools = tools
        self.model = llm_sdk.Small_LLM_Model()  # type: ignore
        self.errors: dict[str, list[str]] = {}

    def run_prompts(self,
                    prompts: list[str],
                    on_result: Optional[Callable[[list[dict]], None]] = None
                    ) -> tuple[list[dict], dict[str, list[str]]]:

        results = []

        for prompt in tqdm(prompts, desc="Prompt processing"):
            self.errors[prompt] = []
            res_str = self._generate_json(prompt)

            try:
                res_dict = json.loads(res_str)
                res_dict["prompt"] = prompt

                results.append(self._normalize_result(res_dict, prompt))

                if on_result:
                    on_result(results)

            except json.decoder.JSONDecodeError:
                self.errors[prompt] += [
                    "A fatal error occured (invalid json). "
                    "Its result will be omitted."]

        return results, self.errors

    def _normalize_result(self, res_dict: dict, prompt: str) -> dict:
        tool = None
        tool_name = res_dict.get("name")

        if tool_name is None:
            return res_dict

        # Get the right function
        for t in self.tools:
            if t.name == tool_name:
                tool = t

            elif t.name == ('fn_' + tool_name):
                res_dict["name"] = t.name
                tool = t

        if tool is None:
            return res_dict

        # Check parameters
        for p_name, p_val in res_dict.get("parameters", {}).items():
            p_type = tool.parameters.get(p_name)

            # Remove parameters unsolicited keys
            if p_type is None:
                res_dict["parameters"].pop(p_name)

            # Convert parameters values to the expected type
            else:
                try:
                    res_dict["parameters"][p_name] = p_type(p_val)
                except ValueError:
                    err_type = type(p_val).__name__
                    self.errors[prompt] += [
                        f"Using {tool.name}, cannot convert parameter "
                        f"'{p_name}' value: '{p_val}' ({err_type}, "
                        f"expected {p_type.__name__}). Maybe check your "
                        "prompt?"
                        ]

        # Remove output unsolicited keys
        for key in res_dict.keys():
            if key not in ("prompt", "name", "parameters"):
                res_dict.pop(key)

        return res_dict

    def _generate_json(self, prompt: str) -> str:
        prompt_str = self.build_prompt(prompt)

        current_input_ids = self._to_token_list(
            self.model.encode(prompt_str))

        generated_tokens: list[int] = []
        open_braces = 1

        for _ in range(MAX_TOKENS):
            logits = self.model.get_logits_from_input_ids(current_input_ids)

            next_token_id = int(max(range(len(logits)),
                                    key=lambda i: logits[i]))

            current_input_ids.append(next_token_id)
            generated_tokens.append(next_token_id)

            latest_char = self.model.decode([next_token_id])
            open_braces += latest_char.count('{')
            open_braces -= latest_char.count('}')

            if open_braces <= 0:
                break

        return str('{' + self.model.decode(generated_tokens))

    def build_prompt(self, prompt: str) -> str:
        tools_list = ""
        for tool in self.tools:
            parameters = [f"{p_name}: {p_type.__name__}"
                          for p_name, p_type in tool.parameters.items()]
            param_str = ", ".join(parameters)
            tools_list += f"- {tool.name}({param_str}): {tool.description}\n"

        return (
            f"Available functions: \n{tools_list}"
            "You are a function calling assistant.\n"
            "Produce the output as a JSON with a top-level key called 'name' "
            "and a second-level key named 'parameters' like this:\n"
            '{"name": "<function_name>", "parameters": {<args>}}\n'
            "Preserve ALL exact quotes, punctuation, and characters from the "
            "user query verbatim in string parameters.\n"
            "If the user query contains quotes inside text, escape them "
            "properly with \\\" inside the JSON string values.\n"
            "Do not use parameters that are not given in the available "
            "functions.\n"
            "The output ends when the JSON is properly closed.\n"
            f"User query: {prompt}\n"
            "Clean JSON output: {"
        )

    @staticmethod
    def _to_token_list(tensor: Tensor) -> list[int]:
        res = tensor.tolist() if hasattr(tensor, "tolist") else list(tensor)

        while (isinstance(res, list)
               and len(res) > 0
               and isinstance(res[0], list)):
            res = res[0]

        return res
