import llm_sdk
import json
from typing import Callable, Optional
from tqdm import tqdm
from torch import Tensor

from .tools import Tool

MAX_TOKENS = 100


class FunctionCalling:
    def __init__(self, tools: list[Tool]):
        self.tools = tools
        self.model = llm_sdk.Small_LLM_Model()  # type: ignore

    def run_prompts(self,
                    prompts: list[str],
                    on_result: Optional[Callable[[list[dict]], None]] = None
                    ) -> tuple[list[dict], list[str]]:

        results = []
        errors = []

        for prompt in tqdm(prompts, desc="Prompt processing"):
            res_str = self._single_function_call(prompt)
            try:
                res_dict = json.loads(res_str)
                res_dict["prompt"] = prompt

                results.append(self._format_sgl_res(res_dict))

                if on_result:
                    on_result(results)

            except json.decoder.JSONDecodeError:
                errors.append(f"Error on prompt '{prompt}'. "
                              f"Response was '{res_str}'.")

        return results, errors

    def _format_sgl_res(self, res_dict: dict) -> dict:
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

        # Check each function parameter type (correct if necessary)
        for p_name, p_val in res_dict.get("parameters", {}).items():
            p_type = tool.parameters.get(p_name)
            if p_type is None:
                continue
            res_dict["parameters"][p_name] = p_type(p_val)

        return res_dict

    def _single_function_call(self, prompt: str) -> str:
        prompt_str = self._format_prompt(prompt)

        current_input_ids = self._tensor_to_token(
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

        return '{' + self.model.decode(generated_tokens)

    def _format_prompt(self, prompt: str) -> str:
        tools_list = ""
        for tool in self.tools:
            parameters = [f'''{p_name}: {str(p_type).split("'")[1]}'''
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
    def _tensor_to_token(tensor: Tensor) -> list[int]:
        res = tensor.tolist() if hasattr(tensor, "tolist") else list(tensor)

        while (isinstance(res, list)
               and len(res) > 0
               and isinstance(res[0], list)):
            res = res[0]

        return res

    @staticmethod
    def _decode_token_string(s: str) -> str:
        return s.replace(u"\u2581", ' ').replace('Ġ', ' ')
