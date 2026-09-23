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
                    ) -> list[dict]:

        results = []

        for prompt in tqdm(prompts, desc="Prompt processing"):
            res_str = self._single_function_call(prompt)
            print("RES_STR: ", res_str)
            res_dict = json.loads(res_str)
            res_dict["prompt"] = prompt

            results.append(self._format_sgl_res(res_dict))

            if on_result:
                on_result(results)

        return results

    def _format_sgl_res(self, res_dict: dict) -> dict:
        tool = None
        print(f"searching tool, res_dict: {res_dict}, "
              f"res_dict.name: {res_dict.get("name")}")
        for t in self.tools:
            if t.name == res_dict.get("name"):
                tool = t

        if tool is None:
            print(f"tool not found, res_dict: {res_dict}, "
                  f"res_dict.name: {res_dict.get("name")}")
            return {}

        for p_name, p_val in res_dict.get("parameters", {}).items():
            p_type = tool.parameters.get(p_name)
            if p_type is None:
                print(f"parameter not found in definitions: {p_name}, existing"
                      f" parameters: {res_dict.get("parameters", {}).keys()}")
                continue
            res_dict["parameters"][p_name] = p_type(p_val)

        return res_dict

    def _single_function_call(self, prompt: str) -> str:
        prompt_str = self._format_prompt(prompt)

        current_input_ids = self._tensor_to_token(
            self.model.encode(prompt_str))

        generated_tokens: list[int] = []
        closing_braces_count = 0

        for _ in range(MAX_TOKENS):
            logits = self.model.get_logits_from_input_ids(current_input_ids)

            next_token_id = int(max(range(len(logits)),
                                    key=lambda i: logits[i]))

            current_input_ids.append(next_token_id)
            generated_tokens.append(next_token_id)

            latest_char = self.model.decode([next_token_id])
            closing_braces_count += latest_char.count('}')

            if closing_braces_count >= 2:
                break

        raw_output = '{' + self.model.decode(generated_tokens)
        # "".join([self.model.decode(t) t in generated_tokens])
        return self._close_json(raw_output)

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

    @staticmethod
    def _close_json(response: str) -> str:
        braces = 0
        i = 0
        for i, char in enumerate(response):
            braces += (char == '{')
            braces -= (char == '}')

            if i > 0 and braces == 0:
                break

        return response[:i+1]
