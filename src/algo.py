import llm_sdk
import json
from typing import Callable, Optional
from tqdm import tqdm
from torch import Tensor

from .models import Tool

MAX_TOKENS = 100


class FunctionCalling:
    def __init__(self, tools: list[Tool]):
        self.tools = tools
        self.model = llm_sdk.Small_LLM_Model()

    def run_prompts(self,
                    prompts: list[str],
                    max_workers: int = 5,
                    on_result: Optional[Callable[[list[dict]], None]] = None
                    ) -> list[dict]:

        results = []

        for prompt in tqdm(prompts, desc="Prompt processing"):
            res_str = self._single_function_call(prompt)
            res_dict = json.loads(res_str)
            res_dict["prompt"] = prompt

            results.append(self._format_sgl_res(res_dict))

            if on_result:
                on_result(results)

        return results

    @staticmethod
    def _format_sgl_res(res_dict: dict) -> dict:
        for key, val in res_dict.get("parameters", {}).items():
            if isinstance(val, float) and val.is_integer():
                res_dict["parameters"][key] = int(val)

        return res_dict

    def _single_function_call(self, prompt: str) -> str:
        prompt = self._format_prompt(prompt)

        current_input_ids = self._tensor_to_token(self.model.encode(prompt))
        init_ids = self._tensor_to_token(self.model.encode('{'))

        generated_tokens: list[int] = list(init_ids)

        for _ in range(MAX_TOKENS):
            logits = self.model.get_logits_from_input_ids(current_input_ids)
            current_text = self.model.decode(generated_tokens)

            next_token_id = int(max(range(len(logits)),
                                    key=lambda i: logits[i]))

            current_input_ids.append(next_token_id)
            generated_tokens.append(next_token_id)

            if current_text.endswith("}}"):
                break

        # print("".join([model.decode(t) for t in generated_tokens]))
        return self._test_end_json2("".join([self.model.decode(t)
                                            for t in generated_tokens]))
        # return "".join([model.decode(t) for t in generated_tokens])

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
            # "If the prompt gives a string with quotes, reproduce the quotes in"
            " the response.\n"
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
    def _test_end_json2(response: str) -> str:
        braces = 0
        i = 0
        for i, char in enumerate(response):
            braces += (char == '{')
            braces -= (char == '}')

            if i > 0 and braces == 0:
                break

        return response[:i+1]

    @staticmethod
    def _test_end_json(response: str) -> str | None:
        opened = response.count('{')
        closed = 0
        for i, char in enumerate(response):
            if char == '}':
                closed += 1
            if opened == closed:
                return response[:i+1]

        return None
