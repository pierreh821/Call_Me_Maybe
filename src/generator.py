from typing import Any
import llm_sdk

MAX_TOKENS = 100


class JsonGenerator:
    """Generate a JSON object with greedy, model-guided token decoding."""

    def __init__(self) -> None:
        """Initialize the default small language model."""
        self.model = llm_sdk.Small_LLM_Model()  # type: ignore

    def generate(self, prompt: str) -> str:
        """Generate a JSON object from a prepared function-calling prompt.

        Args:
            prompt: Prompt containing function definitions and a user query.

        Returns:
            The generated JSON object as text.
        """
        current_input_ids = self._to_token_list(
            self.model.encode(prompt))

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

    @staticmethod
    def _to_token_list(tensor: Any) -> list[int]:
        """Flatten an SDK tensor-like value into a list of token IDs.

        Args:
            tensor: Tensor-like value returned by the SDK encoder.

        Returns:
            Token IDs represented as a one-dimensional list.
        """
        res = tensor.tolist() if hasattr(tensor, "tolist") else list(tensor)

        while (isinstance(res, list)
               and len(res) > 0
               and isinstance(res[0], list)):
            res = res[0]

        return res
