from typing import Any
from llm_sdk import Small_LLM_Model  # type: ignore

from .constraints import TokenConstraint
from .vocab import Vocab


class Generator:
    """Generate a JSON object with greedy, model-guided token decoding."""

    def __init__(self, model: Small_LLM_Model, vocab: Vocab) -> None:
        """Initialize the default small language model."""
        self.model = model
        self.vocab = vocab

    def generate(self, prompt: str, constraint: TokenConstraint,
                 max_tokens: int = 30) -> str:
        input_ids = self._to_token_list(self.model.encode(prompt))
        generated = ""

        for _ in range(max_tokens):
            resolved = constraint.resolved(generated)
            if resolved is not None:
                return resolved

            allowed = constraint.allowed_ids(generated)
            if not allowed:
                break

            logits = self.model.get_logits_from_input_ids(input_ids)
            next_token_id = max(allowed, key=lambda i: logits[i])
            text = self.vocab.token_to_txt(next_token_id)

            stop_fragment = constraint.stop_at(generated, text)
            if stop_fragment is not None:
                return generated + stop_fragment  # coupe net, jette le reste

            input_ids.append(next_token_id)
            generated += text

        return constraint.resolved(generated) or generated

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
