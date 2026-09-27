import json
from pathlib import Path

from llm_sdk import Small_LLM_Model  # type: ignore


class Vocab:
    """Map SDK vocabulary token IDs to their cleaned text representations.

    The vocabulary is loaded once from the tokenizer file exposed by the SDK.
    """

    def __init__(self, path: Path) -> None:
        """Load a token-to-ID JSON vocabulary and reverse its mapping.

        Args:
            path: Path to the tokenizer vocabulary JSON file.
        """
        raw: dict[str, int] = json.loads(path.read_text())
        self._id_to_str: dict[int, str] = {
            token_id: self._clean(token)
            for token, token_id in raw.items()
        }

    @staticmethod
    def _clean(token: str) -> str:
        """Replace tokenizer markers with their represented whitespace.

        Args:
            token: Raw token string from the vocabulary file.

        Returns:
            Token text with common leading-space and newline markers decoded.
        """
        return (token
                .replace("Ġ", " ")
                .replace("▁", " ")
                .replace("Ċ", "\n"))

    def token_to_txt(self, token_id: int) -> str:
        """Look up the cleaned text for one token ID.

        Args:
            token_id: ID returned by the model's logits.

        Returns:
            The corresponding token text, or an empty string if not found.
        """
        return self._id_to_str.get(token_id, "")

    def items(self) -> list[tuple[int, str]]:
        """Return all token ID and text pairs.

        Returns:
            A list of ``(token_id, token_text)`` pairs.
        """
        return list(self._id_to_str.items())

    @classmethod
    def from_model(cls, model: Small_LLM_Model) -> "Vocab":
        """Load the vocabulary from the file path exposed by the SDK model.

        Args:
            model: Initialized model that provides its vocabulary file path.

        Returns:
            A vocabulary instance associated with the model's tokenizer.
        """
        return cls(Path(model.get_path_to_vocab_file()))
