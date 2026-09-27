import json
from pathlib import Path

from llm_sdk import Small_LLM_Model  # type: ignore


class Vocab:
    """id -> token text, loaded once from the SDK's vocab file."""

    def __init__(self, path: Path) -> None:
        raw: dict[str, int] = json.loads(path.read_text())
        self._id_to_str: dict[int, str] = {id: self._clean(token)
                                           for token, id in raw.items()}

    @staticmethod
    def _clean(token: str) -> str:
        return (token
                .replace("Ġ", " ")
                .replace("▁", " ")
                .replace("Ċ", "\n"))

    def token_to_txt(self, token_id: int) -> str:
        return self._id_to_str.get(token_id, "")

    def items(self) -> list[tuple[int, str]]:
        return list(self._id_to_str.items())

    @classmethod
    def from_model(cls, model: Small_LLM_Model) -> "Vocab":
        """Build the vocab straight from the SDK's own vocab file path."""
        return cls(Path(model.get_path_to_vocab_file()))
