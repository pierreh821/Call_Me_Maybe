from typing import Protocol

from .vocab import Vocab


class TokenConstraint(Protocol):
    def allowed_ids(self, generated: str) -> set[int]:
        """Token ids that can legally follow `generated`."""

    def resolved(self, generated: str) -> str | None:
        """The final value, if `generated` already determines it uniquely."""

    def is_stop(self, token_id: int, generated: str) -> bool:
        """Returns true if the character means absolute stop"""


class ChoiceConstraint:
    """Restrict generation to a fixed set of candidate strings."""

    def __init__(self, vocab: Vocab, choices: list[str]) -> None:
        self.choices = choices
        self.vocab = vocab

    def _remaining(self, generated: str) -> list[str]:
        return [c for c in self.choices if c.startswith(generated)]

    def allowed_ids(self, generated: str) -> set[int]:
        # Remaining choices
        remaining = self._remaining(generated)

        return {
            token_id for token_id, text in self.vocab.items()
            if text and any(
                choice[len(generated):].startswith(text)
                for choice in remaining
            )  # Keeping tokens which writes the available functions' rest
        }

    def resolved(self, generated: str) -> str | None:
        """
        Returns the only function left if it exists, else None.

        Args:
            generated (str): generated text

        Returns:
            str | None: function name (str) if there is only one function
                matching the generated text, None if there is many functions
                left or no one found
        """
        remaining = self._remaining(generated)
        return remaining[0] if len(remaining) == 1 else None

    def is_stop(self, token_id: int, generated: str) -> bool:
        return False


class NumericalConstraint:
    """Digits, one optional leading '-', one optional '.'."""

    STOP_CHARS = (" ", "\n", '"', ",", "}")

    def __init__(self, vocab: Vocab, allow_float: bool) -> None:
        self.vocab = vocab
        self.allow_float = allow_float

    def _char_ok(self, generated: str, ch: str) -> bool:
        if ch.isdigit():
            return True
        if ch == '-':
            # Works if there is no char before '-'
            return generated == ''
        if ch == '.' and self.allow_float:
            # Works if there is already digits before '.'
            return '.' not in generated and any(c.isdigit() for c in generated)
        return False

    def _is_number_token(self, generated: str, text: str) -> bool:
        return bool(text) and all(
            self._char_ok(generated + text[:i], c)
            for i, c in enumerate(text))

    def number_ids(self, generated: str) -> set[int]:
        return {t_id for t_id, text in self.vocab.items()
                if self._is_number_token(generated, text)}

    def stop_ids(self, generated: str) -> set[int]:
        if not any(c.isdigit() for c in generated):
            return set()
        return {t_id for t_id, text in self.vocab.items()
                if text and text[0] in self.STOP_CHARS}

    def allowed_ids(self, generated: str) -> set[int]:
        # We try every char of the token to test if it is compatible with nb
        return self.number_ids(generated) | self.stop_ids(generated)

    def resolved(self, generated: str) -> str | None:
        return None

    def is_stop(self, token_id: int, generated: str) -> bool:
        if not any(c.isdigit() for c in generated):
            return False

        text = self.vocab.token_to_txt(token_id)
        return bool(text) and text[0] in self.STOP_CHARS


class RawTextConstraint:
    PAIRS = {"(": ")", "[": "]", "{": "}"}

    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    def allowed_ids(self, generated: str) -> set[int]:
        return {t_id for t_id, _ in self.vocab.items()}

    def resolved(self, generated: str) -> str | None:
        return None

    def _unbalanced(self, generated: str) -> bool:
        stack: list[str] = []
        for ch in generated:
            if ch in self.PAIRS:
                stack.append(ch)
            elif ch in self.PAIRS.values() and stack:
                stack.pop()
        return bool(stack)

    def is_stop(self, token_id: int, generated: str) -> bool:
        if self._unbalanced(generated):
            return False
        text = self.vocab.token_to_txt(token_id)
        return "\n" in text
