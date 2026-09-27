from typing import Protocol

from .vocab import Vocab


class TokenConstraint(Protocol):
    """Interface for stateful rules that restrict token generation."""

    def allowed_ids(self, generated: str) -> set[int]:
        """Return token IDs that can legally follow the generated text.

        Args:
            generated: Text generated so far.

        Returns:
            The IDs of vocabulary tokens allowed at this point.
        """

    def resolved(self, generated: str) -> str | None:
        """Return a completed value when the prefix determines one.

        Args:
            generated: Text generated so far.

        Returns:
            The resolved value, or ``None`` while generation is needed.
        """

    def stop_at(self, generated: str, token_text: str) -> str | None:
        """Return the part of a token to keep when it ends generation.

        Args:
            generated: Text generated before the candidate token.
            token_text: Decoded text of the candidate token.

        Returns:
            A token fragment to append, or ``None`` to append the whole token.
        """
        ...


class ChoiceConstraint:
    """Restrict generation to a fixed set of candidate strings."""

    def __init__(self, vocab: Vocab, choices: list[str]) -> None:
        """Create a constraint that only permits the supplied choices.

        Args:
            vocab: Vocabulary used to map allowed token text to IDs.
            choices: Candidate strings from which one value must be generated.
        """
        self.choices = choices
        self.vocab = vocab

    def _remaining(self, generated: str) -> list[str]:
        """Return choices that still match the generated prefix.

        Args:
            generated: Text generated so far.

        Returns:
            Choices beginning with ``generated``.
        """
        return [c for c in self.choices if c.startswith(generated)]

    def allowed_ids(self, generated: str) -> set[int]:
        """Find vocabulary tokens that preserve at least one candidate.

        Args:
            generated: Text generated so far.

        Returns:
            IDs whose text is a prefix of the remainder of a valid choice.
        """
        remaining = self._remaining(generated)

        return {
            token_id for token_id, text in self.vocab.items()
            if text and any(
                choice[len(generated):].startswith(text)
                for choice in remaining
            )  # Keeping tokens which writes the available functions' rest
        }

    def resolved(self, generated: str) -> str | None:
        """Resolve the prefix when exactly one candidate remains.

        Args:
            generated: Text generated so far.

        Returns:
            The sole matching choice, or ``None`` if there are zero or multiple
            matches.
        """
        remaining = self._remaining(generated)
        return remaining[0] if len(remaining) == 1 else None

    def stop_at(self, generated: str, token_text: str) -> str | None:
        """Keep choice generation running until a choice is resolved.

        Args:
            generated: Text generated so far.
            token_text: Decoded text of the candidate token.

        Returns:
            Always ``None``; choice completion is handled by ``resolved``.
        """
        return None


class NumericalConstraint:
    """Constrain numeric values to digits and optional sign/decimal point."""

    STOP_CHARS: tuple[str, ...] = (" ", "\n", '"', ",", "}")

    def __init__(self, vocab: Vocab, allow_float: bool) -> None:
        """Initialize numeric token filtering.

        Args:
            vocab: Vocabulary to filter at each generation step.
            allow_float: Whether a decimal point is permitted.
        """
        self.vocab = vocab
        self.allow_float = allow_float

    def _char_ok(self, generated: str, ch: str) -> bool:
        """Check whether one character is legal after the current prefix.

        Args:
            generated: Numeric text generated before ``ch``.
            ch: Character being validated.

        Returns:
            Whether ``ch`` preserves the supported numeric format.
        """
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
        """Check whether an entire token can extend a numeric prefix.

        Args:
            generated: Numeric text generated so far.
            text: Decoded candidate token text.

        Returns:
            Whether every character in ``text`` is permitted in sequence.
        """
        return bool(text) and all(
            self._char_ok(generated + text[:i], c)
            for i, c in enumerate(text))

    def number_ids(self, generated: str) -> set[int]:
        """Return IDs for tokens that extend the numeric value.

        Args:
            generated: Numeric text generated so far.

        Returns:
            IDs of vocabulary tokens containing only legal numeric text.
        """
        return {t_id for t_id, text in self.vocab.items()
                if self._is_number_token(generated, text)}

    def stop_ids(self, generated: str) -> set[int]:
        """Return IDs for delimiters after at least one numeric digit.

        Args:
            generated: Numeric text generated so far.

        Returns:
            IDs of vocabulary tokens beginning with a recognized delimiter.
        """
        if not any(c.isdigit() for c in generated):
            return set()
        return {t_id for t_id, text in self.vocab.items()
                if text and text[0] in self.STOP_CHARS}

    def allowed_ids(self, generated: str) -> set[int]:
        """Return numeric continuation and termination token IDs.

        Args:
            generated: Numeric text generated so far.

        Returns:
            IDs for a valid numeric continuation or a stopping delimiter.
        """
        return self.number_ids(generated) | self.stop_ids(generated)

    def resolved(self, generated: str) -> str | None:
        """Leave numeric completion to delimiter detection.

        Args:
            generated: Numeric text generated so far.

        Returns:
            Always ``None`` because a numeric prefix is not self-terminating.
        """
        return None

    def stop_at(self, generated: str, token_text: str) -> str | None:
        """Stop before a delimiter once at least one digit was generated.

        Args:
            generated: Numeric text generated so far.
            token_text: Decoded text of the candidate token.

        Returns:
            An empty fragment to discard a leading delimiter, or ``None``.
        """
        if not any(c.isdigit() for c in generated):
            return None
        if token_text and token_text[0] in self.STOP_CHARS:
            return ""
        return None


class RawTextConstraint:
    """Allow free-form text and stop at an un-nested quote or newline."""

    STOP_CHARS: tuple[str, ...] = ('"', "\n")
    PAIRS: dict[str, str] = {"(": ")", "[": "]", "{": "}"}

    def __init__(self, vocab: Vocab) -> None:
        """Initialize free-form text generation.

        Args:
            vocab: Vocabulary whose tokens are available for text generation.
        """
        self.vocab = vocab

    def allowed_ids(self, generated: str) -> set[int]:
        """Return all vocabulary IDs for unconstrained text continuation.

        Args:
            generated: Text generated so far (not used to filter tokens).

        Returns:
            IDs of all vocabulary tokens.
        """
        return {t_id for t_id, _ in self.vocab.items()}

    def resolved(self, generated: str) -> str | None:
        """Leave completion to the text stopping rules.

        Args:
            generated: Text generated so far.

        Returns:
            Always ``None`` because raw text is not self-terminating.
        """
        return None

    def _stack_of(self, text: str) -> list[str]:
        """Collect unmatched opening delimiters in ``text``.

        Args:
            text: Text whose parentheses, brackets, and braces are scanned.

        Returns:
            Opening delimiter characters that have not yet been closed.
        """
        stack: list[str] = []
        for ch in text:
            if ch in self.PAIRS:
                stack.append(ch)
            elif ch in self.PAIRS.values() and stack:
                stack.pop()
        return stack

    def stop_at(self, generated: str, token_text: str) -> str | None:
        """Stop at the first un-nested string boundary in a token.

        Args:
            generated: Text generated before the candidate token.
            token_text: Decoded text of the candidate token.

        Returns:
            The token prefix before a stopping character, or ``None`` when
            the complete token can be appended.
        """
        stack = self._stack_of(generated)
        for i, ch in enumerate(token_text):
            if ch in self.PAIRS:
                stack.append(ch)
            elif ch in self.PAIRS.values() and stack:
                stack.pop()
            elif ch in self.STOP_CHARS and not stack:
                return token_text[:i]
        return None
