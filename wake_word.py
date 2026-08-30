"""Text-level wake word detection over AssemblyAI transcripts.

Recognition happens server-side, so the wake word is matched on the returned text
rather than on raw audio: no extra model, no extra CPU on the Pi. Matching is fuzzy
per word because speech-to-text spells short names inconsistently ("pi", "pie", "py").
"""

import re
import unicodedata
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher
from typing import NamedTuple

DEFAULT_PHRASE = "hey pi"
DEFAULT_THRESHOLD = 0.8

_WORD_RE = re.compile(r"[a-z0-9']+")


def normalize(text: str) -> list[str]:
    """Lowercase, drop accents and punctuation, and split into words."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _WORD_RE.findall(ascii_text)


def _within_one_edit(spoken: str, expected: str) -> bool:
    """One insertion, deletion or substitution apart — 'py'/'hay' for 'pi'/'hey'."""
    if abs(len(spoken) - len(expected)) > 1:
        return False
    matcher = SequenceMatcher(None, spoken, expected)
    shared = sum(block.size for block in matcher.get_matching_blocks())
    return max(len(spoken), len(expected)) - shared <= 1


def _similar(spoken: str, expected: str, threshold: float) -> bool:
    if spoken == expected:
        return True
    if threshold >= 1.0:
        return False
    # Short wake words ('pi') never clear a ratio test, so allow a single letter of slop.
    return (
        SequenceMatcher(None, spoken, expected).ratio() >= threshold
        or _within_one_edit(spoken, expected)
    )


class Detection(NamedTuple):
    phrase: str
    command: str
    """Everything the speaker said after the wake word; empty if nothing followed."""


class WakeWordDetector:
    def __init__(
        self,
        phrases: Sequence[str] = (DEFAULT_PHRASE,),
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        if not phrases:
            raise ValueError("At least one wake phrase is required.")
        self.threshold = threshold
        self._phrases = [(phrase, normalize(phrase)) for phrase in phrases]
        for phrase, words in self._phrases:
            if not words:
                raise ValueError(f"Wake phrase {phrase!r} contains no usable words.")

    def keyterms(self) -> list[str]:
        """Wake phrases as AssemblyAI keyterms, to bias recognition toward them."""
        return [phrase for phrase, _ in self._phrases]

    def detect(self, transcript: str) -> Detection | None:
        words = normalize(transcript)
        best: Detection | None = None
        for phrase, expected in self._phrases:
            end = self._match_end(words, expected)
            if end is None:
                continue
            command = " ".join(words[end:])
            # Prefer the phrase that consumed the most words: "hey pi five" over "hey pi".
            if best is None or len(command) < len(best.command):
                best = Detection(phrase=phrase, command=command)
        return best

    def _match_end(self, words: Sequence[str], expected: Sequence[str]) -> int | None:
        """Index just past the last wake-phrase word, scanning from the end of the text."""
        for start in range(len(words) - len(expected), -1, -1):
            window = words[start : start + len(expected)]
            if all(
                _similar(spoken, want, self.threshold) for spoken, want in zip(window, expected)
            ):
                return start + len(expected)
        return None


def strip_wake_words(transcript: str, phrases: Iterable[str]) -> str:
    """Transcript with any leading wake phrase removed, for logging or echoing back."""
    detection = WakeWordDetector(tuple(phrases)).detect(transcript)
    return detection.command if detection else " ".join(normalize(transcript))
