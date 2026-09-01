"""Provider-independent streaming speech-to-text.

Both entrypoints only need "give me chunks of 16-bit PCM, call me back with turns",
so each provider is wrapped in that shape and selected with ``--provider``. Adding a
provider means adding a module with a ``stream(chunks, on_turn)`` method here.
"""

import os
from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple, Protocol


class Turn(NamedTuple):
    transcript: str
    end_of_turn: bool
    """False while the text can still change, True once the speaker has stopped."""


class Transcriber(Protocol):
    name: str
    model: str

    def stream(self, chunks: Iterable[bytes], on_turn: Callable[[Turn], None]) -> None:
        """Consume audio until ``chunks`` runs out, reporting turns as they arrive."""

    def close(self) -> None:
        """Flush and end the session; safe to call more than once."""


PROVIDERS = ("deepgram", "assemblyai")
API_KEY_VARIABLES = {
    "deepgram": "DEEPGRAM_API_KEY",
    "assemblyai": "ASSEMBLYAI_API_KEY",
}


def api_key(provider: str) -> str:
    variable = API_KEY_VARIABLES[provider]
    key = os.environ.get(variable)
    if not key:
        raise SystemExit(f"{variable} is not set.")
    return key


def build_transcriber(
    provider: str,
    sample_rate: int,
    model: str | None = None,
    language: str = "en",
    keyterms: Sequence[str] = (),
) -> Transcriber:
    key = api_key(provider)
    if provider == "deepgram":
        from deepgram_stt import DEFAULT_MODEL, DeepgramTranscriber

        return DeepgramTranscriber(
            api_key=key,
            sample_rate=sample_rate,
            model=model or DEFAULT_MODEL,
            language=language,
            keyterms=keyterms,
        )
    if provider == "assemblyai":
        from assemblyai_stt import DEFAULT_MODEL, AssemblyAITranscriber

        return AssemblyAITranscriber(
            api_key=key,
            sample_rate=sample_rate,
            model=model or DEFAULT_MODEL,
            language=language,
            keyterms=keyterms,
        )
    raise SystemExit(f"Unknown provider {provider!r}. Choose from: {', '.join(PROVIDERS)}.")
