"""AssemblyAI Universal-Streaming behind the common transcriber interface."""

import sys
import threading
from collections.abc import Callable, Iterable, Sequence

from assemblyai.streaming.v3 import (
    Encoding,
    RealTimeError,
    RealTimeEvents,
    RealTimeParameters,
    RealTimeTranscriber,
    RealTimeTranscriberOptions,
    TerminationEvent,
    TurnEvent,
)

from stt import Turn

DEFAULT_MODEL = "universal-streaming-english"
# AssemblyAI bills for the time the socket is open, so termination must be acknowledged
# before the process exits, even on a slow link.
TERMINATE_TIMEOUT_S = 60.0


class AssemblyAITranscriber:
    name = "assemblyai"

    def __init__(
        self,
        api_key: str,
        sample_rate: int,
        model: str = DEFAULT_MODEL,
        language: str = "en",
        keyterms: Sequence[str] = (),
    ) -> None:
        self.model = model
        self._closed = threading.Event()
        self._client = RealTimeTranscriber(
            RealTimeTranscriberOptions(terminate_timeout=TERMINATE_TIMEOUT_S),
            api_key=api_key,
        )
        self._parameters = RealTimeParameters(
            sample_rate=sample_rate,
            encoding=Encoding.pcm_s16le,
            speech_model=model,
            language_codes=[language],
            format_turns=True,
            keyterms_prompt=list(keyterms) or None,
        )
        self._client.on(RealTimeEvents.Termination, _on_terminated)
        self._client.on(RealTimeEvents.Error, _on_error)

    def stream(self, chunks: Iterable[bytes], on_turn: Callable[[Turn], None]) -> None:
        def forward(_client: RealTimeTranscriber, event: TurnEvent) -> None:
            if event.transcript:
                on_turn(Turn(event.transcript, end_of_turn=event.end_of_turn))

        self._client.on(RealTimeEvents.Turn, forward)
        self._client.connect(self._parameters)
        try:
            self._client.stream(chunks)
        finally:
            self.close()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._client.disconnect(terminate=True)


def _on_terminated(_client: RealTimeTranscriber, event: TerminationEvent) -> None:
    print(f"[session ended — {event.audio_duration_seconds}s of audio]", flush=True)


def _on_error(_client: RealTimeTranscriber, error: RealTimeError) -> None:
    print(f"[error] {error}", file=sys.stderr, flush=True)
