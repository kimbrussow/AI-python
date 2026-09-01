"""Deepgram Nova-3 streaming client over a raw WebSocket.

Deepgram's realtime endpoint takes raw PCM as binary frames and answers with JSON
``Results`` messages, so the whole client is a URL, one header and a reader thread —
no extra SDK on the Pi beyond ``websockets``.

Finality needs both of Deepgram's end-of-speech signals: ``speech_final`` comes from
the audio-level voice activity detector and misses turns in a noisy room, while
``UtteranceEnd`` is derived from word timings and fires there instead. Interim results
are streamed as partial turns for live display.
"""

import contextlib
import json
import sys
import threading
from collections.abc import Callable, Iterable, Sequence
from urllib.parse import urlencode

from websockets.sync.client import connect

from stt import Turn

ENDPOINT = "wss://api.deepgram.com/v1/listen"
DEFAULT_MODEL = "nova-3"
# Silence (ms) that ends a turn, and the gap between words that ends one when the
# room is too noisy for the voice activity detector. Deepgram sends interim results
# every second, so utterance_end_ms below 1000 buys nothing.
ENDPOINTING_MS = 300
UTTERANCE_END_MS = 1000
READER_JOIN_TIMEOUT_S = 5.0


class TurnTracker:
    """Assembles Deepgram messages into partial turns and one final turn per utterance."""

    def __init__(self) -> None:
        self._settled = ""
        """Finalized segments of the current utterance, waiting for an end-of-speech signal."""

    def feed(self, message: dict) -> Turn | None:
        kind = message.get("type")
        if kind == "UtteranceEnd":
            return self.flush()
        if kind != "Results":
            return None

        alternatives = message.get("channel", {}).get("alternatives") or [{}]
        transcript = (alternatives[0].get("transcript") or "").strip()
        if not transcript:
            # A silent interim carries no text; an empty final still ends the utterance.
            return self.flush() if message.get("speech_final") else None

        if message.get("speech_final"):
            self._settled = self._join(transcript)
            return self.flush()
        if message.get("is_final"):
            self._settled = self._join(transcript)
            return Turn(self._settled, end_of_turn=False)
        return Turn(self._join(transcript), end_of_turn=False)

    def _join(self, transcript: str) -> str:
        return f"{self._settled} {transcript}".strip() if self._settled else transcript

    def flush(self) -> Turn | None:
        """End the current utterance now — the end-of-speech signal or the socket closing."""
        if not self._settled:
            return None
        transcript, self._settled = self._settled, ""
        return Turn(transcript, end_of_turn=True)


def query_parameters(
    sample_rate: int,
    model: str = DEFAULT_MODEL,
    language: str = "en",
    keyterms: Sequence[str] = (),
) -> list[tuple[str, str]]:
    """Query string pairs for the streaming endpoint; ``keyterm`` repeats per term."""
    parameters = [
        ("model", model),
        ("language", language),
        ("encoding", "linear16"),
        ("sample_rate", str(sample_rate)),
        ("channels", "1"),
        ("smart_format", "true"),
        ("interim_results", "true"),
        ("endpointing", str(ENDPOINTING_MS)),
        ("utterance_end_ms", str(UTTERANCE_END_MS)),
    ]
    parameters += [("keyterm", term) for term in keyterms]
    return parameters


class DeepgramTranscriber:
    """Streams 16-bit PCM to Deepgram and reports turns on the calling thread's callback."""

    name = "deepgram"

    def __init__(
        self,
        api_key: str,
        sample_rate: int,
        model: str = DEFAULT_MODEL,
        language: str = "en",
        keyterms: Sequence[str] = (),
    ) -> None:
        self.api_key = api_key
        self.parameters = query_parameters(sample_rate, model, language, keyterms)
        self.model = model
        self._connection = None
        self._closed = threading.Event()

    def url(self) -> str:
        return f"{ENDPOINT}?{urlencode(self.parameters)}"

    def stream(self, chunks: Iterable[bytes], on_turn: Callable[[Turn], None]) -> None:
        with connect(
            self.url(),
            additional_headers={"Authorization": f"Token {self.api_key}"},
            max_size=None,
        ) as connection:
            self._connection = connection
            reader = threading.Thread(
                target=self._read, args=(connection, on_turn), name="deepgram-reader", daemon=True
            )
            reader.start()
            try:
                for chunk in chunks:
                    connection.send(chunk)
            finally:
                self.close()
                reader.join(timeout=READER_JOIN_TIMEOUT_S)

    def _read(self, connection, on_turn: Callable[[Turn], None]) -> None:
        tracker = TurnTracker()
        try:
            for message in connection:
                if isinstance(message, bytes):
                    continue
                turn = tracker.feed(json.loads(message))
                if turn:
                    on_turn(turn)
        except Exception as error:  # noqa: BLE001 — the socket dies on shutdown; report and stop
            if not self._closed.is_set():
                print(f"[deepgram] {error}", file=sys.stderr, flush=True)
        # The last words of a session arrive without an end-of-speech signal behind them.
        last = tracker.flush()
        if last:
            on_turn(last)

    def close(self) -> None:
        """Ask Deepgram to finish the open utterance and close, so no audio is dropped."""
        if self._closed.is_set() or self._connection is None:
            return
        self._closed.set()
        with contextlib.suppress(Exception):  # already gone, nothing left to flush
            self._connection.send(json.dumps({"type": "CloseStream"}))
