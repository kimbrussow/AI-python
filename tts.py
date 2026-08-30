"""Offline text-to-speech playback to a USB speaker.

Two backends, both entirely on-device so a reply never waits on the network:

* ``piper`` — neural voices, needs a downloaded ``.onnx`` model (best quality on a Pi 5)
* ``espeak-ng`` — always available via apt, robotic but instant

Audio is written to PortAudio as raw 16-bit PCM, so the output device can be selected
by name (``--speaker USB``) exactly like the capture device, and numpy is not needed.
"""

import io
import json
import shutil
import subprocess
import threading
import wave
from pathlib import Path

import sounddevice as sd

from audio_io import resolve_device

BACKENDS = ("piper", "espeak-ng")
PIPER_FALLBACK_RATE = 22050
PLAYBACK_BLOCK_FRAMES = 1024


class TTSError(RuntimeError):
    pass


def available_backends() -> list[str]:
    return [name for name in BACKENDS if shutil.which(name)]


def _pick_backend(requested: str) -> str:
    if requested != "auto":
        if not shutil.which(requested):
            raise TTSError(f"{requested} is not installed (try: sudo apt install espeak-ng).")
        return requested
    found = available_backends()
    if not found:
        raise TTSError(
            "No text-to-speech backend found. Install one with 'sudo apt install espeak-ng' "
            "or 'pip install piper-tts' plus a voice model."
        )
    return found[0]


def _piper_sample_rate(model: Path) -> int:
    """Piper models ship a sidecar JSON describing the voice, including its rate."""
    for candidate in (model.with_suffix(model.suffix + ".json"), model.with_suffix(".json")):
        if candidate.is_file():
            config = json.loads(candidate.read_text(encoding="utf-8"))
            rate = config.get("audio", {}).get("sample_rate")
            if isinstance(rate, int):
                return rate
    return PIPER_FALLBACK_RATE


def _run(command: list[str], stdin_text: str | None = None) -> bytes:
    process = subprocess.run(
        command,
        input=stdin_text.encode("utf-8") if stdin_text is not None else None,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        message = process.stderr.decode("utf-8", "replace").strip()
        raise TTSError(f"{command[0]} failed: {message or f'exit code {process.returncode}'}")
    if not process.stdout:
        raise TTSError(f"{command[0]} produced no audio.")
    return process.stdout


def _decode_wav(data: bytes) -> tuple[bytes, int, int]:
    with wave.open(io.BytesIO(data), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise TTSError("Expected 16-bit PCM from the text-to-speech backend.")
        return wav.readframes(wav.getnframes()), wav.getframerate(), wav.getnchannels()


class Speaker:
    """Synthesizes text and plays it on an output device, one utterance at a time."""

    def __init__(
        self,
        backend: str = "auto",
        device: str | None = None,
        voice: str = "en",
        words_per_minute: int = 165,
        piper_model: str | None = None,
    ) -> None:
        self.backend = _pick_backend(backend)
        self.device = resolve_device(device, kind="output")
        self.voice = voice
        self.words_per_minute = words_per_minute
        self.piper_model = Path(piper_model) if piper_model else None
        self._lock = threading.Lock()
        if self.backend == "piper":
            if self.piper_model is None:
                raise TTSError("--piper-model is required for the piper backend.")
            if not self.piper_model.is_file():
                raise TTSError(f"Piper model not found: {self.piper_model}")

    def command(self) -> list[str]:
        if self.backend == "piper":
            assert self.piper_model is not None
            return ["piper", "--model", str(self.piper_model), "--output-raw"]
        return [
            "espeak-ng",
            "-v",
            self.voice,
            "-s",
            str(self.words_per_minute),
            "--stdout",
        ]

    def synthesize(self, text: str) -> tuple[bytes, int, int]:
        """Return (raw 16-bit PCM, sample rate, channel count) for ``text``."""
        audio = _run(self.command(), stdin_text=text)
        if self.backend == "piper":
            assert self.piper_model is not None
            return audio, _piper_sample_rate(self.piper_model), 1
        return _decode_wav(audio)

    def play(self, pcm: bytes, sample_rate: int, channels: int = 1) -> None:
        bytes_per_frame = 2 * channels
        stream = sd.RawOutputStream(
            samplerate=sample_rate,
            device=self.device,
            channels=channels,
            dtype="int16",
            blocksize=PLAYBACK_BLOCK_FRAMES,
        )
        with stream:
            block = PLAYBACK_BLOCK_FRAMES * bytes_per_frame
            for offset in range(0, len(pcm), block):
                stream.write(pcm[offset : offset + block])

    def say(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        with self._lock:
            self.play(*self.synthesize(text))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Speak text on a USB speaker.")
    parser.add_argument("text", nargs="+")
    parser.add_argument("--backend", default="auto", choices=("auto", *BACKENDS))
    parser.add_argument("--speaker", help="output device index or name substring, e.g. 'USB'")
    parser.add_argument("--voice", default="en", help="espeak-ng voice, e.g. 'en-gb'")
    parser.add_argument("--words-per-minute", type=int, default=165)
    parser.add_argument("--piper-model", help="path to a piper .onnx voice model")
    args = parser.parse_args()

    speaker = Speaker(
        backend=args.backend,
        device=args.speaker,
        voice=args.voice,
        words_per_minute=args.words_per_minute,
        piper_model=args.piper_model,
    )
    speaker.say(" ".join(args.text))


if __name__ == "__main__":
    main()
