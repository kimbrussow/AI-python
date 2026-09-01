#!/usr/bin/env python3
"""Real-time speech-to-text from a USB microphone on a Raspberry Pi 5.

Captures raw 16-bit PCM from the microphone with sounddevice (PortAudio) and streams it
to Deepgram Nova-3 (or AssemblyAI Universal-Streaming with ``--provider assemblyai``),
printing partial turns live and final turns on their own line.
"""

import argparse
import contextlib
import queue
import signal
import sys
import threading
import time
import wave
from collections.abc import Iterator

import sounddevice as sd

from audio_io import list_devices, pick_sample_rate, resolve_device
from stt import PROVIDERS, Turn, build_transcriber

BLOCK_MS = 50

audio_queue: queue.Queue[bytes | None] = queue.Queue()
stop_event = threading.Event()
transcript_file = None
printed_partial = False


def on_turn(turn: Turn) -> None:
    global printed_partial
    if turn.end_of_turn:
        print(f"\r\033[K{turn.transcript}", flush=True)
        printed_partial = False
        if transcript_file:
            transcript_file.write(f"{time.strftime('%H:%M:%S')} {turn.transcript}\n")
            transcript_file.flush()
    else:
        print(f"\r\033[K… {turn.transcript}", end="", flush=True)
        printed_partial = True


def audio_callback(indata, frames, time_info, status) -> None:
    if status:
        print(f"[audio] {status}", file=sys.stderr, flush=True)
    audio_queue.put(bytes(indata))


def wav_chunks(path: str) -> Iterator[bytes]:
    """Play a 16-bit mono WAV at real-time pace — smoke test without a microphone."""
    with wave.open(path, "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise SystemExit("WAV must be 16-bit mono PCM.")
        frames_per_block = int(wav.getframerate() * BLOCK_MS / 1000)
        while not stop_event.is_set():
            data = wav.readframes(frames_per_block)
            if not data:
                return
            yield data
            time.sleep(BLOCK_MS / 1000)


def audio_chunks():
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            return
        yield chunk


def main() -> None:
    global transcript_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true", help="show input devices and exit")
    parser.add_argument("--device", help="input device index or name substring (e.g. 'USB')")
    parser.add_argument("--sample-rate", type=int, help="override the capture sample rate")
    parser.add_argument("--provider", default=PROVIDERS[0], choices=PROVIDERS)
    parser.add_argument(
        "--model",
        "--speech-model",
        dest="model",
        help="recognition model (default: nova-3 / universal-streaming-english)",
    )
    parser.add_argument("--language", "--language-code", dest="language", default="en")
    parser.add_argument(
        "--keyterms",
        nargs="*",
        default=[],
        help="domain words to bias recognition, e.g. --keyterms Raspberry GPIO",
    )
    parser.add_argument("--save", help="append final transcripts to this file")
    parser.add_argument(
        "--wav", help="stream a 16-bit mono WAV instead of the mic (connectivity smoke test)"
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.wav:
        device = None
        with wave.open(args.wav, "rb") as wav:
            sample_rate = wav.getframerate()
    else:
        device = resolve_device(args.device)
        sample_rate = args.sample_rate or pick_sample_rate(device)

    exit_stack = contextlib.ExitStack()
    if args.save:
        transcript_file = exit_stack.enter_context(
            open(args.save, "a", encoding="utf-8")  # noqa: SIM115 — closed by the stack
        )

    transcriber = build_transcriber(
        provider=args.provider,
        sample_rate=sample_rate,
        model=args.model,
        language=args.language,
        keyterms=args.keyterms,
    )

    signal.signal(signal.SIGINT, lambda *_: (stop_event.set(), audio_queue.put(None)))
    signal.signal(signal.SIGTERM, lambda *_: (stop_event.set(), audio_queue.put(None)))

    print(
        f"[{transcriber.name} {transcriber.model} @ {sample_rate} Hz] "
        "listening — press Ctrl+C to stop\n",
        flush=True,
    )
    try:
        if args.wav:
            transcriber.stream(wav_chunks(args.wav), on_turn)
        else:
            stream = sd.RawInputStream(
                samplerate=sample_rate,
                blocksize=int(sample_rate * BLOCK_MS / 1000),
                device=device,
                channels=1,
                dtype="int16",
                callback=audio_callback,
            )
            with stream:
                transcriber.stream(audio_chunks(), on_turn)
    finally:
        transcriber.close()
        if printed_partial:
            print()
        exit_stack.close()


if __name__ == "__main__":
    main()
