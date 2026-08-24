#!/usr/bin/env python3
"""Real-time speech-to-text from a USB microphone on a Raspberry Pi 5 using AssemblyAI.

Captures raw 16-bit PCM from the microphone with sounddevice (PortAudio) and streams
it to AssemblyAI's Universal-Streaming API, printing partial turns live and final
turns on their own line.
"""

import argparse
import os
import queue
import signal
import sys
import threading
import time
import wave
from datetime import datetime
from typing import Iterator, Optional

import sounddevice as sd
from assemblyai.streaming.v3 import (
    BeginEvent,
    Encoding,
    RealTimeError,
    RealTimeEvents,
    RealTimeParameters,
    RealTimeTranscriber,
    RealTimeTranscriberOptions,
    TerminationEvent,
    TurnEvent,
)

PREFERRED_SAMPLE_RATE = 16000
BLOCK_MS = 50

audio_queue: "queue.Queue[Optional[bytes]]" = queue.Queue()
stop_event = threading.Event()
transcript_file = None
printed_partial = False


def list_devices() -> None:
    print(sd.query_devices())


def resolve_device(device: Optional[str]) -> Optional[int]:
    if device is None:
        return None
    try:
        return int(device)
    except ValueError:
        pass
    matches = [
        index
        for index, info in enumerate(sd.query_devices())
        if device.lower() in info["name"].lower() and info["max_input_channels"] > 0
    ]
    if not matches:
        raise SystemExit(f"No input device matching {device!r}. Run with --list-devices.")
    return matches[0]


def pick_sample_rate(device: Optional[int]) -> int:
    """16 kHz keeps bandwidth low, but many USB mics only run at 44.1/48 kHz."""
    for rate in (PREFERRED_SAMPLE_RATE, 48000, 44100, 32000, 8000):
        try:
            sd.check_input_settings(device=device, channels=1, samplerate=rate, dtype="int16")
            return rate
        except Exception:
            continue
    raise SystemExit("Microphone does not support any usable sample rate.")


def on_begin(client: RealTimeTranscriber, event: BeginEvent) -> None:
    print(f"[session {event.id}] listening — press Ctrl+C to stop\n", flush=True)


def on_turn(client: RealTimeTranscriber, event: TurnEvent) -> None:
    global printed_partial
    if not event.transcript:
        return
    if event.end_of_turn:
        print(f"\r\033[K{event.transcript}", flush=True)
        printed_partial = False
        if transcript_file:
            transcript_file.write(f"{datetime.now():%H:%M:%S} {event.transcript}\n")
            transcript_file.flush()
    else:
        print(f"\r\033[K… {event.transcript}", end="", flush=True)
        printed_partial = True


def on_terminated(client: RealTimeTranscriber, event: TerminationEvent) -> None:
    if printed_partial:
        print()
    print(f"[session ended — {event.audio_duration_seconds}s of audio]", flush=True)


def on_error(client: RealTimeTranscriber, error: RealTimeError) -> None:
    print(f"[error] {error}", file=sys.stderr, flush=True)


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
    parser.add_argument("--speech-model", default="universal-streaming-english")
    parser.add_argument("--language-code", default="en")
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

    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise SystemExit("ASSEMBLYAI_API_KEY is not set.")

    if args.wav:
        device = None
        with wave.open(args.wav, "rb") as wav:
            sample_rate = wav.getframerate()
    else:
        device = resolve_device(args.device)
        sample_rate = args.sample_rate or pick_sample_rate(device)
    blocksize = int(sample_rate * BLOCK_MS / 1000)

    if args.save:
        transcript_file = open(args.save, "a", encoding="utf-8")

    client = RealTimeTranscriber(
        RealTimeTranscriberOptions(terminate_timeout=10.0),
        api_key=api_key,
    )
    client.on(RealTimeEvents.Begin, on_begin)
    client.on(RealTimeEvents.Turn, on_turn)
    client.on(RealTimeEvents.Termination, on_terminated)
    client.on(RealTimeEvents.Error, on_error)

    signal.signal(signal.SIGINT, lambda *_: (stop_event.set(), audio_queue.put(None)))
    signal.signal(signal.SIGTERM, lambda *_: (stop_event.set(), audio_queue.put(None)))

    client.connect(
        RealTimeParameters(
            sample_rate=sample_rate,
            encoding=Encoding.pcm_s16le,
            speech_model=args.speech_model,
            language_codes=[args.language_code],
            format_turns=True,
            keyterms_prompt=args.keyterms or None,
        )
    )

    try:
        if args.wav:
            client.stream(wav_chunks(args.wav))
        else:
            stream = sd.RawInputStream(
                samplerate=sample_rate,
                blocksize=blocksize,
                device=device,
                channels=1,
                dtype="int16",
                callback=audio_callback,
            )
            with stream:
                client.stream(audio_chunks())
    finally:
        client.disconnect(terminate=True)
        if transcript_file:
            transcript_file.close()


if __name__ == "__main__":
    main()
