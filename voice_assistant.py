#!/usr/bin/env python3
"""Wake-word voice loop for a Raspberry Pi 5: USB mic in, USB speaker out.

Audio is captured from a USB microphone with sounddevice and streamed to Deepgram
Nova-3 (or AssemblyAI with ``--provider assemblyai``). Every finalized turn is checked
for the wake phrase; the words that follow it are the command, answered out loud
through a USB speaker.

    ./.venv/bin/python voice_assistant.py --device USB --speaker USB

    "hey pi, what time is it"        -> spoken time
    "hey pi"  ... "what time is it"  -> asks "yes?" first, then answers

Replies come from a handler of your choice (``--respond-command``), so the recognition,
wake word and audio plumbing stay independent of what the assistant actually says.
"""

import argparse
import queue
import signal
import subprocess
import sys
import threading
import time

import sounddevice as sd

from audio_io import list_devices, pick_sample_rate, resolve_device
from stt import PROVIDERS, Turn, build_transcriber
from tts import BACKENDS, Speaker, TTSError
from wake_word import DEFAULT_PHRASE, DEFAULT_THRESHOLD, WakeWordDetector

BLOCK_MS = 50
ACKNOWLEDGEMENT = "Yes?"
EXIT_WORDS = ("goodbye", "good bye", "stop listening", "shut down", "go to sleep")
RESPONSE_TIMEOUT_S = 15


def builtin_response(command: str) -> str:
    """Answers that need no configuration, so the loop is useful out of the box."""
    if "time" in command:
        return f"It is {time.strftime('%-I:%M %p')}."
    if "date" in command or "day is it" in command:
        return f"Today is {time.strftime('%A, %B %-d')}."
    return f"You said: {command}"


class Responder:
    """Turns a recognized command into words to speak."""

    def __init__(self, respond_command: str | None = None) -> None:
        self.respond_command = respond_command

    def reply(self, command: str) -> str:
        if not self.respond_command:
            return builtin_response(command)
        try:
            result = subprocess.run(
                self.respond_command,
                shell=True,
                input=command.encode("utf-8"),
                capture_output=True,
                check=False,
                timeout=RESPONSE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return "The response handler timed out."
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", "replace").strip()
            print(f"[responder] exit {result.returncode}: {detail}", file=sys.stderr, flush=True)
            return "The response handler failed."
        return result.stdout.decode("utf-8", "replace").strip() or "I have no answer for that."


class Assistant:
    def __init__(
        self,
        detector: WakeWordDetector,
        speaker: Speaker,
        responder: Responder,
        follow_up: bool = True,
    ) -> None:
        self.detector = detector
        self.speaker = speaker
        self.responder = responder
        self.follow_up = follow_up
        self.audio_queue: queue.Queue[bytes | None] = queue.Queue()
        self.speech_queue: queue.Queue[str | None] = queue.Queue()
        self.stop_event = threading.Event()
        # Set while audio is playing, so the assistant never transcribes its own voice.
        self.speaking = threading.Event()
        self.awaiting_command = False
        self._speech_thread = threading.Thread(target=self._speech_worker, daemon=True)

    # -- audio capture -------------------------------------------------------
    def audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[audio] {status}", file=sys.stderr, flush=True)
        if self.speaking.is_set():
            return
        self.audio_queue.put(bytes(indata))

    def audio_chunks(self):
        while True:
            chunk = self.audio_queue.get()
            if chunk is None:
                return
            yield chunk

    # -- speaking ------------------------------------------------------------
    def speak(self, text: str) -> None:
        self.speech_queue.put(text)

    def _speech_worker(self) -> None:
        while True:
            text = self.speech_queue.get()
            if text is None:
                return
            self.speaking.set()
            try:
                self.speaker.say(text)
            except TTSError as error:
                print(f"[tts] {error}", file=sys.stderr, flush=True)
            finally:
                self._drain_audio()
                self.speaking.clear()

    def _drain_audio(self) -> None:
        """Discard microphone blocks buffered during playback (echo of our own reply)."""
        while True:
            try:
                chunk = self.audio_queue.get_nowait()
            except queue.Empty:
                return
            if chunk is None:
                self.audio_queue.put(None)
                return

    # -- transcript handling -------------------------------------------------
    def handle_turn(self, transcript: str) -> None:
        detection = self.detector.detect(transcript)
        if detection is None:
            if not self.awaiting_command:
                return
            command = transcript.strip()
        else:
            command = detection.command
        self.awaiting_command = False

        if not command:
            self.awaiting_command = self.follow_up
            self.speak(ACKNOWLEDGEMENT)
            return

        print(f"> {command}", flush=True)
        if any(word in command.lower() for word in EXIT_WORDS):
            self.speak("Goodbye.")
            self.shutdown()
            return
        self.speak(self.responder.reply(command))

    def shutdown(self) -> None:
        self.stop_event.set()
        self.audio_queue.put(None)

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        self._speech_thread.start()

    def finish(self) -> None:
        self.speech_queue.put(None)
        self._speech_thread.join(timeout=RESPONSE_TIMEOUT_S)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true", help="show audio devices and exit")
    parser.add_argument("--device", help="input device index or name substring (e.g. 'USB')")
    parser.add_argument("--speaker", help="output device index or name substring (e.g. 'USB')")
    parser.add_argument("--sample-rate", type=int, help="override the capture sample rate")
    parser.add_argument(
        "--wake-word",
        action="append",
        dest="wake_words",
        help=f"wake phrase, repeatable (default: {DEFAULT_PHRASE!r})",
    )
    parser.add_argument(
        "--wake-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="per-word similarity needed to accept the wake phrase (0-1)",
    )
    parser.add_argument(
        "--no-follow-up",
        action="store_true",
        help="ignore a bare wake word instead of asking for the command",
    )
    parser.add_argument(
        "--respond-command",
        help="shell command answering a command on stdin, reply on stdout",
    )
    parser.add_argument("--tts-backend", default="auto", choices=("auto", *BACKENDS))
    parser.add_argument("--voice", default="en", help="espeak-ng voice, e.g. 'en-gb'")
    parser.add_argument("--words-per-minute", type=int, default=165)
    parser.add_argument("--piper-model", help="path to a piper .onnx voice model")
    parser.add_argument("--provider", default=PROVIDERS[0], choices=PROVIDERS)
    parser.add_argument(
        "--model",
        "--speech-model",
        dest="model",
        help="recognition model (default: nova-3 / universal-streaming-english)",
    )
    parser.add_argument("--language", "--language-code", dest="language", default="en")
    parser.add_argument("--keyterms", nargs="*", default=[], help="extra words to bias recognition")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    detector = WakeWordDetector(
        phrases=tuple(args.wake_words or (DEFAULT_PHRASE,)),
        threshold=args.wake_threshold,
    )
    speaker = Speaker(
        backend=args.tts_backend,
        device=args.speaker,
        voice=args.voice,
        words_per_minute=args.words_per_minute,
        piper_model=args.piper_model,
    )
    assistant = Assistant(
        detector=detector,
        speaker=speaker,
        responder=Responder(args.respond_command),
        follow_up=not args.no_follow_up,
    )

    device = resolve_device(args.device)
    sample_rate = args.sample_rate or pick_sample_rate(device)
    transcriber = build_transcriber(
        provider=args.provider,
        sample_rate=sample_rate,
        model=args.model,
        language=args.language,
        keyterms=detector.keyterms() + args.keyterms,
    )

    def on_turn(turn: Turn) -> None:
        if turn.end_of_turn:
            print(f"\r\033[K{turn.transcript}", flush=True)
            assistant.handle_turn(turn.transcript)
        else:
            print(f"\r\033[K… {turn.transcript}", end="", flush=True)

    signal.signal(signal.SIGINT, lambda *_: assistant.shutdown())
    signal.signal(signal.SIGTERM, lambda *_: assistant.shutdown())

    phrases = ", ".join(repr(phrase) for phrase in detector.keyterms())
    print(
        f"[{transcriber.name} {transcriber.model} @ {sample_rate} Hz] "
        f"say {phrases} — Ctrl+C to stop\n",
        flush=True,
    )
    assistant.start()
    try:
        stream = sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=int(sample_rate * BLOCK_MS / 1000),
            device=device,
            channels=1,
            dtype="int16",
            callback=assistant.audio_callback,
        )
        with stream:
            transcriber.stream(assistant.audio_chunks(), on_turn)
    finally:
        transcriber.close()
        assistant.finish()


if __name__ == "__main__":
    main()
