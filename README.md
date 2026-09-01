# Raspberry Pi 5 + USB mic → Deepgram Nova-3 live speech-to-text

Streams 16-bit PCM audio from a USB microphone straight to Deepgram's Nova-3 realtime
WebSocket API and prints text as you speak. AssemblyAI Universal-Streaming is still
available with `--provider assemblyai`.

Two entrypoints:

| Script | What it does |
| --- | --- |
| `transcribe_mic.py` | prints live transcription of everything it hears |
| `voice_assistant.py` | waits for a wake word, then answers out loud on a USB speaker |

## 1. Install

```bash
chmod +x setup.sh
./setup.sh
```

This installs PortAudio (`libportaudio2`, needed by `sounddevice`) and `espeak-ng`
(text-to-speech), creates a venv, and installs the `websockets`, `assemblyai` and
`sounddevice` packages.

## 2. API key

Get a key from https://console.deepgram.com/ → API Keys, then:

```bash
echo 'export DEEPGRAM_API_KEY=<your-key>' >> ~/.bashrc
source ~/.bashrc
```

For `--provider assemblyai`, set `ASSEMBLYAI_API_KEY` instead (key from
https://www.assemblyai.com/dashboard/home).

## 3. Find the mic

```bash
arecord -l                                       # ALSA card/device numbers
./.venv/bin/python transcribe_mic.py --list-devices
```

## 4. Run

```bash
./.venv/bin/python transcribe_mic.py --device USB
```

Partial results update in place on one line (`… hello wor`), finalized turns print
on their own line. Ctrl+C ends the session cleanly (important: both providers bill
streaming per second of open connection, not per second of audio).

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--device 2` / `--device USB` | pick input by index or name substring |
| `--sample-rate 48000` | force a rate if auto-detection picks a bad one |
| `--keyterms GPIO Raspberry` | bias recognition toward domain words (Nova-3 keyterm prompting) |
| `--save transcript.txt` | append timestamped final turns to a file |
| `--provider assemblyai` | use AssemblyAI Universal-Streaming instead of Deepgram |
| `--model nova-2 --language de` | pick a model and language, per provider |
| `--wav sample.wav` | smoke-test the API/network with a 16-bit mono WAV, no mic needed |

## 5. Wake word + spoken replies

```bash
./.venv/bin/python voice_assistant.py --device USB --speaker USB
```

Say `hey pi, what time is it` — or just `hey pi`, wait for “Yes?”, then speak the
command. `hey pi, goodbye` ends the session.

The wake word is matched on the text the provider returns, not on raw audio, so there is
no second model to run on the Pi. Matching tolerates one letter per word, because
short names come back spelled inconsistently (`hey pie`, `hey py`). The microphone is
ignored while a reply is playing, so the assistant never answers itself.

| Flag | Purpose |
| --- | --- |
| `--speaker USB` / `--speaker 1` | output device by name substring or index |
| `--wake-word 'hey computer'` | change the phrase; repeat the flag for several |
| `--wake-threshold 1.0` | require the wake phrase to be transcribed exactly |
| `--no-follow-up` | ignore a bare wake word instead of asking for the command |
| `--respond-command ./answer.sh` | own reply logic: command text on stdin, reply on stdout |
| `--provider assemblyai` | swap the recognition backend |
| `--tts-backend piper --piper-model en_US-amy-medium.onnx` | neural voice instead of espeak-ng |
| `--voice en-gb --words-per-minute 150` | espeak-ng voice and pace |

Without `--respond-command` the assistant answers the time and date and otherwise
echoes what it heard. Any executable works as the brain — for example an LLM call:

```bash
#!/usr/bin/env bash
# answer.sh — stdin is what the user said, stdout is what the Pi says back
read -r command
curl -s https://api.example.com/chat -d "{\"prompt\":\"$command\"}" | jq -r .reply
```

Speaker check on its own, no microphone or API key needed:

```bash
./.venv/bin/python tts.py --speaker USB "Raspberry Pi audio is working"
```

## Tests

```bash
./.venv/bin/python -m pytest tests
./.venv/bin/ruff check .
```

Wake-word matching, the reply handlers and the audio muting logic are covered without
hardware or network access.

## Pi-specific notes

- **Sample rate**: the script tries 16 kHz first and falls back to 48/44.1 kHz if the
  mic doesn't support it, then tells the provider the rate actually used. No resampling,
  so there's no quality loss or CPU cost.
- **End of turn**: Deepgram's `speech_final` comes from a voice activity detector that a
  noisy room can keep busy, so `utterance_end_ms` (word-timing gaps) is enabled as well
  and either signal closes a turn.
- **Make the USB mic the default** (optional) — create `~/.asoundrc` with card N from
  `arecord -l`:
  ```
  pcm.!default { type asym capture.pcm "hw:N,0" }
  ```
- **Input too quiet**: `alsamixer` → F6 (pick the USB device) → F4 (capture) → raise gain.
- **No sound from the speaker**: `aplay -l` lists playback devices; `speaker-test -c2 -twav`
  verifies the default. Pass the USB device by name with `--speaker USB`, or make it the
  default by adding `playback.pcm "hw:M,0"` to the `pcm.!default` block in `~/.asoundrc`.
- **Assistant hears itself**: keep the speaker off the mic's desk, or lower the volume with
  `alsamixer` → F6 → F3 — capture is muted during playback, but a loud room still echoes
  into the next turn.
- **Overflow warnings** (`input overflow`): raise `BLOCK_MS` in the script to 100.
- **Headless/boot autostart** — `/etc/systemd/system/stt.service`:
  ```ini
  [Unit]
  Description=Deepgram live transcription
  After=network-online.target sound.target

  [Service]
  User=pi
  WorkingDirectory=/home/pi/pi5-assemblyai-stt
  Environment=DEEPGRAM_API_KEY=<your-key>
  ExecStart=/home/pi/pi5-assemblyai-stt/.venv/bin/python voice_assistant.py --device USB --speaker USB
  Restart=always

  [Install]
  WantedBy=multi-user.target
  ```
  Then `sudo systemctl enable --now stt`.
- **Network**: streaming needs a stable outbound WSS connection to `api.deepgram.com`
  (or `streaming.assemblyai.com`); prefer Ethernet or 5 GHz Wi-Fi. A Pi 5 uses <5% CPU for
  capture since all recognition happens server-side.
