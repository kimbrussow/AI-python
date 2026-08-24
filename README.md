# Raspberry Pi 5 + USB mic → AssemblyAI live speech-to-text

Streams 16-bit PCM audio from a USB microphone straight to AssemblyAI's
Universal-Streaming WebSocket API and prints text as you speak.

## 1. Install

```bash
chmod +x setup.sh
./setup.sh
```

This installs PortAudio (`libportaudio2`, needed by `sounddevice`), creates a venv,
and installs the `assemblyai` + `sounddevice` packages.

## 2. API key

Get a key from https://www.assemblyai.com/dashboard/home, then:

```bash
echo 'export ASSEMBLYAI_API_KEY=<your-key>' >> ~/.bashrc
source ~/.bashrc
```

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
on their own line. Ctrl+C ends the session cleanly (important: AssemblyAI bills
streaming per second of open connection, not per second of audio).

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--device 2` / `--device USB` | pick input by index or name substring |
| `--sample-rate 48000` | force a rate if auto-detection picks a bad one |
| `--keyterms GPIO Raspberry` | bias recognition toward domain words |
| `--save transcript.txt` | append timestamped final turns to a file |
| `--speech-model universal-streaming-multilingual --language-code de` | non-English |
| `--wav sample.wav` | smoke-test the API/network with a 16-bit mono WAV, no mic needed |

## Pi-specific notes

- **Sample rate**: the script tries 16 kHz first and falls back to 48/44.1 kHz if the
  mic doesn't support it, then tells AssemblyAI the rate actually used. No resampling,
  so there's no quality loss or CPU cost.
- **Make the USB mic the default** (optional) — create `~/.asoundrc` with card N from
  `arecord -l`:
  ```
  pcm.!default { type asym capture.pcm "hw:N,0" }
  ```
- **Input too quiet**: `alsamixer` → F6 (pick the USB device) → F4 (capture) → raise gain.
- **Overflow warnings** (`input overflow`): raise `BLOCK_MS` in the script to 100.
- **Headless/boot autostart** — `/etc/systemd/system/stt.service`:
  ```ini
  [Unit]
  Description=AssemblyAI live transcription
  After=network-online.target sound.target

  [Service]
  User=pi
  WorkingDirectory=/home/pi/pi5-assemblyai-stt
  Environment=ASSEMBLYAI_API_KEY=<your-key>
  ExecStart=/home/pi/pi5-assemblyai-stt/.venv/bin/python transcribe_mic.py --device USB --save /home/pi/transcript.txt
  Restart=always

  [Install]
  WantedBy=multi-user.target
  ```
  Then `sudo systemctl enable --now stt`.
- **Network**: streaming needs a stable outbound WSS connection to
  `streaming.assemblyai.com`; prefer Ethernet or 5 GHz Wi-Fi. A Pi 5 uses <5% CPU for
  capture since all recognition happens server-side.
