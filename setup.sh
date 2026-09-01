#!/usr/bin/env bash
# One-time setup on a Raspberry Pi 5 (Raspberry Pi OS Bookworm, 64-bit).
set -euo pipefail

cd "$(dirname "$0")"

sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libportaudio2 portaudio19-dev alsa-utils \
  espeak-ng

python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

echo
echo "Detected ALSA capture devices:"
arecord -l || true
echo
echo "Detected ALSA playback devices:"
aplay -l || true
echo
echo "Next:"
echo "  export DEEPGRAM_API_KEY=<your-key>   # or ASSEMBLYAI_API_KEY with --provider assemblyai"
echo "  ./.venv/bin/python transcribe_mic.py --list-devices"
echo "  ./.venv/bin/python transcribe_mic.py --device USB"
echo "  ./.venv/bin/python voice_assistant.py --device USB --speaker USB"
