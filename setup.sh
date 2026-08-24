#!/usr/bin/env bash
# One-time setup on a Raspberry Pi 5 (Raspberry Pi OS Bookworm, 64-bit).
set -euo pipefail

cd "$(dirname "$0")"

sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libportaudio2 portaudio19-dev alsa-utils

python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

echo
echo "Detected ALSA capture devices:"
arecord -l || true
echo
echo "Next:"
echo "  export ASSEMBLYAI_API_KEY=<your-key>"
echo "  ./.venv/bin/python transcribe_mic.py --list-devices"
echo "  ./.venv/bin/python transcribe_mic.py --device USB"
