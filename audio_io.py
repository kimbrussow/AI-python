"""Device discovery and sample-rate negotiation for PortAudio (sounddevice) I/O."""

import sounddevice as sd

PREFERRED_SAMPLE_RATE = 16000
CAPTURE_RATE_CANDIDATES = (PREFERRED_SAMPLE_RATE, 48000, 44100, 32000, 8000)


def list_devices() -> None:
    print(sd.query_devices())


def resolve_device(device: str | None, kind: str = "input") -> int | None:
    """Turn an index or a name substring (e.g. 'USB') into a PortAudio device index."""
    if device is None:
        return None
    try:
        return int(device)
    except ValueError:
        pass
    channels_key = "max_input_channels" if kind == "input" else "max_output_channels"
    matches = [
        index
        for index, info in enumerate(sd.query_devices())
        if device.lower() in info["name"].lower() and info[channels_key] > 0
    ]
    if not matches:
        raise SystemExit(f"No {kind} device matching {device!r}. Run with --list-devices.")
    return matches[0]


def pick_sample_rate(device: int | None) -> int:
    """16 kHz keeps bandwidth low, but many USB mics only run at 44.1/48 kHz."""
    for rate in CAPTURE_RATE_CANDIDATES:
        try:
            sd.check_input_settings(device=device, channels=1, samplerate=rate, dtype="int16")
            return rate
        except (sd.PortAudioError, ValueError):
            continue
    raise SystemExit("Microphone does not support any usable sample rate.")
