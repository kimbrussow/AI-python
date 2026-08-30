import io
import json
import wave

import pytest

import tts
from tts import Speaker, TTSError, _decode_wav


@pytest.fixture
def espeak_installed(monkeypatch):
    monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/espeak-ng" if name == "espeak-ng" else None)


def wav_bytes(pcm: bytes, rate: int = 16000, channels: int = 1, width: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def test_decode_wav_returns_pcm_rate_and_channels():
    pcm, rate, channels = _decode_wav(wav_bytes(b"\x01\x00\x02\x00", rate=22050))
    assert (pcm, rate, channels) == (b"\x01\x00\x02\x00", 22050, 1)


def test_decode_wav_rejects_8_bit_audio():
    with pytest.raises(TTSError):
        _decode_wav(wav_bytes(b"\x01\x02", width=1))


def test_espeak_command_includes_voice_and_rate(espeak_installed):
    speaker = Speaker(voice="en-gb", words_per_minute=200)
    assert speaker.backend == "espeak-ng"
    assert speaker.command() == ["espeak-ng", "-v", "en-gb", "-s", "200", "--stdout"]


def test_synthesize_decodes_backend_wav(espeak_installed, monkeypatch):
    monkeypatch.setattr(tts, "_run", lambda command, stdin_text=None: wav_bytes(b"\x00\x01"))
    assert Speaker().synthesize("hello") == (b"\x00\x01", 16000, 1)


def test_say_ignores_blank_text(espeak_installed, monkeypatch):
    monkeypatch.setattr(tts, "_run", lambda *a, **k: pytest.fail("should not synthesize"))
    Speaker().say("   ")


def test_missing_backend_is_reported(monkeypatch):
    monkeypatch.setattr(tts.shutil, "which", lambda name: None)
    with pytest.raises(TTSError, match="No text-to-speech backend"):
        Speaker()


def test_explicit_backend_must_be_installed(monkeypatch):
    monkeypatch.setattr(tts.shutil, "which", lambda name: None)
    with pytest.raises(TTSError, match="espeak-ng is not installed"):
        Speaker(backend="espeak-ng")


def test_piper_requires_an_existing_model(monkeypatch):
    monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/piper")
    with pytest.raises(TTSError, match="--piper-model is required"):
        Speaker(backend="piper")
    with pytest.raises(TTSError, match="model not found"):
        Speaker(backend="piper", piper_model="/nope/voice.onnx")


def test_piper_rate_comes_from_sidecar_config(tmp_path, monkeypatch):
    monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/piper")
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"")
    model.with_suffix(".onnx.json").write_text(json.dumps({"audio": {"sample_rate": 16000}}))
    speaker = Speaker(backend="piper", piper_model=str(model))
    monkeypatch.setattr(tts, "_run", lambda command, stdin_text=None: b"\x00\x01")
    assert speaker.command()[:3] == ["piper", "--model", str(model)]
    assert speaker.synthesize("hi") == (b"\x00\x01", 16000, 1)


def test_piper_rate_falls_back_without_config(tmp_path, monkeypatch):
    monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/piper")
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"")
    speaker = Speaker(backend="piper", piper_model=str(model))
    monkeypatch.setattr(tts, "_run", lambda command, stdin_text=None: b"")
    assert speaker.synthesize("hi")[1] == tts.PIPER_FALLBACK_RATE


def test_run_reports_backend_failure():
    with pytest.raises(TTSError, match="failed"):
        tts._run(["sh", "-c", "echo boom >&2; exit 3"])


def test_run_reports_empty_output():
    with pytest.raises(TTSError, match="no audio"):
        tts._run(["sh", "-c", "true"])
