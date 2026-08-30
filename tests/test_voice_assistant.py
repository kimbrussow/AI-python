import queue

import pytest

from voice_assistant import Assistant, Responder, builtin_response
from wake_word import WakeWordDetector


class FakeSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


def make_assistant(follow_up=True):
    speaker = FakeSpeaker()
    assistant = Assistant(
        detector=WakeWordDetector(),
        speaker=speaker,
        responder=Responder(),
        follow_up=follow_up,
    )
    return assistant, speaker


def spoken(assistant):
    """Replies queued for playback, without starting the speech thread."""
    out = []
    while True:
        try:
            out.append(assistant.speech_queue.get_nowait())
        except queue.Empty:
            return out


def test_command_after_wake_word_is_answered():
    assistant, _ = make_assistant()
    assistant.handle_turn("Hey Pi, turn on the lamp.")
    assert spoken(assistant) == ["You said: turn on the lamp"]


def test_speech_without_wake_word_is_ignored():
    assistant, _ = make_assistant()
    assistant.handle_turn("The kettle is boiling.")
    assert spoken(assistant) == []


def test_bare_wake_word_asks_for_the_command_then_answers_it():
    assistant, _ = make_assistant()
    assistant.handle_turn("Hey Pi")
    assert spoken(assistant) == ["Yes?"]
    assert assistant.awaiting_command
    assistant.handle_turn("what time is it")
    assert not assistant.awaiting_command
    assert spoken(assistant)[0].startswith("It is ")


def test_follow_up_can_be_disabled():
    assistant, _ = make_assistant(follow_up=False)
    assistant.handle_turn("Hey Pi")
    assert not assistant.awaiting_command
    assistant.handle_turn("what time is it")
    assert spoken(assistant) == ["Yes?"]


def test_exit_phrase_stops_the_loop():
    assistant, _ = make_assistant()
    assistant.handle_turn("Hey Pi, goodbye")
    assert spoken(assistant) == ["Goodbye."]
    assert assistant.stop_event.is_set()
    assert assistant.audio_queue.get_nowait() is None


def test_microphone_audio_is_dropped_while_speaking():
    assistant, _ = make_assistant()
    assistant.audio_callback(b"\x01\x00", 1, None, None)
    assistant.speaking.set()
    assistant.audio_callback(b"\x02\x00", 1, None, None)
    assert assistant.audio_queue.qsize() == 1


def test_drain_audio_keeps_the_stop_sentinel():
    assistant, _ = make_assistant()
    assistant.audio_queue.put(b"\x01\x00")
    assistant.audio_queue.put(None)
    assistant._drain_audio()
    assert assistant.audio_queue.get_nowait() is None


def test_speech_worker_plays_queued_text_and_clears_speaking_flag():
    assistant, speaker = make_assistant()
    assistant.start()
    assistant.speak("hello there")
    assistant.finish()
    assert speaker.said == ["hello there"]
    assert not assistant.speaking.is_set()


@pytest.mark.parametrize("command", ["what time is it", "tell me the time"])
def test_builtin_time_response(command):
    assert builtin_response(command).startswith("It is ")


def test_builtin_date_response():
    assert builtin_response("what is the date").startswith("Today is ")


def test_external_responder_receives_command_on_stdin():
    responder = Responder("tr '[:lower:]' '[:upper:]'")
    assert responder.reply("hello pi") == "HELLO PI"


def test_external_responder_failure_is_spoken_not_raised():
    assert Responder("exit 1").reply("hello") == "The response handler failed."


def test_external_responder_empty_output_has_a_fallback():
    assert Responder("true").reply("hello") == "I have no answer for that."


def test_external_responder_timeout(monkeypatch):
    monkeypatch.setattr("voice_assistant.RESPONSE_TIMEOUT_S", 0.2)
    assert Responder("sleep 5").reply("hello") == "The response handler timed out."
