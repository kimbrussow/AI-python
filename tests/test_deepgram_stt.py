from urllib.parse import parse_qsl, urlsplit

from deepgram_stt import DeepgramTranscriber, TurnTracker, query_parameters


def results(transcript: str, is_final: bool = False, speech_final: bool = False) -> dict:
    return {
        "type": "Results",
        "is_final": is_final,
        "speech_final": speech_final,
        "channel": {"alternatives": [{"transcript": transcript}]},
    }


def test_interim_results_are_partial_turns():
    tracker = TurnTracker()
    assert tracker.feed(results("hey")) == ("hey", False)
    assert tracker.feed(results("hey pi")) == ("hey pi", False)


def test_speech_final_ends_the_turn():
    tracker = TurnTracker()
    tracker.feed(results("hey pi"))
    assert tracker.feed(results("hey pi what time is it", speech_final=True)) == (
        "hey pi what time is it",
        True,
    )


def test_finalized_segments_accumulate_until_end_of_speech():
    tracker = TurnTracker()
    assert tracker.feed(results("hey pi", is_final=True)) == ("hey pi", False)
    assert tracker.feed(results("turn on", is_final=False)) == ("hey pi turn on", False)
    assert tracker.feed(results("turn on the light", is_final=True)) == (
        "hey pi turn on the light",
        False,
    )
    assert tracker.feed({"type": "UtteranceEnd"}) == ("hey pi turn on the light", True)


def test_utterance_end_finalizes_when_the_vad_missed_the_pause():
    tracker = TurnTracker()
    tracker.feed(results("hey pi", is_final=True))
    assert tracker.feed({"type": "UtteranceEnd"}) == ("hey pi", True)
    # Nothing pending, so a second signal for the same utterance is not a turn.
    assert tracker.feed({"type": "UtteranceEnd"}) is None


def test_empty_and_unknown_messages_are_ignored():
    tracker = TurnTracker()
    assert tracker.feed(results("")) is None
    assert tracker.feed({"type": "Metadata", "duration": 1.5}) is None
    assert tracker.feed({"type": "Results", "channel": {}}) is None


def test_interim_text_is_dropped_when_the_speaker_is_corrected():
    tracker = TurnTracker()
    tracker.feed(results("hey pie"))
    assert tracker.feed(results("hey pi hello", speech_final=True)) == ("hey pi hello", True)


def test_flush_ends_the_utterance_when_the_socket_closes():
    tracker = TurnTracker()
    tracker.feed(results("hey pi hello", is_final=True))
    assert tracker.flush() == ("hey pi hello", True)
    assert tracker.flush() is None


def test_query_repeats_keyterm_per_term():
    parameters = query_parameters(16000, keyterms=["hey pi", "GPIO"])
    assert [value for key, value in parameters if key == "keyterm"] == ["hey pi", "GPIO"]


def test_url_carries_model_rate_and_pcm_encoding():
    transcriber = DeepgramTranscriber(api_key="k", sample_rate=48000, model="nova-3")
    parts = urlsplit(transcriber.url())
    query = dict(parse_qsl(parts.query))
    assert parts.scheme == "wss"
    assert query["model"] == "nova-3"
    assert query["sample_rate"] == "48000"
    assert query["encoding"] == "linear16"
    assert query["channels"] == "1"
    assert query["interim_results"] == "true"
