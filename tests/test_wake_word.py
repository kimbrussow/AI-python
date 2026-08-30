import pytest

from wake_word import WakeWordDetector, normalize, strip_wake_words


def test_normalize_drops_punctuation_and_case():
    assert normalize("Hey, Pi! What's up?") == ["hey", "pi", "what's", "up"]


def test_detects_wake_word_and_returns_command():
    detection = WakeWordDetector().detect("Hey Pi, what time is it?")
    assert detection is not None
    assert detection.phrase == "hey pi"
    assert detection.command == "what time is it"


def test_bare_wake_word_yields_empty_command():
    detection = WakeWordDetector().detect("Hey Pi.")
    assert detection is not None
    assert detection.command == ""


@pytest.mark.parametrize("transcript", ["hey pie", "Hay pi", "hey py"])
def test_accepts_common_mistranscriptions(transcript):
    assert WakeWordDetector().detect(transcript) is not None


@pytest.mark.parametrize("transcript", ["hey there", "pi is tasty", "turn on the light"])
def test_rejects_unrelated_speech(transcript):
    assert WakeWordDetector().detect(transcript) is None


def test_threshold_can_require_exact_words():
    strict = WakeWordDetector(threshold=1.0)
    assert strict.detect("hey pie") is None
    assert strict.detect("hey pi go") is not None


def test_wake_word_mid_sentence_uses_last_occurrence():
    detection = WakeWordDetector().detect("hey pi hello hey pi what is the date")
    assert detection is not None
    assert detection.command == "what is the date"


def test_multiple_phrases_prefer_longest_match():
    detector = WakeWordDetector(phrases=("hey pi", "hey pi five"))
    detection = detector.detect("hey pi five turn off the lamp")
    assert detection is not None
    assert detection.phrase == "hey pi five"
    assert detection.command == "turn off the lamp"


def test_keyterms_are_the_configured_phrases():
    assert WakeWordDetector(phrases=("hey pi", "computer")).keyterms() == ["hey pi", "computer"]


def test_empty_phrase_list_rejected():
    with pytest.raises(ValueError):
        WakeWordDetector(phrases=())


def test_punctuation_only_phrase_rejected():
    with pytest.raises(ValueError):
        WakeWordDetector(phrases=("!!!",))


def test_strip_wake_words_without_match_returns_normalized_text():
    assert strip_wake_words("Turn off the lamp!", ["hey pi"]) == "turn off the lamp"
