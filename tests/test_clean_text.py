from src.ingestion.clean_text import clean_text


def test_clean_text_removes_punctuation_and_lowercases():
    raw = "The quick brown FOXES were jumping over lazy dogs near the riverbank!!!"
    expected = "the quick brown foxes were jumping over lazy dogs near the riverbank"
    assert clean_text(raw) == expected


def test_clean_text_collapses_whitespace():
    raw = "Hello   world\nthis\t is \r\n spaced"
    cleaned = clean_text(raw)
    assert cleaned == "hello world this is spaced"
