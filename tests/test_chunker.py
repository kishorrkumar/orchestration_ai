"""
Tests for Streaming Clause Chunker and Indian English Normalization.
"""

from orchestration.chunker.bridge import ClauseChunker, normalize_indian_english_text


def test_currency_and_text_normalization():
    raw1 = "I want to pay ₹5000 for the flight."
    norm1 = normalize_indian_english_text(raw1)
    assert "five thousand rupees" in norm1.lower()

    raw2 = "The package costs 12500 INR."
    norm2 = normalize_indian_english_text(raw2)
    assert "twelve thousand" in norm2.lower()
    assert "rupees" in norm2.lower()

    raw3 = "We use AI and ML in Bengaluru."
    norm3 = normalize_indian_english_text(raw3)
    assert "A.I." in norm3
    assert "M.L." in norm3


def test_clause_chunker_first_chunk_early_flush():
    chunker = ClauseChunker(first_chunk_min_words=2, first_chunk_max_words=6)

    # Feeding first few words with comma
    c1 = chunker.feed_token("Namaste Priya, ")
    assert len(c1) == 1
    assert "Namaste Priya," in c1[0]

    # Feeding incomplete subsequent clause
    c2 = chunker.feed_token("how can I help")
    assert len(c2) == 0  # no punctuation yet and < min_words

    # Now finish clause with question mark
    c3 = chunker.feed_token(" you today with your schedule?")
    assert len(c3) == 1
    assert "schedule?" in c3[0]


def test_clause_chunker_sentence_boundaries():
    chunker = ClauseChunker()
    tokens = ["Why", "do", "programmers", "prefer", "dark", "mode?", "Because", "light", "attracts", "bugs!"]
    emitted = []
    for token in tokens:
        res = chunker.feed_token(token + " ")
        emitted.extend(res)
    emitted.extend(chunker.flush())

    assert len(emitted) >= 2
    assert "dark mode?" in emitted[0]
    assert "bugs!" in emitted[1]
