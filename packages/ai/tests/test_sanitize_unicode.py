import pytest
from sentarc_ai.utils.sanitize_unicode import sanitize_surrogates

def test_sanitize_surrogates():
    # Valid text should remain unchanged
    assert sanitize_surrogates("Hello World") == "Hello World"
    
    # Valid emojis (using proper Unicode encoding, not loose surrogates strings)
    assert sanitize_surrogates("Hello 🙈 World") == "Hello 🙈 World"
    
    # Isolated high surrogate (e.g., D83D from an improperly split string)
    assert sanitize_surrogates("Hello \ud83d World") == "Hello  World"
    assert sanitize_surrogates("Start \ud83d") == "Start "
    
    # Isolated low surrogate (e.g., DE48)
    assert sanitize_surrogates("Hello \ude48 World") == "Hello  World"
    assert sanitize_surrogates("\ude48 End") == " End"
    
    # Multiple isolated surrogates sprinkled in text
    assert sanitize_surrogates("\ud83d Isolated \ude48 surrogates \ud83d everywhere \ude48") == " Isolated  surrogates  everywhere "
    
    # Ensure a valid surrogate pair written with escapes survives if Python parses it as a full char
    text = "Valid \ud83d\ude48 Pair"
    assert sanitize_surrogates(text) == text
    
    # Edge cases
    assert sanitize_surrogates("") == ""
    assert sanitize_surrogates("\ud800") == ""  # Lowest high surrogate
    assert sanitize_surrogates("\udbff") == ""  # Highest high surrogate
    assert sanitize_surrogates("\udc00") == ""  # Lowest low surrogate
    assert sanitize_surrogates("\udfff") == ""  # Highest low surrogate
