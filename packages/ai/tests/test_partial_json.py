import pytest
from sentarc_ai.utils.partial_json import parse_streaming_json

def test_parse_streaming_json_complete():
    # Valid complete json
    res = parse_streaming_json('{"name": "Alice"}')
    assert res == {"name": "Alice"}

def test_parse_streaming_json_partial():
    # Incomplete json should be safely repaired
    res = parse_streaming_json('{"name": "Al')
    assert isinstance(res, dict)
    assert res.get("name") == "Al"

def test_parse_streaming_json_arrays():
    res = parse_streaming_json('{"items": [1, 2')
    assert "items" in res
    assert res["items"] == [1, 2]

def test_parse_streaming_json_empty():
    res = parse_streaming_json('')
    assert res == {}
    
def test_parse_streaming_json_broken_syntax():
    # Badly broken
    res = parse_streaming_json('{[')
    # Should safely return dict
    assert isinstance(res, dict)
