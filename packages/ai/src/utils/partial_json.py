import json
from json_repair import repair_json

def parse_streaming_json(partial_json: str | None) -> dict:
    """
    Attempts to parse potentially incomplete JSON during streaming.
    Always returns a valid object, even if the JSON is incomplete.
    """
    if not partial_json or not partial_json.strip():
        return {}
    
    # Try standard parsing first
    try:
        return json.loads(partial_json)
    except json.JSONDecodeError:
        # Try repairing
        try:
            repaired = repair_json(partial_json, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
            return {}
        except Exception:
            return {}
