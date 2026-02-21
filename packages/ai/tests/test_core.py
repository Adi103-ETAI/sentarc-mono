import pytest
import jsonschema
from sentarc_ai.types import Tool, ToolCallContent
from sentarc_ai.utils.validation import validate_tool_arguments, validate_tool_call
from sentarc_ai.models import resolve_model, list_models

def test_validation_success():
    tool = Tool(
        name="weather",
        description="Get weather",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    )
    
    call = ToolCallContent(id="1", name="weather", arguments={"location": "London"})
    
    # Should not throw
    result = validate_tool_arguments(tool, call)
    assert result["location"] == "London"

def test_validation_failure():
    tool = Tool(
        name="weather",
        description="Get weather",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    )
    
    # Missing required argument
    call = ToolCallContent(id="1", name="weather", arguments={})
    
    with pytest.raises(jsonschema.ValidationError):
        validate_tool_arguments(tool, call)

def test_validate_tool_call_finds_tool():
    tools = [
        Tool(name="a", description="", parameters={"type": "object"}),
        Tool(name="b", description="", parameters={"type": "object", "properties": {"val": {"type": "number"}}})
    ]
    call = ToolCallContent(id="1", name="b", arguments={"val": 42})
    
    result = validate_tool_call(tools, call)
    assert result["val"] == 42
    
def test_validate_tool_call_missing_tool():
    tools = [
        Tool(name="a", description="", parameters={"type": "object"})
    ]
    call = ToolCallContent(id="1", name="b", arguments={})
    
    with pytest.raises(ValueError, match='Tool "b" not found'):
        validate_tool_call(tools, call)

def test_resolve_model_finds_default():
    model = resolve_model("openai", "gpt-4o")
    assert model.id == "gpt-4o"
    assert model.provider == "openai"
    assert model.api == "openai"
    assert model.context_window > 0

def test_resolve_model_creates_unknown():
    # An entirely undefined proxy or self-hosted model
    model = resolve_model("custom-ollama", "llama-3")
    assert model.id == "llama-3"
    assert model.provider == "custom-ollama"
    # Defaults to provider fallback
    assert model.api == "custom-ollama"
