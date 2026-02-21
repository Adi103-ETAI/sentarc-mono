import json
from typing import Any, Dict, List
import jsonschema
from jsonschema import validate, ValidationError

from ..types import Tool, ToolUseContent, ToolCallContent

ToolCallType = ToolUseContent | ToolCallContent

def validate_tool_call(tools: List[Tool], tool_call: ToolCallType) -> Dict[str, Any]:
    """
    Finds a tool by name and validates the tool call arguments against its JSON schema.
    
    Args:
        tools: List of tool definitions.
        tool_call: The tool call from the LLM.
        
    Returns:
        The validated arguments dictionary.
        
    Raises:
        ValueError: If the tool is not found.
        jsonschema.ValidationError: If validation fails, with a formatted message.
    """
    tool = next((t for t in tools if t.name == tool_call.name), None)
    if not tool:
        raise ValueError(f'Tool "{tool_call.name}" not found')
        
    return validate_tool_arguments(tool, tool_call)

def validate_tool_arguments(tool: Tool, tool_call: ToolCallType) -> Dict[str, Any]:
    """
    Validates tool call arguments against the tool's JSON schema.
    
    Args:
        tool: The tool definition with JSON schema.
        tool_call: The tool call from the LLM.
        
    Returns:
        The validated arguments dictionary.
        
    Raises:
        jsonschema.ValidationError: With a formatted message if validation fails.
    """
    try:
        # Load string arguments if it's a JSON string
        args = tool_call.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse tool arguments as JSON: {e}")
                
        # jsonschema validates in place and returns None if successful
        validate(instance=args, schema=tool.parameters)
        return args
    except ValidationError as e:
        # Format validation errors nicely to match TS behavior
        path = ".".join([str(p) for p in e.path]) if e.path else "root"
        error_list = f"  - {path}: {e.message}"
        
        error_message = (
            f'Validation failed for tool "{tool_call.name}":\n'
            f'{error_list}\n\n'
            f'Received arguments:\n'
            f'{json.dumps(tool_call.arguments, indent=2)}'
        )
        raise ValidationError(error_message)
