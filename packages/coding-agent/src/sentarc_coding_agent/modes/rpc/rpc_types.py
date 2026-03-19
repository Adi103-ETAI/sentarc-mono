"""
RPC protocol types for headless operation.
Commands sent as JSON lines on stdin. Responses emitted as JSON lines on stdout.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict, Union


class RpcPromptCommand(TypedDict, total=False):
    id: str
    type: Literal["prompt"]
    message: str
    images: list
    streaming_behavior: str  # "steer" | "followUp"


class RpcSteerCommand(TypedDict, total=False):
    id: str
    type: Literal["steer"]
    message: str


class RpcAbortCommand(TypedDict, total=False):
    id: str
    type: Literal["abort"]


class RpcGetStateCommand(TypedDict, total=False):
    id: str
    type: Literal["get_state"]


class RpcSetModelCommand(TypedDict, total=False):
    id: str
    type: Literal["set_model"]
    provider: str
    model_id: str


class RpcGetMessagesCommand(TypedDict, total=False):
    id: str
    type: Literal["get_messages"]


class RpcSetThinkingLevelCommand(TypedDict, total=False):
    id: str
    type: Literal["set_thinking_level"]
    level: str


class RpcNewSessionCommand(TypedDict, total=False):
    id: str
    type: Literal["new_session"]
    parent_session: Optional[str]


class RpcGetAvailableModelsCommand(TypedDict, total=False):
    id: str
    type: Literal["get_available_models"]


class RpcGetLastAssistantTextCommand(TypedDict, total=False):
    id: str
    type: Literal["get_last_assistant_text"]


RpcCommand = Union[
    RpcPromptCommand,
    RpcSteerCommand,
    RpcAbortCommand,
    RpcGetStateCommand,
    RpcSetModelCommand,
    RpcGetMessagesCommand,
    RpcSetThinkingLevelCommand,
    RpcNewSessionCommand,
    RpcGetAvailableModelsCommand,
    RpcGetLastAssistantTextCommand,
]


class RpcSessionState(TypedDict, total=False):
    model: Any
    thinking_level: str
    is_streaming: bool
    session_id: Optional[str]
    message_count: int


def make_response(
    command_type: str,
    success: bool,
    data: Any = None,
    error: Optional[str] = None,
    id: Optional[str] = None,
) -> dict:
    resp: dict = {"type": "response", "command": command_type, "success": success}
    if id is not None:
        resp["id"] = id
    if success and data is not None:
        resp["data"] = data
    if not success and error is not None:
        resp["error"] = error
    return resp
