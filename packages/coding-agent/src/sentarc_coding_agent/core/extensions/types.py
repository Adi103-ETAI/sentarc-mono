"""Extension system types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ExtensionFlag:
    """A CLI flag registered by an extension."""
    type: str  # "boolean" | "string"
    description: str = ""


@dataclass
class ExtensionCommand:
    """A slash command registered by an extension."""
    name: str
    description: str
    handler: Callable  # async (args: str, ctx: ExtensionContext) -> None


@dataclass
class Extension:
    """A loaded extension module."""
    name: str
    path: str
    flags: Dict[str, ExtensionFlag] = field(default_factory=dict)
    # Lifecycle hooks
    on_start: Optional[Callable] = None
    on_message: Optional[Callable] = None
    on_tool_call: Optional[Callable] = None
    on_session_end: Optional[Callable] = None
    # Event hooks (pi-mono style)
    on_agent_start: Optional[Callable] = None
    on_agent_end: Optional[Callable] = None
    on_before_agent_start: Optional[Callable] = None
    on_session_switch: Optional[Callable] = None
    on_session_start: Optional[Callable] = None
    # Registered commands
    commands: List[ExtensionCommand] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtensionContext:
    """Context passed to extension hooks."""
    args: Dict[str, Any]
    cwd: str
    session_manager: Any
    agent: Any
    has_ui: bool = False
    notify: Optional[Callable] = None  # (message: str, level: str) -> None
