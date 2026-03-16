"""
Base types and protocols for sentarc-tui components.
"""
from typing import Protocol, runtime_checkable
from abc import abstractmethod


@runtime_checkable
class Component(Protocol):
    """Base component interface — all TUI components must implement render()."""

    @abstractmethod
    def render(self, width: int) -> list[str]:
        """Render the component into a list of lines, each at most `width` chars wide."""
        ...

    def handle_input(self, data: str) -> None:
        """Handle raw input data directed at this component."""
        ...

    def invalidate(self) -> None:
        """Mark the component as needing a re-render."""
        ...


@runtime_checkable
class Focusable(Protocol):
    """Mixin protocol for components that can receive keyboard focus."""

    focused: bool


class OverlayOptions:
    """Options for showing a component in an overlay/modal layer."""

    def __init__(self, title: str = "", closeable: bool = True) -> None:
        self.title = title
        self.closeable = closeable

    def __repr__(self) -> str:
        return f"OverlayOptions(title={self.title!r}, closeable={self.closeable})"
