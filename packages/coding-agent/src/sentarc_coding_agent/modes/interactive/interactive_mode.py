"""
Interactive TUI mode using textual.
Full-featured terminal UI with scrollable message history, multi-line input,
real-time streaming display, tool execution visualization, and a footer status bar.
"""
from __future__ import annotations

import asyncio
from typing import Dict, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Header, TextArea

from sentarc_coding_agent.modes.interactive.components.assistant_message import AssistantMessageWidget
from sentarc_coding_agent.modes.interactive.components.footer import FooterWidget
from sentarc_coding_agent.modes.interactive.components.tool_execution import ToolExecutionWidget
from sentarc_coding_agent.modes.interactive.components.user_message import UserMessageWidget
from sentarc_coding_agent.modes.interactive.theme import DARK_THEME, InteractiveTheme


class ArcInteractiveApp(App):
    """Main interactive TUI application for arc."""

    TITLE = "arc - coding agent"

    BINDINGS = [
        Binding("ctrl+c", "abort_or_quit", "Abort/Quit", show=True),
        Binding("f1", "show_help", "Help", show=False),
        Binding("ctrl+l", "clear_screen", "Clear", show=False),
    ]

    CSS = """
    ArcInteractiveApp {
        background: $background;
    }
    #messages {
        height: 1fr;
        overflow-y: auto;
        padding: 0 1;
    }
    #input-area {
        height: auto;
        max-height: 10;
        border: solid $primary;
        margin: 0 1 1 1;
    }
    #input-editor {
        height: auto;
        min-height: 3;
        background: $surface;
    }
    FooterWidget {
        height: 1;
        background: #2d2d2d;
        color: white;
        padding: 0 1;
        dock: bottom;
    }
    """

    def __init__(self, agent_session, theme: InteractiveTheme = DARK_THEME, **kwargs):
        super().__init__(**kwargs)
        self._agent_session = agent_session
        self._theme = theme
        self._agent = agent_session if hasattr(agent_session, "subscribe") else agent_session.agent
        self._assistant_widget: Optional[AssistantMessageWidget] = None
        self._tool_widgets: Dict[str, ToolExecutionWidget] = {}
        self._is_streaming = False

        self._agent.subscribe(self._handle_agent_event)

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="messages"):
            pass
        yield TextArea("", id="input-editor")
        yield FooterWidget(id="footer-bar")

    def on_mount(self) -> None:
        self.query_one("#input-editor").focus()
        model = self._agent._state.model
        if model:
            model_name = getattr(model, "id", str(model))
            self.query_one(FooterWidget).update(model_name=model_name)

    def on_key(self, event) -> None:
        # Ctrl+J (newline char) = send message
        if event.key == "ctrl+j":
            self._send_message()
            event.prevent_default()

    def _send_message(self) -> None:
        editor = self.query_one("#input-editor", TextArea)
        text = editor.text.strip()
        if not text or self._is_streaming:
            return
        editor.clear()

        messages = self.query_one("#messages", VerticalScroll)
        user_widget = UserMessageWidget(text)
        self.call_after_refresh(messages.mount, user_widget)
        self.call_after_refresh(messages.scroll_end)

        asyncio.create_task(self._agent.prompt(text))

    def action_abort_or_quit(self) -> None:
        if self._is_streaming:
            if hasattr(self._agent, "abort"):
                self._agent.abort()
        else:
            self.exit()

    def action_show_help(self) -> None:
        self.notify("Ctrl+J: Send | Ctrl+C: Abort/Quit | F1: Help", title="Keybindings")

    def action_clear_screen(self) -> None:
        self.query_one("#messages", VerticalScroll).remove_children()

    def _handle_agent_event(self, event) -> None:
        """Handle events from the agent — called from background thread."""
        self.call_from_thread(self._dispatch_event, event)

    def _dispatch_event(self, event) -> None:
        """Dispatch agent event to UI (runs on UI thread)."""
        event_type = getattr(event, "type", None)
        messages = self.query_one("#messages", VerticalScroll)
        footer = self.query_one(FooterWidget)

        if event_type == "agent_start":
            self._is_streaming = True
            footer.update(thinking_level=self._agent._state.thinking_level)

        elif event_type == "agent_end":
            self._is_streaming = False
            self._assistant_widget = None
            footer.update()

        elif event_type == "message_start":
            msg = getattr(event, "message", None)
            if msg is not None:
                role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
                if role == "assistant":
                    self._assistant_widget = AssistantMessageWidget()
                    self._assistant_widget.is_streaming = True
                    messages.mount(self._assistant_widget)
                    messages.scroll_end(animate=False)

        elif event_type == "message_update":
            if self._assistant_widget:
                ae = getattr(event, "assistant_message_event", None)
                if ae is not None:
                    ae_type = ae.get("type") if isinstance(ae, dict) else getattr(ae, "type", None)
                    if ae_type == "text_delta":
                        delta = ae.get("delta", "") if isinstance(ae, dict) else getattr(ae, "delta", "")
                        self._assistant_widget.append_text(delta)
                        messages.scroll_end(animate=False)

        elif event_type == "message_end":
            if self._assistant_widget:
                self._assistant_widget.set_streaming(False)
                self._assistant_widget = None

        elif event_type == "tool_execution_start":
            tool_call_id = getattr(event, "tool_call_id", "")
            tool_name = getattr(event, "tool_name", "")
            args = getattr(event, "args", {})
            tw = ToolExecutionWidget(tool_name, args)
            self._tool_widgets[tool_call_id] = tw
            messages.mount(tw)
            messages.scroll_end(animate=False)

        elif event_type == "tool_execution_end":
            tool_call_id = getattr(event, "tool_call_id", "")
            tw = self._tool_widgets.pop(tool_call_id, None)
            if tw:
                tw.set_result(getattr(event, "result", None), is_error=getattr(event, "is_error", False))


async def run_interactive_mode(agent_session, theme: InteractiveTheme = DARK_THEME) -> None:
    """Start the interactive TUI."""
    app = ArcInteractiveApp(agent_session, theme=theme)
    await app.run_async()
