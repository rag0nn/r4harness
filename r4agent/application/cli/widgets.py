from __future__ import annotations

import logging

from textual.events import Key
from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Label, Markdown, Static, TextArea

from r4agent import Message
from r4agent.struct.base import Roles


class PromptTextArea(TextArea):
    """Textarea that maps the TUI-specific keyboard actions to the app."""

    def load_text_at_end(self, text: str) -> None:
        """Replace the text and put the cursor after the last inserted character."""
        self.load_text(text)
        lines = text.splitlines() or [""]
        self.move_cursor((len(lines) - 1, len(lines[-1])))

    async def _on_key(self, event: Key) -> None:
        """Handle submit, cancellation, and command completion before Textual input."""
        if event.key == "ctrl+b" and self.app.cancel_query():
            event.stop()
            event.prevent_default()
            return

        if event.key == "shift+enter":
            event.stop()
            event.prevent_default()
            self.app.submit_prompt(self.text)
            return

        if event.key == "tab" and self.app.complete_command_hint():
            event.stop()
            event.prevent_default()
            return

        await super()._on_key(event)


class MessageBlock(Container):
    """Render one domain message without knowing how the agent produced it."""

    def __init__(self, message: Message, tool_call=None) -> None:
        self.message = message
        self.tool_call = tool_call
        super().__init__(classes=f"message-block {message.role}")

    @staticmethod
    def format_tool_display(tool_call) -> str:
        """Normalize tuple, dictionary, and SDK tool-call shapes for display."""
        if not tool_call:
            return "tool"

        if isinstance(tool_call, (tuple, list)) and len(tool_call) >= 2:
            name = tool_call[0]
            args = tool_call[1] or {}
        elif isinstance(tool_call, dict):
            function = tool_call.get("function", {})
            name = tool_call.get("name") or function.get("name")
            args = tool_call.get("arguments") or function.get("arguments") or {}
        else:
            function = getattr(tool_call, "function", None)
            name = getattr(function, "name", None)
            args = getattr(function, "arguments", {}) or {}

        if not name:
            return "tool"
        if isinstance(args, dict) and args:
            return f"{name}: {args}"
        if args not in (None, "", {}, []):
            return f"{name}: {args}"
        return f"{name}: {{}}"

    def compose(self) -> ComposeResult:
        yield Label(self.message.role.upper(), classes="message-role")
        if self.message.role == Roles.tool:
            yield Static(
                self.format_tool_display(self.tool_call),
                classes="message-content",
                markup=False,
            )
        else:
            yield Markdown(self.message.content or "", classes="message-content")


class ToolChain(Container):
    """Visually group tool results with the assistant message that follows them."""

    def __init__(self, blocks: list[MessageBlock]) -> None:
        self.blocks = blocks
        super().__init__(classes="tool-chain")

    def compose(self) -> ComposeResult:
        yield from self.blocks


class ModelBanner(Static):
    """Display the active model names without coupling the widget to providers."""

    def __init__(self, text: str = "", *, id: str | None = None) -> None:
        self._banner_text = text
        super().__init__("", markup=False, id=id)
        self.add_class("model-banner")

    def render(self):
        return self._banner_text

    def set_text(self, text: str) -> None:
        self._banner_text = text
        self.refresh(layout=True)


class TUILogHandler(logging.Handler):
    """Forward Python log records to the Textual app thread."""

    def __init__(self, app) -> None:
        super().__init__()
        self.app = app

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        try:
            self.app.call_from_thread(self.app.append_log, message)
        except RuntimeError:
            self.app.append_log(message)
