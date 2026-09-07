from __future__ import annotations

import logging
from collections import deque

from textual.containers import VerticalScroll
from textual.widgets import Static

from .widgets import TUILogHandler


class LogController:
    """Own root logger capture and bounded log projection for the TUI."""

    def __init__(self, app, max_lines: int = 200) -> None:
        self.app = app
        self.lines: deque[str] = deque(maxlen=max_lines)
        self.previous_handlers: list[logging.Handler] = []
        self.previous_level = logging.WARNING
        self.handler: TUILogHandler | None = None

    def capture(self) -> None:
        """Replace terminal stream handlers while retaining file handlers."""
        root_logger = logging.getLogger()
        self.previous_level = root_logger.level
        root_logger.setLevel(logging.INFO)
        self.previous_handlers = list(root_logger.handlers)
        self.handler = TUILogHandler(self.app)
        self.handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        for handler in self.previous_handlers:
            if not isinstance(handler, logging.FileHandler):
                root_logger.removeHandler(handler)
        root_logger.addHandler(self.handler)

    def append(self, message: str) -> None:
        self.lines.append(message)
        self.render()

    def render(self) -> None:
        """Refresh the visible log list and keep the newest entry in view."""
        try:
            logs_view = self.app.query_one("#logs-view", VerticalScroll)
        except Exception:
            return

        logs_view.remove_children()
        for line in self.lines:
            logs_view.mount(Static(line, markup=False, classes="log-line"))
        logs_view.scroll_end(animate=False)

    def restore(self) -> None:
        """Restore logger handlers and level after the TUI exits."""
        root_logger = logging.getLogger()
        if self.handler is not None:
            root_logger.removeHandler(self.handler)
        for handler in self.previous_handlers:
            if handler not in root_logger.handlers:
                root_logger.addHandler(handler)
        root_logger.setLevel(self.previous_level)
