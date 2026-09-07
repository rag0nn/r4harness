from __future__ import annotations

from pathlib import Path

from .handler import Handler


class SessionController:
    """Provide the TUI with a narrow session persistence interface."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler

    def list_chats(self) -> list[Path]:
        """Return only regular chat files exposed to the load picker."""
        return sorted(
            (path for path in self.handler.list_chats() if path.is_file()),
            key=lambda path: path.name,
        )

    def save(self) -> None:
        self.handler.save_chat()

    def load(self, path: Path) -> None:
        self.handler.load_chat(path)

    def reset(self) -> None:
        self.handler.reset_chat()
