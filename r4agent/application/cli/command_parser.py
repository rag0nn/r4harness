from __future__ import annotations


class CommandParser:
    """Parse the slash-command fragments understood by the TUI."""

    COMMANDS = ("/reset", "/save", "/load", "/change", "/select", "/exit")
    ARGUMENT_COMMANDS = ("/load", "/change", "/select")

    @classmethod
    def looks_like_command(cls, text: str) -> bool:
        """Return whether text is a supported command rather than a file path."""
        candidate = text.strip()
        if not candidate.startswith("/"):
            return False
        if candidate in cls.COMMANDS:
            return True
        if candidate.startswith(tuple(f"{command} " for command in cls.ARGUMENT_COMMANDS)):
            return True
        tail = candidate[1:]
        return "/" not in tail and "." not in tail

    @staticmethod
    def fragment(text: str) -> str | None:
        """Extract the slash fragment on the last prompt line, if one exists."""
        lines = text.splitlines()
        last_line = (lines[-1] if lines else "").strip()
        if last_line.startswith("/"):
            return last_line

        separator = last_line.rfind(" /")
        if separator >= 0:
            return last_line[separator + 1:]
        return None
