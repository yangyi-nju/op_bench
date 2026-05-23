from __future__ import annotations

import sys
import time
from collections.abc import Callable


Progress = Callable[[str], None]


def noop_progress(message: str) -> None:
    return None


class ProgressLogger:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def __call__(self, message: str) -> None:
        if not self.enabled:
            return
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}m{remainder:04.1f}s"


def format_command(command: list[str], max_chars: int = 220) -> str:
    text = " ".join(_quote_part(part) for part in command)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _quote_part(part: str) -> str:
    if not part:
        return "''"
    if any(character.isspace() for character in part):
        return repr(part)
    return part
