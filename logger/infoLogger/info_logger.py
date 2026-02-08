"""
Lightweight logger for training progress and runtime messages.
"""
import sys
from datetime import datetime
from typing import Optional


class InfoLogger:
    """Simple text logger with timestamp and log level."""

    def __init__(self, name: str = "FD-DB", enabled: bool = True):
        self.name = name
        self.enabled = enabled

    def _format_message(self, level: str, message: str) -> str:
        """Format a message with timestamp, logger name, and level."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] [{self.name}] [{level}] {message}"

    def info(self, message: str):
        """Print an INFO message."""
        if self.enabled:
            print(self._format_message("INFO", message), file=sys.stdout)

    def debug(self, message: str):
        """Print a DEBUG message."""
        if self.enabled:
            print(self._format_message("DEBUG", message), file=sys.stdout)

    def warning(self, message: str):
        """Print a WARNING message."""
        if self.enabled:
            print(self._format_message("WARNING", message), file=sys.stdout)

    def error(self, message: str):
        """Print an ERROR message."""
        if self.enabled:
            print(self._format_message("ERROR", message), file=sys.stderr)

    def progress(self, epoch: int, step: int, **metrics):
        """Print a training progress line with optional metrics."""
        if not self.enabled:
            return

        msg = f"Epoch {epoch} | Step {step}"
        for key, value in metrics.items():
            if hasattr(value, "item"):
                msg += f" | {key}: {value.item():.4f}"
            elif isinstance(value, (int, float)):
                msg += f" | {key}: {value:.4f}" if isinstance(value, float) else f" | {key}: {value}"
            else:
                msg += f" | {key}: {value}"

        print(self._format_message("PROGRESS", msg), file=sys.stdout)


# Global default logger instance.
default_logger = InfoLogger()


def get_logger(name: Optional[str] = None) -> InfoLogger:
    """Get the default logger or create a named logger."""
    if name is None:
        return default_logger
    return InfoLogger(name=name)
