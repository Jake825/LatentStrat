"""Environment loading helpers."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_environment(path: str | Path | None = None) -> bool:
    """Load local environment variables without overriding the OS environment."""

    return load_dotenv(dotenv_path=path, override=False)
