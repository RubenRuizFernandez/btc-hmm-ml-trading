"""Minimal .env loader for the MT5 runner."""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_file(path: str | Path, override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file into the process environment."""
    env_path = Path(path)
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _parse_env_value(value.strip())
        if not key:
            continue

        if key in os.environ and not override:
            loaded[key] = os.environ[key]
            continue

        os.environ[key] = value
        loaded[key] = value

    return loaded


def _parse_env_value(value: str) -> str:
    if not value:
        return ""

    if value[0] in {"'", '"'} and value[-1] == value[0]:
        return value[1:-1]

    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value
