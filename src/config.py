from __future__ import annotations

from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_config(path: Path) -> dict[str, Any]:
    """Load a flat YAML config without requiring PyYAML in the bootstrap env."""
    try:
        import yaml

        return yaml.safe_load(path.read_text())
    except ModuleNotFoundError:
        config: dict[str, Any] = {}
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", maxsplit=1)
            config[key.strip()] = _parse_scalar(value)
        return config

