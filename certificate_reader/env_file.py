from __future__ import annotations

import os
import re
from pathlib import Path


ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class EnvFileError(ValueError):
    """Raised when a dotenv-style file is malformed."""


def load_env_file(path: Path, *, override: bool = False) -> int:
    if not path.exists():
        return 0
    if not path.is_file():
        raise EnvFileError(f"Environment file is not a file: {path}")

    loaded = 0
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise EnvFileError(f"Invalid environment line {line_number}: missing '='")
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        if not ENV_KEY_PATTERN.fullmatch(key):
            raise EnvFileError(f"Invalid environment key on line {line_number}: {key}")
        if key in os.environ and not override:
            continue
        os.environ[key] = _parse_value(value.strip(), line_number)
        loaded += 1
    return loaded


def _parse_value(value: str, line_number: int) -> str:
    if len(value) < 2:
        return value
    quote = value[0]
    if quote not in {"'", '"'}:
        return value
    if value[-1] != quote:
        raise EnvFileError(f"Invalid quoted value on line {line_number}: missing closing quote")
    parsed = value[1:-1]
    if quote == '"':
        return bytes(parsed, "utf-8").decode("unicode_escape")
    return parsed
