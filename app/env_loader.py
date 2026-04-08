from __future__ import annotations

import os
from pathlib import Path


def load_local_env() -> None:
    """Load key=value pairs from project .env into process env if missing."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        if key:
            current = os.environ.get(key, "")
            if not current.strip():
                os.environ[key] = value
