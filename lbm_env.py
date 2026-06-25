"""Small environment parsing helpers for the LBM driver."""

from __future__ import annotations

import os
from pathlib import Path


def env_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    value = os.environ.get(name, default).lower()
    if value not in choices:
        allowed = ", ".join(repr(choice) for choice in choices)
        raise ValueError(f"{name} must be one of: {allowed}")
    return value


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def project_root() -> Path:
    return Path(__file__).resolve().parent


def taichi_cache_path() -> Path:
    value = os.environ.get("LBM_TI_CACHE_DIR") or os.environ.get("TI_OFFLINE_CACHE_FILE_PATH")
    path = Path(value).expanduser() if value else project_root() / ".taichi_cache" / "ticache"
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def taichi_cache_config() -> dict[str, object]:
    path = taichi_cache_path()
    return {
        "offline_cache": True,
        "offline_cache_file_path": str(path),
    }
