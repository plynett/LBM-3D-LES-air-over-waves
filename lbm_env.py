"""Small environment parsing helpers for the LBM driver."""

from __future__ import annotations

import os


def env_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    value = os.environ.get(name, default).lower()
    if value not in choices:
        allowed = ", ".join(repr(choice) for choice in choices)
        raise ValueError(f"{name} must be one of: {allowed}")
    return value


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))
