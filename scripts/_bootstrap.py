"""Shared path + env bootstrap for the CLI scripts.

Every script starts with:

    from _bootstrap import setup
    setup()

which makes the `thinkspark` package importable regardless of the working directory and
loads the project `.env` into the environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup() -> Path:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from thinkspark.config import load_env
    load_env(PROJECT_ROOT / ".env")
    return PROJECT_ROOT
