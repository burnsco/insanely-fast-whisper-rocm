"""Helpers for locating and loading project environment files."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT_ENV_FILE = PROJECT_ROOT / ".env"
USER_CONFIG_DIR = Path.home() / ".config" / "insanely-fast-whisper-rocm"
USER_ENV_FILE = USER_CONFIG_DIR / ".env"

_cli_debug_mode = "--debug" in sys.argv
USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT_ENV_EXISTS = PROJECT_ROOT_ENV_FILE.exists()
USER_ENV_EXISTS = USER_ENV_FILE.exists()
_env_debug_mode_temp = os.getenv("LOG_LEVEL", "").upper() == "DEBUG"

SHOW_DEBUG_PRINTS = _cli_debug_mode or _env_debug_mode_temp

if SHOW_DEBUG_PRINTS:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logging.getLogger("insanely_fast_whisper_rocm").setLevel(logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("torio").setLevel(logging.WARNING)


def debug_print(message: str) -> None:
    """Log environment-loading messages when debug output is enabled.

    Args:
        message: The message to log.
    """
    if SHOW_DEBUG_PRINTS:
        logger.debug(message)


def load_project_env() -> None:
    """Load project and user ``.env`` files in precedence order.

    The project root ``.env`` is loaded first, followed by the user-specific
    config file. The user file therefore wins when both define the same key.
    """
    if PROJECT_ROOT_ENV_EXISTS:
        debug_print(f"Loading project .env: {PROJECT_ROOT_ENV_FILE}")
        load_dotenv(PROJECT_ROOT_ENV_FILE, override=True)
    else:
        debug_print(f"No project .env found at: {PROJECT_ROOT_ENV_FILE}")

    if USER_ENV_EXISTS:
        debug_print(f"Loading user .env: {USER_ENV_FILE}")
        load_dotenv(USER_ENV_FILE, override=True)
