"""Centralized configuration for the Insanely Fast Whisper ROCm app.

This module is the single source of truth for environment-backed settings.
It loads project and user ``.env`` files exactly once at import time, exposes
typed constants for the rest of the application, and owns the small set of
intentional runtime environment mutations needed for ROCm/PyTorch bootstrap.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Literal, cast

from insanely_fast_whisper_rocm.utils.env_loader import (
    PROJECT_ROOT,
    USER_CONFIG_DIR,
    USER_ENV_EXISTS,
    USER_ENV_FILE,
    debug_print,
    load_project_env,
)

logger = logging.getLogger(__name__)


def _env_text(name: str, default: str | None = None) -> str | None:
    """Return a string environment value.

    Args:
        name: Environment variable name.
        default: Fallback value when the variable is unset.

    Returns:
        The environment value or the provided default.
    """
    return os.getenv(name, default)


def _env_flag(name: str, default: bool = False) -> bool:
    """Return a boolean environment value.

    Args:
        name: Environment variable name.
        default: Fallback value when the variable is unset.

    Returns:
        Parsed boolean value.
    """
    fallback = "true" if default else "false"
    return os.getenv(name, fallback).lower() == "true"


def _env_int(name: str, default: int) -> int:
    """Return an integer environment value.

    Args:
        name: Environment variable name.
        default: Fallback integer.

    Returns:
        Parsed integer value.
    """
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    """Return a float environment value.

    Args:
        name: Environment variable name.
        default: Fallback float.

    Returns:
        Parsed floating-point value.
    """
    return float(os.getenv(name, str(default)))


def _env_csv(name: str, default: str) -> list[str]:
    """Return a comma-delimited environment value as a list.

    Args:
        name: Environment variable name.
        default: Fallback CSV string.

    Returns:
        List of non-empty stripped values.
    """
    raw_value = os.getenv(name, default)
    return [value.strip() for value in raw_value.split(",") if value.strip()]


def _configure_pytorch_allocator(default_value: str) -> str:
    """Set the correct PyTorch allocator environment variables for ROCm.

    Args:
        default_value: Allocation config string to apply.

    Returns:
        The effective allocator configuration.
    """
    allocator_value = os.getenv("PYTORCH_ALLOC_CONF") or os.getenv(
        "PYTORCH_HIP_ALLOC_CONF"
    )
    effective_value = allocator_value or default_value

    try:
        torch_spec = importlib.util.find_spec("torch")
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
        torch_spec = None

    use_new_name = True
    if torch_spec is not None:
        try:
            torch_version = pkg_version("torch")
        except PackageNotFoundError:
            torch_version = None

        if torch_version is not None:
            major, minor, *_ = map(int, torch_version.split(".")[:2])
            use_new_name = (major, minor) >= (2, 9)

    if use_new_name:
        os.environ["PYTORCH_ALLOC_CONF"] = effective_value
    else:
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = effective_value

    # Set both names defensively so subprocesses and older helpers stay aligned.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", effective_value)
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", effective_value)
    return effective_value


def set_huggingface_cache_paths(cache_root: Path) -> None:
    """Apply explicit Hugging Face cache paths to process config and env vars.

    Args:
        cache_root: Base cache directory to use for Hugging Face assets.
    """
    global HF_HOME
    global HUGGINGFACE_HUB_CACHE

    hub_root = cache_root / "hub"
    hub_root.mkdir(parents=True, exist_ok=True)

    HF_HOME = str(cache_root)
    HUGGINGFACE_HUB_CACHE = str(hub_root)
    os.environ["HF_HOME"] = HF_HOME
    os.environ["HUGGINGFACE_HUB_CACHE"] = HUGGINGFACE_HUB_CACHE


load_project_env()

IS_TEST_ENV = ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules)

# Project paths and config aliases
CONFIG_DIR = USER_CONFIG_DIR
ENV_FILE = USER_ENV_FILE

# Model configuration
DEFAULT_MODEL = _env_text("WHISPER_MODEL", "distil-whisper/distil-large-v3.5") or ""
DEFAULT_DEVICE = _env_text("WHISPER_DEVICE", "0") or "0"
DEFAULT_BATCH_SIZE = _env_int("WHISPER_BATCH_SIZE", 4)
DEFAULT_DTYPE = _env_text("WHISPER_DTYPE", "float16") or "float16"
DEFAULT_BETTER_TRANSFORMER = _env_flag("WHISPER_BETTER_TRANSFORMER", default=False)
DEFAULT_CONDITION_ON_PREV_TOKENS = _env_flag(
    "WHISPER_CONDITION_ON_PREV_TOKENS",
    default=True,
)
DEFAULT_SEQUENTIAL_LONG_FORM = _env_flag(
    "WHISPER_SEQUENTIAL_LONG_FORM",
    default=False,
)
DEFAULT_CHUNK_LENGTH = _env_int("WHISPER_CHUNK_LENGTH", 30)
DEFAULT_PROGRESS_GROUP_SIZE = _env_int("PROGRESS_GROUP_SIZE", 4)

_timestamp_type = _env_text("WHISPER_TIMESTAMP_TYPE", "word") or "word"
if _timestamp_type not in {"chunk", "word"}:
    _timestamp_type = "word"
DEFAULT_TIMESTAMP_TYPE: Literal["chunk", "word"] = cast(
    Literal["chunk", "word"], _timestamp_type
)

DEFAULT_LANGUAGE = _env_text("WHISPER_LANGUAGE", "None") or "None"
DEFAULT_DIARIZATION_MODEL = (
    _env_text("WHISPER_DIARIZATION_MODEL", "pyannote/speaker-diarization") or ""
)

# Runtime and file handling
UPLOAD_DIR = _env_text("WHISPER_UPLOAD_DIR", "data/temp_uploads") or "data/temp_uploads"
DEFAULT_TRANSCRIPTS_DIR = (
    _env_text("WHISPER_TRANSCRIPTS_DIR", "data/transcripts") or "data/transcripts"
)
SAVE_TRANSCRIPTIONS = _env_flag("SAVE_TRANSCRIPTIONS", default=True)
APP_TIMEZONE = (
    _env_text(
        "APP_TIMEZONE",
        _env_text("FILENAME_TIMEZONE", _env_text("TZ", "UTC")),
    )
    or "UTC"
)
FILENAME_TIMEZONE = APP_TIMEZONE

HF_TOKEN = _env_text("HF_TOKEN")
HF_HOME = _env_text("HF_HOME")
HUGGINGFACE_HUB_CACHE = _env_text("HUGGINGFACE_HUB_CACHE")

# Processing limits
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 32
COMMAND_TIMEOUT_SECONDS = 3600
MAX_AUDIO_SIZE_MB = 100
MAX_CONCURRENT_REQUESTS = 10
TEMP_FILE_TTL_SECONDS = 3600
MIN_SPEAKERS = 1
MAX_SPEAKERS = 10

# Audio chunking
AUDIO_CHUNK_DURATION = _env_float("AUDIO_CHUNK_DURATION", 600.0)
AUDIO_CHUNK_OVERLAP = _env_float("AUDIO_CHUNK_OVERLAP", 1.0)
AUDIO_CHUNK_MIN_DURATION = _env_float("AUDIO_CHUNK_MIN_DURATION", 5.0)

# Subtitle/readability defaults
USE_READABLE_SUBTITLES = _env_flag("USE_READABLE_SUBTITLES", default=True)
MAX_LINE_CHARS = _env_int("MAX_LINE_CHARS", 42)
MAX_LINES_PER_BLOCK = _env_int("MAX_LINES_PER_BLOCK", 2)
MAX_BLOCK_CHARS = _env_int("MAX_BLOCK_CHARS", MAX_LINE_CHARS * MAX_LINES_PER_BLOCK)
MAX_BLOCK_CHARS_SOFT = _env_int("MAX_BLOCK_CHARS_SOFT", 90)
MIN_CPS = _env_float("MIN_CPS", 12.0)
MAX_CPS = _env_float("MAX_CPS", 17.0)
MIN_SEGMENT_DURATION_SEC = _env_float("MIN_SEGMENT_DURATION_SEC", 1.2)
MAX_SEGMENT_DURATION_SEC = _env_float("MAX_SEGMENT_DURATION_SEC", 5.5)
MIN_WORD_DURATION_SEC = _env_float("MIN_WORD_DURATION_SEC", 0.04)
DISPLAY_BUFFER_SEC = _env_float("DISPLAY_BUFFER_SEC", 0.2)
SOFT_BOUNDARY_WORDS = _env_csv("SOFT_BOUNDARY_WORDS", "and,but,or,so,for,nor,yet")
INTERJECTION_WHITELIST = _env_csv("INTERJECTION_WHITELIST", "um,uh,ah,er,like")

# Timestamp stabilization
DEFAULT_STABILIZE = _env_flag("STABILIZE_DEFAULT", default=True)
DEFAULT_DEMUCS = _env_flag("DEMUCS_DEFAULT", default=False)
DEFAULT_VAD = _env_flag("VAD_DEFAULT", default=False)
DEFAULT_VAD_THRESHOLD = _env_float("VAD_THRESHOLD_DEFAULT", 0.35)
DEFAULT_SUPPRESS_TS_TOKENS = _env_flag("SUPPRESS_TS_TOKENS_DEFAULT", default=True)
DEFAULT_GAP_PADDING = _env_text("GAP_PADDING_DEFAULT", " ...") or " ..."
DEFAULT_ADJUST_GAPS = _env_flag("ADJUST_GAPS_DEFAULT", default=True)
_nonspeech_skip_raw = (_env_text("NONSPEECH_SKIP_DEFAULT", "") or "").strip()
DEFAULT_NONSPEECH_SKIP = float(_nonspeech_skip_raw) if _nonspeech_skip_raw else None

# Subtitle synchronization
DEFAULT_SUBTITLE_SYNC = _env_flag("SUBTITLE_SYNC_DEFAULT", default=True)
ALASS_BINARY = _env_text("ALASS_BINARY", "alass") or "alass"
ALASS_TIMEOUT_SECONDS = _env_int("ALASS_TIMEOUT_SECONDS", 180)
ALASS_SPLIT_PENALTY = _env_int("ALASS_SPLIT_PENALTY", 7)
ALASS_NO_SPLITS = _env_flag("ALASS_NO_SPLITS", default=False)

# WebUI defaults tuned for subtitle generation workflows.
_webui_timestamp_type = _env_text("WEBUI_TIMESTAMP_TYPE", "word") or "word"
if _webui_timestamp_type not in {"chunk", "word"}:
    _webui_timestamp_type = "word"
WEBUI_DEFAULT_TIMESTAMP_TYPE: Literal["chunk", "word"] = cast(
    Literal["chunk", "word"], _webui_timestamp_type
)
_webui_task = _env_text("WEBUI_TASK", "transcribe") or "transcribe"
if _webui_task not in {"transcribe", "translate"}:
    _webui_task = "transcribe"
WEBUI_DEFAULT_TASK: Literal["transcribe", "translate"] = cast(
    Literal["transcribe", "translate"],
    _webui_task,
)
WEBUI_DEFAULT_STABILIZE = _env_flag("WEBUI_STABILIZE_DEFAULT", default=True)
WEBUI_DEFAULT_DEMUCS = _env_flag("WEBUI_DEMUCS_DEFAULT", default=False)
WEBUI_DEFAULT_VAD = _env_flag("WEBUI_VAD_DEFAULT", default=True)
WEBUI_DEFAULT_VAD_THRESHOLD = _env_float("WEBUI_VAD_THRESHOLD_DEFAULT", 0.35)
WEBUI_DEFAULT_SUBTITLE_SYNC = _env_flag("WEBUI_SUBTITLE_SYNC_DEFAULT", default=True)
WEBUI_DEFAULT_SUPPRESS_TS_TOKENS = _env_flag(
    "WEBUI_SUPPRESS_TS_TOKENS_DEFAULT",
    default=True,
)
WEBUI_DEFAULT_GAP_PADDING = (
    _env_text("WEBUI_GAP_PADDING_DEFAULT", DEFAULT_GAP_PADDING) or DEFAULT_GAP_PADDING
)
WEBUI_DEFAULT_ADJUST_GAPS = _env_flag("WEBUI_ADJUST_GAPS_DEFAULT", default=True)
_webui_nonspeech_skip_raw = (
    _env_text("WEBUI_NONSPEECH_SKIP_DEFAULT", "") or ""
).strip()
WEBUI_DEFAULT_NONSPEECH_SKIP = (
    float(_webui_nonspeech_skip_raw) if _webui_nonspeech_skip_raw else None
)

# ROCm and runtime bootstrap
ROCM_PATH = _env_text("ROCM_PATH")
HSA_OVERRIDE_GFX_VERSION = _env_text("HSA_OVERRIDE_GFX_VERSION")
HIP_LAUNCH_BLOCKING = _env_flag("HIP_LAUNCH_BLOCKING", default=False)
TORCHAUDIO_USE_SOUNDFILE = _env_text("TORCHAUDIO_USE_SOUNDFILE")
PYTORCH_HIP_ALLOC_CONF = _configure_pytorch_allocator(
    "garbage_collection_threshold:0.7,max_split_size_mb:128"
)
PYTORCH_ALLOC_CONF = os.environ.get("PYTORCH_ALLOC_CONF", PYTORCH_HIP_ALLOC_CONF)

# Misc runtime toggles
IFW_EAGER_MODEL_RELEASE = _env_flag("IFW_EAGER_MODEL_RELEASE", default=False)
SKIP_FS_CHECKS = _env_flag("IFW_SKIP_FS_CHECKS", default=False) or IS_TEST_ENV

# API and UI configuration
API_TITLE = "Insanely Fast Whisper API"
API_DESCRIPTION = "ROCm-first Whisper transcription and translation service."
API_HOST = _env_text("API_HOST", "0.0.0.0") or "0.0.0.0"
API_PORT = _env_int("API_PORT", 8888)
DEV_API_PORT = _env_int("DEV_API_PORT", 8889)
WEBUI_HOST = _env_text("WEBUI_HOST", "0.0.0.0") or "0.0.0.0"
WEBUI_PORT = _env_int("WEBUI_PORT", 7860)
DEV_WEBUI_PORT = _env_int("DEV_WEBUI_PORT", 7862)
DEFAULT_RESPONSE_FORMAT = "json"
LOG_LEVEL = _env_text("LOG_LEVEL", "INFO") or "INFO"

try:
    API_VERSION = pkg_version("insanely-fast-whisper-rocm")
except PackageNotFoundError:
    API_VERSION = "2.1.5"

# Response formats
RESPONSE_FORMAT_JSON = "json"
RESPONSE_FORMAT_TEXT = "text"
RESPONSE_FORMAT_VERBOSE_JSON = "verbose_json"
RESPONSE_FORMAT_SRT = "srt"
RESPONSE_FORMAT_VTT = "vtt"
SUPPORTED_RESPONSE_FORMATS = {
    RESPONSE_FORMAT_JSON,
    RESPONSE_FORMAT_TEXT,
    RESPONSE_FORMAT_VERBOSE_JSON,
    RESPONSE_FORMAT_SRT,
    RESPONSE_FORMAT_VTT,
}

# Supported upload formats
SUPPORTED_AUDIO_FORMATS: set[str] = {
    ".mp3",
    ".flac",
    ".wav",
    ".m4a",
}
SUPPORTED_VIDEO_FORMATS: set[str] = {
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
}
SUPPORTED_UPLOAD_FORMATS: set[str] = SUPPORTED_AUDIO_FORMATS | SUPPORTED_VIDEO_FORMATS

if USER_ENV_EXISTS:
    debug_print(f"Loaded user .env from {USER_ENV_FILE}")
debug_print(
    "Config initialized "
    f"model={DEFAULT_MODEL} device={DEFAULT_DEVICE} batch_size={DEFAULT_BATCH_SIZE} "
    f"log_level={LOG_LEVEL}"
)

__all__ = [
    "API_DESCRIPTION",
    "API_HOST",
    "API_PORT",
    "API_TITLE",
    "API_VERSION",
    "APP_TIMEZONE",
    "AUDIO_CHUNK_DURATION",
    "AUDIO_CHUNK_MIN_DURATION",
    "AUDIO_CHUNK_OVERLAP",
    "COMMAND_TIMEOUT_SECONDS",
    "CONFIG_DIR",
    "DEFAULT_ADJUST_GAPS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_BETTER_TRANSFORMER",
    "DEFAULT_CHUNK_LENGTH",
    "DEFAULT_CONDITION_ON_PREV_TOKENS",
    "DEFAULT_DEMUCS",
    "DEFAULT_DEVICE",
    "DEFAULT_DIARIZATION_MODEL",
    "DEFAULT_DTYPE",
    "DEFAULT_GAP_PADDING",
    "DEFAULT_LANGUAGE",
    "DEFAULT_MODEL",
    "DEFAULT_NONSPEECH_SKIP",
    "DEFAULT_PROGRESS_GROUP_SIZE",
    "DEFAULT_RESPONSE_FORMAT",
    "DEFAULT_SEQUENTIAL_LONG_FORM",
    "DEFAULT_STABILIZE",
    "DEFAULT_SUBTITLE_SYNC",
    "DEFAULT_SUPPRESS_TS_TOKENS",
    "DEFAULT_TIMESTAMP_TYPE",
    "DEFAULT_TRANSCRIPTS_DIR",
    "DEFAULT_VAD",
    "DEFAULT_VAD_THRESHOLD",
    "DEV_API_PORT",
    "DEV_WEBUI_PORT",
    "DISPLAY_BUFFER_SEC",
    "ENV_FILE",
    "FILENAME_TIMEZONE",
    "HF_HOME",
    "HF_TOKEN",
    "HIP_LAUNCH_BLOCKING",
    "HSA_OVERRIDE_GFX_VERSION",
    "HUGGINGFACE_HUB_CACHE",
    "IFW_EAGER_MODEL_RELEASE",
    "INTERJECTION_WHITELIST",
    "IS_TEST_ENV",
    "LOG_LEVEL",
    "MAX_AUDIO_SIZE_MB",
    "MAX_BATCH_SIZE",
    "MAX_BLOCK_CHARS",
    "MAX_BLOCK_CHARS_SOFT",
    "MAX_CONCURRENT_REQUESTS",
    "MAX_CPS",
    "MAX_LINE_CHARS",
    "MAX_LINES_PER_BLOCK",
    "MAX_SEGMENT_DURATION_SEC",
    "MAX_SPEAKERS",
    "MIN_BATCH_SIZE",
    "MIN_CPS",
    "MIN_SEGMENT_DURATION_SEC",
    "MIN_SPEAKERS",
    "MIN_WORD_DURATION_SEC",
    "PROJECT_ROOT",
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_HIP_ALLOC_CONF",
    "RESPONSE_FORMAT_JSON",
    "RESPONSE_FORMAT_SRT",
    "RESPONSE_FORMAT_TEXT",
    "RESPONSE_FORMAT_VERBOSE_JSON",
    "RESPONSE_FORMAT_VTT",
    "ROCM_PATH",
    "SAVE_TRANSCRIPTIONS",
    "SKIP_FS_CHECKS",
    "SOFT_BOUNDARY_WORDS",
    "SUPPORTED_AUDIO_FORMATS",
    "SUPPORTED_RESPONSE_FORMATS",
    "SUPPORTED_UPLOAD_FORMATS",
    "SUPPORTED_VIDEO_FORMATS",
    "TEMP_FILE_TTL_SECONDS",
    "TORCHAUDIO_USE_SOUNDFILE",
    "UPLOAD_DIR",
    "USE_READABLE_SUBTITLES",
    "USER_CONFIG_DIR",
    "USER_ENV_FILE",
    "WEBUI_HOST",
    "WEBUI_PORT",
    "ALASS_BINARY",
    "ALASS_TIMEOUT_SECONDS",
    "ALASS_SPLIT_PENALTY",
    "ALASS_NO_SPLITS",
    "set_huggingface_cache_paths",
]
