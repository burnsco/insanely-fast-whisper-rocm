"""Helpers for selecting a writable Hugging Face cache directory."""

from __future__ import annotations

from pathlib import Path

from insanely_fast_whisper_rocm.utils import constant

PROJECT_ROOT = constant.PROJECT_ROOT


def ensure_local_hf_cache_env() -> Path | None:
    """Ensure a writable local Hugging Face cache is configured.

    Returns:
        The configured local cache directory when this function sets one, else
        ``None`` if existing environment configuration is preserved.
    """
    if constant.HF_HOME or constant.HUGGINGFACE_HUB_CACHE:
        return None

    cache_root = PROJECT_ROOT / ".hf-cache"
    constant.set_huggingface_cache_paths(cache_root)
    return cache_root
