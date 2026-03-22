"""Helpers for selecting a writable Hugging Face cache directory."""

from __future__ import annotations

import os
from pathlib import Path

from insanely_fast_whisper_rocm.utils.constants import PROJECT_ROOT


def ensure_local_hf_cache_env() -> Path | None:
    """Ensure a writable local Hugging Face cache is configured.

    Returns:
        The configured local cache directory when this function sets one, else
        ``None`` if existing environment configuration is preserved.
    """
    if os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE"):
        return None

    cache_root = PROJECT_ROOT / ".hf-cache"
    hub_root = cache_root / "hub"
    hub_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_root)
    return cache_root
