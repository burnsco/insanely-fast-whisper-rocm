"""Tests for Hugging Face cache directory resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from insanely_fast_whisper_rocm.utils import constant
from insanely_fast_whisper_rocm.utils.download_hf_model import (
    _resolve_effective_cache_dir,
)


def test_resolve_effective_cache_dir_prefers_explicit_path() -> None:
    """Prefer an explicit cache directory over environment settings."""
    explicit = Path("/tmp/explicit-cache")

    assert _resolve_effective_cache_dir(explicit) == explicit


def test_resolve_effective_cache_dir_uses_hub_cache_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the explicit Hub cache env var when present."""
    monkeypatch.setattr(constant, "HF_HOME", None)
    monkeypatch.setattr(constant, "HUGGINGFACE_HUB_CACHE", "/tmp/hub-cache")

    assert _resolve_effective_cache_dir(None) == "/tmp/hub-cache"


def test_resolve_effective_cache_dir_uses_hf_home_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Derive the cache directory from ``HF_HOME`` when needed."""
    monkeypatch.setattr(constant, "HUGGINGFACE_HUB_CACHE", None)
    monkeypatch.setattr(constant, "HF_HOME", "/tmp/hf-home")

    assert _resolve_effective_cache_dir(None) == Path("/tmp/hf-home") / "hub"
