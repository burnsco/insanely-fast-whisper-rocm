"""Tests for Hugging Face cache helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from insanely_fast_whisper_rocm.utils import hf_cache


def test_ensure_local_hf_cache_env_sets_repo_local_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Configure a repo-local cache when no HF cache env vars are set."""
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setattr(hf_cache, "PROJECT_ROOT", tmp_path)

    cache_root = hf_cache.ensure_local_hf_cache_env()

    assert cache_root == tmp_path / ".hf-cache"
    assert (tmp_path / ".hf-cache" / "hub").is_dir()
    assert os.environ["HF_HOME"] == str(tmp_path / ".hf-cache")
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(tmp_path / ".hf-cache" / "hub")


def test_ensure_local_hf_cache_env_preserves_existing_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave existing HF cache configuration untouched."""
    monkeypatch.setenv("HF_HOME", "/custom/hf-home")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", "/custom/hf-cache")

    cache_root = hf_cache.ensure_local_hf_cache_env()

    assert cache_root is None
    assert os.environ["HF_HOME"] == "/custom/hf-home"
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == "/custom/hf-cache"
