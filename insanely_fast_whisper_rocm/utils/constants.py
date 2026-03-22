"""Backward-compatible re-export of the centralized configuration module."""

from __future__ import annotations

from insanely_fast_whisper_rocm.utils import constant as _constant
from insanely_fast_whisper_rocm.utils.constant import *  # noqa: F403

__all__ = _constant.__all__
