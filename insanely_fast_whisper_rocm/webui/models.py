"""Typed models used by the optional Gradio WebUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from insanely_fast_whisper_rocm.utils import constant as constants


@dataclass
class TranscriptionConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for a WebUI transcription request."""

    model: str = constants.DEFAULT_MODEL
    device: str = constants.DEFAULT_DEVICE
    batch_size: int = constants.DEFAULT_BATCH_SIZE
    timestamp_type: Literal["chunk", "word"] = constants.DEFAULT_TIMESTAMP_TYPE
    language: str = constants.DEFAULT_LANGUAGE
    task: Literal["transcribe", "translate"] = "transcribe"
    dtype: str = "float16"
    chunk_length: int = 30
    chunk_duration: float | None = None
    chunk_overlap: float | None = None
    stabilize: bool = constants.DEFAULT_STABILIZE
    demucs: bool = constants.DEFAULT_DEMUCS
    vad: bool = constants.DEFAULT_VAD
    vad_threshold: float = constants.DEFAULT_VAD_THRESHOLD
    suppress_ts_tokens: bool = constants.DEFAULT_SUPPRESS_TS_TOKENS
    gap_padding: str = constants.DEFAULT_GAP_PADDING
    adjust_gaps: bool = constants.DEFAULT_ADJUST_GAPS
    nonspeech_skip: float | None = constants.DEFAULT_NONSPEECH_SKIP


@dataclass
class FileHandlingConfig:
    """Configuration for WebUI output and temporary files."""

    save_transcriptions: bool = True
    temp_uploads_dir: str = constants.DEFAULT_TRANSCRIPTS_DIR
