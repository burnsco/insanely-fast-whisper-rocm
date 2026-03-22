"""Helpers for building WebUI download payloads."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from insanely_fast_whisper_rocm.core.formatters import FORMATTERS
from insanely_fast_whisper_rocm.utils.filename_generator import (
    FilenameGenerator,
    StandardFilenameStrategy,
    TaskType,
)

logger = logging.getLogger(__name__)

_filename_generator = FilenameGenerator(strategy=StandardFilenameStrategy())


def prepare_temp_downloadable_file(
    raw_data: dict[str, Any],
    format_type: str,
    original_audio_stem: str,
    temp_dir: Path,
    task: TaskType,
) -> str:
    """Generate and persist a temporary downloadable file for the WebUI.

    Args:
        raw_data: Raw transcription result data.
        format_type: Target output format such as ``txt`` or ``srt``.
        original_audio_stem: Stem of the original audio file.
        temp_dir: Directory where the temporary export should be written.
        task: Task used for filename generation.

    Returns:
        Absolute path to the generated temporary file.

    Raises:
        ValueError: If the requested output formatter does not exist.
        OSError: If writing the file fails.
    """
    formatter = FORMATTERS.get(format_type)
    if not formatter:
        raise ValueError(f"No formatter available for type: {format_type}")

    if (
        format_type == "srt"
        and isinstance(raw_data.get("srt_synced_text"), str)
        and raw_data["srt_synced_text"].strip()
    ):
        content = raw_data["srt_synced_text"]
    else:
        content = formatter.format(raw_data)
    filename = _filename_generator.create_filename(
        audio_path=original_audio_stem,
        task=task,
        extension=format_type,
    )
    temp_file_path = temp_dir / f"temp_dl_{filename}"
    temp_file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(temp_file_path, "w", encoding="utf-8") as export_file:
            export_file.write(content)
    except OSError as error:
        raise OSError(f"Failed to write temporary file: {temp_file_path}") from error

    logger.info("Created temporary download file: %s", temp_file_path)
    return str(temp_file_path)


def build_ui_json_summary(
    raw_result: dict[str, Any],
    *,
    json_file_path: str | None,
    max_text_preview_chars: int = 2_000,
) -> dict[str, Any]:
    """Build a compact JSON summary for Gradio display.

    Args:
        raw_result: Raw transcription result.
        json_file_path: Path to the saved JSON artifact, if available.
        max_text_preview_chars: Maximum preview length for transcript text.

    Returns:
        Compact summary suitable for rendering in the UI.
    """
    text = raw_result.get("text")
    chunks = raw_result.get("chunks")
    segments = raw_result.get("segments")

    return {
        "output_file_path": json_file_path,
        "text_len": len(text) if isinstance(text, str) else None,
        "text_preview": (
            text[:max_text_preview_chars] if isinstance(text, str) else None
        ),
        "chunks": len(chunks) if isinstance(chunks, list) else None,
        "segments": len(segments) if isinstance(segments, list) else None,
        "task_type": raw_result.get("task_type"),
        "runtime_seconds": raw_result.get("runtime_seconds"),
        "pipeline_runtime_seconds": raw_result.get("pipeline_runtime_seconds"),
        "processed_at": raw_result.get("processed_at"),
    }
