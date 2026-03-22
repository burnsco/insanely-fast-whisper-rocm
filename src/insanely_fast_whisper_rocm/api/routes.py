"""API route definitions for the Insanely Fast Whisper API.

This module contains clean, focused route definitions that use dependency
injection for ASR pipeline instances and file handling.
"""

import logging
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from insanely_fast_whisper_rocm.api.dependencies import (
    get_backend_config,
    get_file_handler,
)
from insanely_fast_whisper_rocm.api.responses import ResponseFormatter
from insanely_fast_whisper_rocm.audio.processing import extract_audio_from_video
from insanely_fast_whisper_rocm.core.asr_backend import HuggingFaceBackendConfig
from insanely_fast_whisper_rocm.core.errors import OutOfMemoryError
from insanely_fast_whisper_rocm.core.formatters import FORMATTERS
from insanely_fast_whisper_rocm.core.integrations.alass import sync_subtitle_with_alass
from insanely_fast_whisper_rocm.core.integrations.stable_ts import stabilize_timestamps
from insanely_fast_whisper_rocm.core.orchestrator import create_orchestrator
from insanely_fast_whisper_rocm.utils import (
    DEFAULT_DEMUCS,
    DEFAULT_STABILIZE,
    DEFAULT_SUBTITLE_SYNC,
    DEFAULT_TIMESTAMP_TYPE,
    DEFAULT_VAD,
    DEFAULT_VAD_THRESHOLD,
    RESPONSE_FORMAT_JSON,
    SUPPORTED_RESPONSE_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    FileHandler,
)

logger = logging.getLogger(__name__)

router = APIRouter()

TimestampType = Literal["chunk", "word"]
FormatterName = Literal["transcription", "translation"]


def _apply_subtitle_sync(
    *,
    result: dict[str, Any],
    reference_media_path: str,
    subtitle_sync: bool,
    timestamp_type: TimestampType | None = None,
) -> dict[str, Any]:
    """Apply ALASS subtitle synchronization metadata to an ASR result.

    Args:
        result: ASR result dictionary.
        reference_media_path: Original uploaded media path.
        subtitle_sync: Whether synchronization is enabled.
        timestamp_type: Optional timestamp mode hint forwarded to the SRT
            formatter.

    Returns:
        The result dictionary with ``subtitle_sync`` metadata and optional
        ``srt_synced_text``.
    """
    metadata = {
        "enabled": bool(subtitle_sync),
        "engine": "alass",
        "applied": False,
        "reason": "disabled",
        "runtime_ms": None,
    }
    if not subtitle_sync:
        result["subtitle_sync"] = metadata
        return result

    media_path = Path(reference_media_path)
    if media_path.suffix.lower() not in SUPPORTED_VIDEO_FORMATS:
        metadata["reason"] = "non_video_input"
        result["subtitle_sync"] = metadata
        return result

    srt_text = FORMATTERS["srt"].format(result, timestamp_type=timestamp_type)
    synced_srt, metadata = sync_subtitle_with_alass(
        reference_media_path=media_path,
        subtitle_content=srt_text,
    )
    if metadata.get("applied"):
        result["srt_synced_text"] = synced_srt
    result["subtitle_sync"] = metadata
    return result


def _parse_timestamp_type(timestamp_type: str) -> TimestampType:
    """Validate and narrow the requested timestamp type.

    Args:
        timestamp_type: Raw timestamp type from the request payload.

    Returns:
        The validated timestamp type literal.

    Raises:
        HTTPException: If the value is unsupported.
    """
    if timestamp_type not in ("chunk", "word"):
        raise HTTPException(status_code=400, detail="Unsupported timestamp_type")
    if timestamp_type == "chunk":
        return "chunk"
    return "word"


def _format_api_response(
    result: dict[str, Any],
    response_format: str,
    formatter_name: FormatterName,
) -> Response:
    """Format an API result using the correct response formatter branch.

    Args:
        result: ASR result payload to format.
        response_format: Requested output format.
        formatter_name: Selects transcription vs translation formatting.

    Returns:
        Response: Formatted FastAPI response.

    Raises:
        HTTPException: If the response format is unsupported.
    """
    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise HTTPException(status_code=400, detail="Unsupported response_format")
    if formatter_name == "transcription":
        return ResponseFormatter.format_transcription(result, response_format)
    return ResponseFormatter.format_translation(result, response_format)


def _process_audio_request(
    *,
    file: UploadFile,
    response_format: str,
    timestamp_type: str,
    language: str | None,
    task: Literal["transcribe", "translate"],
    stabilize: bool,
    demucs: bool,
    vad: bool,
    vad_threshold: float,
    subtitle_sync: bool,
    backend_config: HuggingFaceBackendConfig,
    file_handler: FileHandler,
    formatter_name: FormatterName,
) -> Response:
    """Process a transcription or translation request end-to-end.

    Args:
        file: Uploaded audio or video input.
        response_format: Requested output format.
        timestamp_type: Requested timestamp mode.
        language: Optional input language.
        task: Whisper task to execute.
        stabilize: Whether to run timestamp stabilization.
        demucs: Whether to enable Demucs during stabilization.
        vad: Whether to enable VAD during stabilization.
        vad_threshold: VAD threshold for stabilization.
        subtitle_sync: Whether to run ALASS subtitle synchronization.
        backend_config: Backend settings for orchestrator execution.
        file_handler: Upload and cleanup helper.
        formatter_name: Selects transcription vs translation response formatting.

    Returns:
        Response: Final formatted response for the request.

    Raises:
        HTTPException: If validation or processing fails.
    """
    file_handler.validate_audio_file(file)
    temp_filepath = file_handler.save_upload(file)
    temp_files_to_cleanup = [temp_filepath]
    processed_audio_path = temp_filepath

    if Path(temp_filepath).suffix.lower() in SUPPORTED_VIDEO_FORMATS:
        try:
            processed_audio_path = extract_audio_from_video(temp_filepath)
            temp_files_to_cleanup.append(processed_audio_path)
        except RuntimeError as conv_error:
            raise HTTPException(status_code=500, detail=str(conv_error)) from conv_error

    try:
        parsed_timestamp_type = _parse_timestamp_type(timestamp_type)
        orchestrator = create_orchestrator()

        try:
            result = orchestrator.run_transcription(
                audio_path=processed_audio_path,
                backend_config=backend_config,
                language=language,
                task=task,
                timestamp_type=parsed_timestamp_type,
            )
        except OutOfMemoryError as oom:
            raise HTTPException(
                status_code=507,
                detail=f"Insufficient GPU memory for {task}: {str(oom)}",
            ) from oom
        except Exception as exc:
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if stabilize:
            try:
                result = stabilize_timestamps(
                    result,
                    demucs=demucs,
                    vad=vad,
                    vad_threshold=vad_threshold,
                )
            except Exception as stab_exc:  # noqa: BLE001
                logger.error("Stabilization failed: %s", stab_exc, exc_info=True)

        result = _apply_subtitle_sync(
            result=result,
            reference_media_path=temp_filepath,
            subtitle_sync=subtitle_sync,
            timestamp_type=parsed_timestamp_type,
        )

        return _format_api_response(result, response_format, formatter_name)
    finally:
        for path in reversed(temp_files_to_cleanup):
            file_handler.cleanup(path)


@router.post(
    "/v1/audio/transcriptions",
    tags=["Transcription"],
    summary="Transcribe Audio",
    description="Convert speech in an audio file to text using the Whisper model",
    responses={
        200: {
            "description": "Successful transcription",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/TranscriptionResponse"}
                },
                "text/plain": {"schema": {"type": "string"}},
            },
        },
        400: {"description": "Invalid request parameters"},
        422: {"description": "Validation error (e.g., unsupported file format)"},
        500: {"description": "Internal server error"},
        503: {"description": "Model not loaded or unavailable"},
    },
)
async def create_transcription(
    file: UploadFile = File(..., description="The audio/video file to transcribe"),  # noqa: B008
    response_format: str = Form(
        RESPONSE_FORMAT_JSON,
        description="Response format (json, verbose_json, text, srt, vtt)",
    ),
    timestamp_type: str = Form(
        DEFAULT_TIMESTAMP_TYPE,
        description="Type of timestamp to generate ('chunk' or 'word')",
    ),
    language: str | None = Form(
        None, description="Source language code (auto-detect if None)"
    ),
    task: Literal["transcribe"] = Form("transcribe", description="ASR task type"),
    stabilize: bool = Form(
        DEFAULT_STABILIZE, description="Enable timestamp stabilization"
    ),
    demucs: bool = Form(DEFAULT_DEMUCS, description="Enable Demucs noise reduction"),
    vad: bool = Form(DEFAULT_VAD, description="Enable Voice Activity Detection"),
    vad_threshold: float = Form(
        DEFAULT_VAD_THRESHOLD, description="VAD threshold for speech detection"
    ),
    subtitle_sync: bool = Form(
        DEFAULT_SUBTITLE_SYNC,
        description="Enable ALASS subtitle synchronization for generated SRT output",
    ),
    backend_config: HuggingFaceBackendConfig = Depends(get_backend_config),  # noqa: B008
    file_handler: FileHandler = Depends(get_file_handler),  # noqa: B008
) -> Response:
    """Transcribe speech in an audio file to text.

    This endpoint processes an audio file and returns its transcription using the
    specified Whisper model. It supports various configuration options including
    timestamp generation.

    Args:
        file: The audio/video file to transcribe.
        response_format: Desired response format ("json", "verbose_json",
            "text", "srt", or "vtt").
        timestamp_type: Type of timestamp to generate ("chunk" or "word")
        language: Optional source language code (auto-detect if None)
        task: ASR task type (must be "transcribe")
        stabilize: Enable timestamp stabilization if True.
        demucs: Enable Demucs noise reduction if True.
        vad: Enable Voice Activity Detection if True.
        vad_threshold: VAD sensitivity threshold (0.0 - 1.0).
        subtitle_sync: Enable ALASS subtitle synchronization for SRT output.
        backend_config: Injected backend configuration
        file_handler: Injected file handler instance

    Returns:
        Union[str, dict]: Transcription result as plain text or JSON with metadata
    """
    logger.info("-" * 50)
    logger.info("Received transcription request:")
    logger.info("  File: %s", file.filename)
    logger.debug("  Timestamp type: %s", timestamp_type)
    logger.debug("  Language: %s", language)
    logger.debug("  Task: %s", task)

    logger.info("Starting transcription process...")
    response = _process_audio_request(
        file=file,
        response_format=response_format,
        timestamp_type=timestamp_type,
        language=language,
        task=task,
        stabilize=stabilize,
        demucs=demucs,
        vad=vad,
        vad_threshold=vad_threshold,
        subtitle_sync=subtitle_sync,
        backend_config=backend_config,
        file_handler=file_handler,
        formatter_name="transcription",
    )
    logger.info("Transcription completed successfully")
    return response


@router.post(
    "/v1/audio/translations",
    tags=["Translation"],
    summary="Translate Audio",
    description="Translate speech in an audio file to English using the Whisper model",
    responses={
        200: {
            "description": "Successful translation",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/TranscriptionResponse"}
                },
                "text/plain": {"schema": {"type": "string"}},
            },
        },
        400: {"description": "Invalid request parameters"},
        422: {"description": "Validation error (e.g., unsupported file format)"},
        500: {"description": "Internal server error"},
        503: {"description": "Model not loaded or unavailable"},
    },
)
async def create_translation(
    file: UploadFile = File(..., description="The audio/video file to translate"),  # noqa: B008
    response_format: str = Form(
        RESPONSE_FORMAT_JSON,
        description="Response format (json, verbose_json, text, srt, vtt)",
    ),
    timestamp_type: str = Form(
        DEFAULT_TIMESTAMP_TYPE,
        description="Type of timestamp to generate ('chunk' or 'word')",
    ),
    language: str | None = Form(
        None, description="Source language code (auto-detect if None)"
    ),
    stabilize: bool = Form(
        DEFAULT_STABILIZE, description="Enable timestamp stabilization"
    ),
    demucs: bool = Form(DEFAULT_DEMUCS, description="Enable Demucs noise reduction"),
    vad: bool = Form(DEFAULT_VAD, description="Enable Voice Activity Detection"),
    vad_threshold: float = Form(
        DEFAULT_VAD_THRESHOLD, description="VAD threshold for speech detection"
    ),
    subtitle_sync: bool = Form(
        DEFAULT_SUBTITLE_SYNC,
        description="Enable ALASS subtitle synchronization for generated SRT output",
    ),
    backend_config: HuggingFaceBackendConfig = Depends(get_backend_config),  # noqa: B008
    file_handler: FileHandler = Depends(get_file_handler),  # noqa: B008
) -> Response:
    """Translate speech in an audio file to English.

    This endpoint processes an audio file in any supported language and translates
    the speech to English using the specified Whisper model. It supports various
    configuration options.

    Args:
        file: The audio/video file to translate.
        response_format: Desired response format ("json" or "text")
        timestamp_type: Type of timestamp to generate ("chunk" or "word")
        language: Optional source language code (auto-detect if None)
        stabilize: Enable timestamp stabilization if True.
        demucs: Enable Demucs noise reduction if True.
        vad: Enable Voice Activity Detection if True.
        vad_threshold: VAD sensitivity threshold (0.0 - 1.0).
        subtitle_sync: Enable ALASS subtitle synchronization for SRT output.
        backend_config: Injected backend configuration
        file_handler: Injected file handler instance

    Returns:
        Union[str, dict]: Translation result as plain text or JSON with metadata
    """
    logger.info("-" * 50)
    logger.info("Received translation request:")
    logger.info("  File: %s", file.filename)
    logger.debug("  Timestamp type: %s", timestamp_type)
    logger.debug("  Language: %s", language)
    logger.debug("  Response format: %s", response_format)

    logger.info("Starting translation process...")
    response = _process_audio_request(
        file=file,
        response_format=response_format,
        timestamp_type=timestamp_type,
        language=language,
        task="translate",
        stabilize=stabilize,
        demucs=demucs,
        vad=vad,
        vad_threshold=vad_threshold,
        subtitle_sync=subtitle_sync,
        backend_config=backend_config,
        file_handler=file_handler,
        formatter_name="translation",
    )
    logger.info("Translation completed successfully")
    return response
