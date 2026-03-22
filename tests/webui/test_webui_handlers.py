"""Unit tests for the WebUI handler functions."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from insanely_fast_whisper_rocm.webui.handlers import TranscriptionConfig, transcribe


@pytest.fixture
def mock_pipeline_and_stabilizer() -> Generator[
    tuple[MagicMock, MagicMock], None, None
]:
    """Fixture to mock orchestrator creation and ``stabilize_timestamps``.

    Yields:
        tuple[MagicMock, MagicMock]: A tuple of (mocked pipeline instance,
        mocked ``stabilize_timestamps`` function).
    """
    with (
        patch(
            "insanely_fast_whisper_rocm.webui.handlers.create_orchestrator"
        ) as mock_create_orchestrator,
        patch(
            "insanely_fast_whisper_rocm.webui.handlers.stabilize_timestamps"
        ) as mock_stabilize,
    ):
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_transcription.return_value = {
            "text": "test transcription"
        }
        mock_create_orchestrator.return_value = mock_orchestrator

        # Make the mocked stabilize_timestamps function return its input
        mock_stabilize.side_effect = lambda result, **kwargs: result

        yield mock_orchestrator, mock_stabilize


def test_transcribe_handler_with_stabilization(
    mock_pipeline_and_stabilizer: tuple[MagicMock, MagicMock],
) -> None:
    """Test that the transcribe handler calls stabilize_timestamps with correct args."""
    _mock_pipeline, mock_stabilize = mock_pipeline_and_stabilizer

    config = TranscriptionConfig(
        stabilize=True,
        demucs=True,
        vad=True,
        vad_threshold=0.6,
        suppress_ts_tokens=False,
        gap_padding=" lead",
        adjust_gaps=False,
        nonspeech_skip=1.5,
    )

    transcribe(audio_file_path="dummy.wav", config=config, file_config=MagicMock())

    mock_stabilize.assert_called_once()
    _call_args, call_kwargs = mock_stabilize.call_args
    assert call_kwargs.get("demucs") is True
    assert call_kwargs.get("vad") is True
    assert call_kwargs.get("vad_threshold") == 0.6
    assert call_kwargs.get("suppress_ts_tokens") is False
    assert call_kwargs.get("gap_padding") == " lead"
    assert call_kwargs.get("adjust_gaps") is False
    assert call_kwargs.get("nonspeech_skip") == 1.5


def test_transcribe_handler_without_stabilization(
    mock_pipeline_and_stabilizer: tuple[MagicMock, MagicMock],
) -> None:
    """Test that stabilize_timestamps is not called when stabilization is disabled."""
    _mock_pipeline, mock_stabilize = mock_pipeline_and_stabilizer

    config = TranscriptionConfig(stabilize=False)

    transcribe(audio_file_path="dummy.wav", config=config, file_config=MagicMock())

    mock_stabilize.assert_not_called()


def test_transcribe_handler_video__applies_subtitle_sync() -> None:
    """Run ALASS sync for video inputs when subtitle sync is enabled."""
    with (
        patch(
            "insanely_fast_whisper_rocm.webui.handlers.create_orchestrator"
        ) as mock_create_orchestrator,
        patch(
            "insanely_fast_whisper_rocm.webui.handlers.extract_audio_from_video",
            return_value="dummy.wav",
        ),
        patch(
            "insanely_fast_whisper_rocm.webui.handlers.sync_subtitle_with_alass"
        ) as mock_sync,
    ):
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_transcription.return_value = {
            "text": "hello",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "chunks": [],
        }
        mock_create_orchestrator.return_value = mock_orchestrator
        mock_sync.return_value = (
            "1\n00:00:01,000 --> 00:00:02,000\nhello\n",
            {
                "enabled": True,
                "engine": "alass",
                "applied": True,
                "reason": None,
                "runtime_ms": 8,
            },
        )

        config = TranscriptionConfig(stabilize=False, subtitle_sync=True)
        result = transcribe(
            audio_file_path="dummy.mp4",
            config=config,
            file_config=MagicMock(),
        )

    mock_sync.assert_called_once()
    assert result["subtitle_sync"]["applied"] is True
    assert "srt_synced_text" in result
