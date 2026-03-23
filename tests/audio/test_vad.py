"""Tests for VAD pre-muting functionality."""

from __future__ import annotations

import os
import tempfile
from unittest import mock

import numpy as np
import pytest
import soundfile as sf
import torch

import insanely_fast_whisper_rocm.audio.vad as vad
from insanely_fast_whisper_rocm.audio.vad import mute_non_speech


@pytest.fixture(autouse=True)
def clear_vad_cache() -> None:
    """Clear the globally cached VAD model state between tests."""
    vad._VAD_MODEL = None
    vad._VAD_UTILS = None


@pytest.fixture
def mock_silero_vad() -> mock.MagicMock:
    """Mock the Silero VAD model loading and execution.

    Yields:
        The mocked load function.
    """
    with mock.patch("insanely_fast_whisper_rocm.audio.vad.torch.hub.load") as mock_load:

        def fake_get_speech_timestamps(
            wav: torch.Tensor, model: str, threshold: float, return_seconds: bool
        ) -> list[dict[str, int]]:
            # Fake speech from sample 16000 to 32000 (1 to 2 seconds at 16kHz)
            return [{"start": 16000, "end": 32000}]

        mock_load.return_value = ("fake_model", [fake_get_speech_timestamps])
        yield mock_load


def test_mute_non_speech(mock_silero_vad: mock.MagicMock) -> None:
    """Test that mute_non_speech zeroes out audio outside of speech segments."""
    # Create a 3-second 16kHz dummy audio file filled with ones
    sr = 16000
    duration_sec = 3
    wav_np = np.ones((sr * duration_sec,), dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        audio_path = tmp_audio.name

    try:
        sf.write(audio_path, wav_np, sr)

        # Run mute_non_speech
        muted_path = mute_non_speech(audio_path, vad_threshold=0.5)

        assert muted_path != audio_path
        assert os.path.exists(muted_path)

        # Load the muted audio
        muted_wav_np, muted_sr = sf.read(muted_path)
        muted_wav = torch.from_numpy(muted_wav_np).unsqueeze(0)
        assert muted_sr == sr
        assert muted_wav.numel() == len(wav_np)

        # Verify the first second (0 to 16000) is zeros
        assert torch.all(muted_wav[:, :16000] == 0)

        # Verify the second second (16000 to 32000) is preserved (lossy PCM check)
        assert torch.all(muted_wav[:, 16000:32000] > 0.99)

        # Verify the third second (32000 to 48000) is zeros
        assert torch.all(muted_wav[:, 32000:] == 0)

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if "muted_path" in locals() and os.path.exists(muted_path):
            os.remove(muted_path)


def test_mute_non_speech_no_speech_found() -> None:
    """Test that mute_non_speech returns original path if no speech is found."""
    with mock.patch("insanely_fast_whisper_rocm.audio.vad.torch.hub.load") as mock_load:

        def fake_get_speech_timestamps(
            wav: torch.Tensor, model: str, threshold: float, return_seconds: bool
        ) -> list[dict[str, int]]:
            return []  # No speech

        mock_load.return_value = ("fake_model", [fake_get_speech_timestamps])

        sr = 16000
        wav_np = np.ones((sr * 1,), dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            audio_path = tmp_audio.name

        try:
            sf.write(audio_path, wav_np, sr)

            muted_path = mute_non_speech(audio_path, vad_threshold=0.5)

            # Should return original path
            assert muted_path == audio_path

        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
