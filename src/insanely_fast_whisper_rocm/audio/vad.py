"""Voice Activity Detection utilities for pre-segmentation muting.

This module uses Silero VAD to detect speech in an audio file, and mathematically
zeroes out any non-speech regions to provide absolute digital silence. Whisper cannot
hallucinate text over perfect mathematical silence, thereby forcing Whisper to align
timestamps exclusively to actual speech without breaking batched alignment.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import torch
import torchaudio

logger = logging.getLogger(__name__)

_VAD_MODEL = None
_VAD_UTILS = None


def _load_vad_model() -> tuple[Any, Any]:
    """Lazy load the Silero VAD model from PyTorch Hub.

    Returns:
        tuple containing the VAD model and its utility functions.

    Raises:
        RuntimeError: If the VAD model fails to load.
    """
    global _VAD_MODEL, _VAD_UTILS

    if _VAD_MODEL is None:
        logger.info("Loading Silero VAD model from PyTorch Hub...")
        try:
            _VAD_MODEL, _VAD_UTILS = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
                verbose=False,
            )
            logger.info("Silero VAD model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load Silero VAD: %s", e)
            raise RuntimeError(f"Could not load VAD model: {e}") from e

    return _VAD_MODEL, _VAD_UTILS


def mute_non_speech(audio_path: str, vad_threshold: float = 0.35) -> str:
    """Mute all non-speech regions in the audio to absolute silence.

    Reads the audio file, computes speech timestamps using Silero VAD at 16kHz,
    maps those timestamps back to the original sample rate, and zeroes out the
    array outside of those intervals. Exported to a new WAV file.

    Args:
        audio_path: Path to the input audio file.
        vad_threshold: Sensitivity threshold for Silero VAD (0.0 to 1.0).

    Returns:
        Path to the newly generated WAV file containing muted non-speech regions.
    """
    try:
        model, utils = _load_vad_model()
        get_speech_timestamps = utils[0]

        import soundfile as sf

        logger.info(
            "Scanning for speech in %s with threshold %f", audio_path, vad_threshold
        )

        # Load audio using soundfile to avoid torchaudio backend constraints
        wav_np, sr = sf.read(audio_path)

        # Convert to mono if there are multiple channels (samples, channels)
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=1)

        wav = torch.from_numpy(wav_np).float().unsqueeze(0)  # Shape: (1, samples)

        # Silero specifically operates at 16kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav_16k = resampler(wav)
        else:
            wav_16k = wav

        # Squeeze out the channel dimension for the VAD model
        wav_16k = wav_16k.squeeze(0)

        # Retrieve speech ranges in sample counts relative to 16kHz format
        speech_timestamps = get_speech_timestamps(
            wav_16k, model, threshold=vad_threshold, return_seconds=False
        )

        if not speech_timestamps:
            logger.warning(
                "No speech found by VAD in %s. Returning uninterrupted audio.",
                audio_path,
            )
            return audio_path

        # Create an identical array filled with digital silence (all zeros)
        muted_wav = torch.zeros_like(wav)

        total_speech_samples = 0
        total_samples = wav.shape[1]

        # Copy over only the segments identified as valid speech
        for seg in speech_timestamps:
            start_samp_16k = seg["start"]
            end_samp_16k = seg["end"]

            # Translate 16kHz sample indices to the original sample rate
            start_orig = int(start_samp_16k * sr / 16000)
            end_orig = int(end_samp_16k * sr / 16000)

            muted_wav[:, start_orig:end_orig] = wav[:, start_orig:end_orig]
            total_speech_samples += end_orig - start_orig

        logger.info(
            "VAD complete: Preserved %d/%d samples as speech (%.1f%%).",
            total_speech_samples,
            total_samples,
            (total_speech_samples / total_samples) * 100 if total_samples else 0,
        )

        # Export to a temporary directory since it's an intermediate processing file
        tmp_dir = tempfile.mkdtemp(prefix="vad_muted_")
        base_name = os.path.basename(audio_path)
        out_name = os.path.splitext(base_name)[0] + ".wav"
        out_path = os.path.join(tmp_dir, out_name)

        sf.write(out_path, muted_wav.squeeze(0).numpy(), sr)
        return out_path

    except Exception as e:
        logger.error("Failed to apply VAD muting to %s: %s", audio_path, e)
        return audio_path
