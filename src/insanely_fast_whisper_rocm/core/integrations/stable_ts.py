"""Stable-ts integration wrapper.

Provides `stabilize_timestamps` to refine Whisper transcription results using
`stable-whisper`'s `transcribe_any` convenience function.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from insanely_fast_whisper_rocm.utils import constant as constants
from insanely_fast_whisper_rocm.utils.timestamp_utils import (
    normalize_timestamp_format,
    validate_timestamps,
)

logger = logging.getLogger(__name__)

try:
    import stable_whisper as _stable_whisper

    # Prefer explicit function if available; also support common alias 'postprocess'.
    stable_whisper: Any | None = _stable_whisper
    _postprocess = cast(
        Callable[..., object] | None,
        getattr(stable_whisper, "postprocess_word_timestamps", None),
    )
    _postprocess_alt = cast(
        Callable[..., object] | None,
        getattr(stable_whisper, "postprocess", None),
    )
except ImportError as err:  # pragma: no cover
    logger.error(
        "stable-whisper is not installed – stabilize_timestamps will be a no-op: %s",
        err,
    )
    stable_whisper = None
    _postprocess = None
    _postprocess_alt = None


def _segments_have_timestamps(seg_list: object) -> bool:
    """Check whether a segment payload contains usable timestamps.

    Args:
        seg_list: Potential sequence of segment dictionaries.

    Returns:
        True if any segment exposes both start and end timestamps.
    """
    if not isinstance(seg_list, list):
        return False
    for segment in seg_list:
        if not isinstance(segment, dict):
            continue
        segment_dict = cast(dict[str, Any], segment)
        if (
            segment_dict.get("start") is not None
            and segment_dict.get("end") is not None
        ):
            return True
    return False


def _to_dict(obj: object) -> dict[str, Any]:
    """Convert the result object returned by *stable-whisper* to a dictionary.

    Args:
        obj: The object returned by stable-whisper; can be a dict or a model-like
            object exposing ``to_dict``/``model_dump``/``_asdict``.

    Returns:
        dict[str, Any]: A dictionary representation of the input object.
    """
    logger.info("_to_dict called with type=%s", type(obj))
    if isinstance(obj, dict):
        return cast(dict[str, Any], obj)
    for attr in ("to_dict", "model_dump", "_asdict"):
        if hasattr(obj, attr):
            converter = cast(Callable[[], dict[str, Any]], getattr(obj, attr))
            return converter()
    return {"text": str(obj)}


def _filter_supported_kwargs(
    func: Callable[..., object],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Return only keyword arguments accepted by ``func``.

    Args:
        func: Callable that will receive keyword arguments.
        kwargs: Candidate keyword arguments.

    Returns:
        A filtered copy containing only supported keyword arguments. If the
        callable accepts ``**kwargs``, all arguments are preserved.
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return kwargs

    parameters = signature.parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return kwargs

    supported = {param.name for param in parameters}
    return {key: value for key, value in kwargs.items() if key in supported}


def _convert_to_stable(result: dict[str, Any]) -> dict[str, Any]:
    """Return *result* reshaped to match Whisper JSON expected by stable-ts."""
    logger.info("_convert_to_stable input keys=%s", list(result.keys()))

    # Use centralized normalization function
    converted = normalize_timestamp_format(result)
    logger.info("_convert_to_stable: normalized timestamp format")

    # Fix individual segment fields and validate timestamps
    segments = cast(list[dict[str, Any]], converted.get("segments", []))
    validated_segments = validate_timestamps(segments)
    converted["segments"] = validated_segments

    # Debug: check if segments are word-level or sentence-level
    if validated_segments:
        sample_seg = validated_segments[0]
        max_segments = min(10, len(validated_segments))
        avg_seg_dur = (
            sum(s["end"] - s["start"] for s in validated_segments[:10]) / max_segments
        )
        logger.info(
            "_convert_to_stable: %d segments, avg_dur=%.3fs, sample_text=%r",
            len(validated_segments),
            avg_seg_dur,
            sample_seg.get("text", "")[:30],
        )
    return converted


def stabilize_timestamps(
    result: dict[str, Any],
    *,
    demucs: bool = False,
    vad: bool = False,
    vad_threshold: float = 0.35,
    suppress_ts_tokens: bool = True,
    gap_padding: str = " ...",
    adjust_gaps: bool = True,
    nonspeech_skip: float | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Return a copy of *result* with word-level timestamps via stable-ts.

    Args:
        result: Base transcription result to refine.
        demucs: Whether to run Demucs denoising.
        vad: Whether to run Voice Activity Detection.
        vad_threshold: VAD threshold when ``vad`` is True.
        suppress_ts_tokens: Whether to suppress timestamp tokens in silent spans.
        gap_padding: Padding used to reduce timestamps starting before speech.
        adjust_gaps: Whether to tighten segment gaps using stable-ts gap logic.
        nonspeech_skip: Optional non-speech duration threshold to skip entirely.
        progress_cb: Optional callback receiving human-readable status updates.

    Returns:
        A refined result dictionary. If stabilization fails, the original
        ``result`` is returned.
    """
    logger.info(
        "stabilize_timestamps called demucs=%s vad=%s vad_threshold=%s "
        "suppress_ts_tokens=%s adjust_gaps=%s nonspeech_skip=%s",
        demucs,
        vad,
        vad_threshold,
        suppress_ts_tokens,
        adjust_gaps,
        nonspeech_skip,
    )
    if stable_whisper is None:
        logger.warning(
            "stable-whisper not available – returning original result unchanged"
        )
        if progress_cb:
            progress_cb("stable-ts unavailable; skipping stabilization")
        return result

    audio_path_str = result.get("original_file") or result.get("audio_file_path")
    if not audio_path_str:
        logger.error(
            "Audio path missing from transcription result; cannot stabilize timestamps"
        )
        if progress_cb:
            progress_cb("stable-ts: audio path missing; skipping")
        return result

    # Respect filesystem in production; allow skipping in tests.
    audio_path = Path(audio_path_str).expanduser().resolve()
    if not audio_path.exists() and not constants.SKIP_FS_CHECKS:
        logger.error("Audio file not found for stabilization: %s", audio_path)
        if progress_cb:
            progress_cb("stable-ts: audio file not found; skipping")
        return result

    # Prepare a stable-whisper–compatible dict
    converted = _convert_to_stable(result)

    def inference_func(*_a: object, **_k: object) -> dict[str, Any]:
        """Return the precomputed converted dict regardless of inputs.

        This mirrors the previous lambda behavior used for transcribe_any.
        """
        return converted

    def _apply_gap_adjustment(refined: object) -> object:
        """Apply stable-ts gap adjustment when the result object supports it.

        Args:
            refined: Stable-ts result object that may expose ``adjust_gaps``.

        Returns:
            The adjusted result object, or the original object when no
            adjustment is available or necessary.
        """
        if not adjust_gaps:
            return refined

        adjust_gaps_method = getattr(refined, "adjust_gaps", None)
        if not callable(adjust_gaps_method):
            return refined

        try:
            adjusted = adjust_gaps_method(one_section=True)
        except TypeError:
            adjusted = adjust_gaps_method()
        except Exception as exc:  # pragma: no cover
            logger.warning("stable-ts gap adjustment failed: %s", exc)
            if progress_cb:
                progress_cb("stable-ts: gap adjustment failed; keeping base timings")
            return refined

        if progress_cb:
            progress_cb("stable-ts: gap adjustment applied")
        return refined if adjusted is None else adjusted

    common_kwargs: dict[str, Any] = {
        "vad": vad,
        "vad_threshold": vad_threshold,
        "suppress_ts_tokens": suppress_ts_tokens,
        "gap_padding": gap_padding,
    }
    if nonspeech_skip is not None:
        common_kwargs["nonspeech_skip"] = nonspeech_skip

    # 1. Preferred paths: postprocess_* style APIs (avoid torchaudio.save)
    for func_name, func in (
        ("postprocess_word_timestamps", _postprocess),
        ("postprocess", _postprocess_alt),
    ):
        if func is None:
            continue
        try:
            if progress_cb:
                progress_cb(f"stable-ts: {func_name} running")
            try:
                refined = func(
                    converted,
                    audio=str(audio_path),
                    demucs=demucs,
                    **common_kwargs,
                )
            except TypeError:
                # Some versions may not accept the same kwargs; retry with minimal args.
                refined = func(converted, audio=str(audio_path))
            refined = _apply_gap_adjustment(refined)
            refined_dict = _to_dict(refined)
            merged = {**result, **refined_dict, "stabilized": True}
            if _segments_have_timestamps(merged.get("segments")):
                merged.pop("chunks", None)

            # Debug: check output from stable-ts
            output_segs = cast(list[dict[str, Any]], merged.get("segments", []))
            if output_segs:
                max_segments = min(10, len(output_segs))
                avg_out_dur = (
                    sum(s["end"] - s["start"] for s in output_segs[:10]) / max_segments
                )
                logger.info(
                    "stable-ts output: %d segments, avg_dur=%.3fs",
                    len(output_segs),
                    avg_out_dur,
                )

            merged.setdefault("segments_count", len(refined_dict.get("segments", [])))
            merged.setdefault("stabilization_path", func_name)
            if progress_cb:
                progress_cb(f"stable-ts: refinement successful ({func_name})")
            return merged
        except Exception as exc:  # pragma: no cover
            logger.warning("%s failed: %s", func_name, exc)
            if progress_cb:
                progress_cb(f"stable-ts: {func_name} failed; trying alt path")

    # 2. Alternative path: lambda-inference via transcribe_any
    # (may require torchaudio.save)
    try:
        if progress_cb:
            progress_cb(
                f"stable-ts: running (demucs={demucs}, vad={vad}, thr={vad_threshold})"
            )
        transcribe_any_kwargs: dict[str, Any] = {
            "audio": str(audio_path),
            "check_sorted": False,
            "vad": vad,
            "vad_threshold": vad_threshold,
            "demucs": demucs,
            "gap_padding": gap_padding,
            # Support both names across stable-whisper versions.
            "suppress_ts_tokens": suppress_ts_tokens,
            # stable-whisper 2.19.x uses this older parameter name.
            "suppress_word_ts": suppress_ts_tokens,
        }
        if nonspeech_skip is not None:
            transcribe_any_kwargs["nonspeech_skip"] = nonspeech_skip
        if demucs:
            transcribe_any_kwargs["denoiser"] = "demucs"

        refined = stable_whisper.transcribe_any(
            inference_func,
            **_filter_supported_kwargs(
                stable_whisper.transcribe_any,
                transcribe_any_kwargs,
            ),
        )
        refined = _apply_gap_adjustment(refined)
        refined_dict = _to_dict(refined)
        output_segs = refined_dict.get("segments", [])
        if output_segs:
            max_segments = min(10, len(output_segs))
            avg_out_dur = (
                sum(s.get("end", 0) - s.get("start", 0) for s in output_segs[:10])
                / max_segments
            )
            logger.info(
                "stable-ts (transcribe_any) output: %d segments, avg_dur=%.3fs",
                len(output_segs),
                avg_out_dur,
            )
        if progress_cb:
            progress_cb("stable-ts: refinement successful")
        merged = {**result, **refined_dict, "stabilized": True}
        if _segments_have_timestamps(merged.get("segments")):
            # Only discard original chunks if we actually obtained usable timestamps
            merged.pop("chunks", None)
        # Enrich with metadata before returning (lazy logging ready)
        merged.setdefault("segments_count", len(refined_dict.get("segments", [])))
        merged.setdefault("stabilization_path", "lambda")
        if progress_cb:
            progress_cb("stable-ts: merging results")
        return merged
    except Exception as exc:  # pragma: no cover
        logger.error("stable-ts lambda inference path failed: %s", exc, exc_info=True)
        if progress_cb:
            progress_cb("stable-ts: alternative path failed")

    # 3. Give up
    logger.error("stable-ts processing failed; returning original result")
    if progress_cb:
        progress_cb("stable-ts: failed; returning original result")
    return result
