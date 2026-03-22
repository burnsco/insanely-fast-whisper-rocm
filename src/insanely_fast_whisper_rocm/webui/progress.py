"""Progress adapters for the optional Gradio WebUI."""

from __future__ import annotations

import gradio as gr

from insanely_fast_whisper_rocm.core.cancellation import CancellationToken
from insanely_fast_whisper_rocm.core.progress import ProgressCallback


class WebUIProgressCallback(ProgressCallback):
    """Translate pipeline progress callbacks into Gradio progress updates."""

    def __init__(
        self,
        tracker: gr.Progress,
        base: float,
        total: int,
        name: str,
        cancel_token: CancellationToken,
    ) -> None:
        """Initialize the progress adapter.

        Args:
            tracker: Gradio progress tracker instance.
            base: Base progress offset for the current file.
            total: Total number of files in the session.
            name: Human-friendly filename.
            cancel_token: Shared cancellation token.
        """
        self.tracker = tracker
        self.base = base
        self.total = total
        self.name = name
        self.cancel_token = cancel_token
        self._total_chunks: int | None = None

    def _update(self, fraction: float | None, message: str) -> None:
        """Apply an update to the Gradio tracker.

        Args:
            fraction: Optional fractional progress for the current file.
            message: Status message to display.
        """
        if self.cancel_token.cancelled:
            return
        if getattr(self.tracker, "cancelled", False):
            self.cancel_token.cancel()
            return

        description = f"{message} ({self.name})"
        if fraction is not None:
            self.tracker(self.base + (fraction / self.total), desc=description)
            return
        self.tracker(None, desc=description)

    def on_model_load_started(self) -> None:
        """Handle model load start."""
        self._update(None, "Loading model...")

    def on_model_load_finished(self) -> None:
        """Handle model load completion."""
        self._update(None, "Model loaded")

    def on_audio_loading_started(self, path: str) -> None:  # noqa: ARG002
        """Handle audio preparation start."""
        self._update(None, "Preparing audio...")

    def on_audio_loading_finished(self, duration_sec: float | None) -> None:  # noqa: ARG002
        """Handle audio preparation completion."""
        self._update(None, "Audio ready")

    def on_chunking_started(self, total_chunks: int | None) -> None:
        """Handle chunking start."""
        self._total_chunks = total_chunks
        self._update(0.0, "Starting transcription...")

    def on_chunk_done(self, index: int) -> None:
        """Handle chunk completion."""
        if not self._total_chunks:
            self._update(None, "Transcribing...")
            return

        chunk_number = index + 1
        fraction = min(chunk_number / self._total_chunks, 1.0)
        self._update(
            fraction,
            f"Processing chunk {chunk_number}/{self._total_chunks}",
        )

    def on_inference_started(self, total_batches: int | None) -> None:  # noqa: ARG002
        """Handle inference start."""

    def on_inference_batch_done(self, index: int) -> None:  # noqa: ARG002
        """Handle inference batch completion."""

    def on_postprocess_started(self, name: str) -> None:
        """Handle post-processing start."""
        self._update(None, f"Post-processing: {name}")

    def on_postprocess_finished(self, name: str) -> None:
        """Handle post-processing completion."""
        self._update(None, f"Post-processing done: {name}")

    def on_export_started(self, total_items: int) -> None:
        """Handle export start."""
        self._update(None, f"Exporting {total_items} file(s)...")

    def on_export_item_done(self, index: int, label: str) -> None:
        """Handle export item completion."""
        self._update(None, f"Exported: {label} ({index + 1})")

    def on_completed(self) -> None:
        """Handle overall completion."""
        self._update(1.0, "Transcription complete")

    def on_error(self, message: str) -> None:
        """Handle error reporting."""
        self._update(None, f"Error: {message}")
