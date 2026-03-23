"""Gradio UI components for the Insanely Fast Whisper API.

This module provides the main UI components and layout for the web interface,
including file upload, processing controls, and result display components.
"""

import logging
from typing import Literal

import gradio as gr

from insanely_fast_whisper_rocm.utils.constant import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_TRANSCRIPTS_DIR,
    MAX_BATCH_SIZE,
    MIN_BATCH_SIZE,
    SUPPORTED_UPLOAD_FORMATS,
    WEBUI_DEFAULT_ADJUST_GAPS,
    WEBUI_DEFAULT_DEMUCS,
    WEBUI_DEFAULT_GAP_PADDING,
    WEBUI_DEFAULT_NONSPEECH_SKIP,
    WEBUI_DEFAULT_STABILIZE,
    WEBUI_DEFAULT_SUBTITLE_SYNC,
    WEBUI_DEFAULT_SUPPRESS_TS_TOKENS,
    WEBUI_DEFAULT_TASK,
    WEBUI_DEFAULT_TIMESTAMP_TYPE,
    WEBUI_DEFAULT_VAD,
    WEBUI_DEFAULT_VAD_THRESHOLD,
)
from insanely_fast_whisper_rocm.webui.handlers import process_transcription_request
from insanely_fast_whisper_rocm.webui.models import (
    FileHandlingConfig,
    TranscriptionConfig,
)

# Configure logger
logger = logging.getLogger("insanely_fast_whisper_rocm.webui.ui")

TimestampType = Literal["chunk", "word"]
TaskType = Literal["transcribe", "translate"]


def _create_model_config_ui(
    default_model: str = DEFAULT_MODEL,
) -> tuple[gr.Textbox, gr.Textbox, gr.Slider]:
    """Helper to create model configuration UI components with a default model.

    Returns:
        tuple[gr.Textbox, gr.Textbox, gr.Slider]: The model, device, and
        batch size controls.
    """
    with gr.Accordion("Model Configuration", open=True):
        model = gr.Textbox(value=default_model, label="Model")
        device = gr.Textbox(
            value=DEFAULT_DEVICE,
            label="Device (use 0 for the first ROCm GPU, or cpu)",
        )
        batch_size = gr.Slider(
            minimum=MIN_BATCH_SIZE,
            maximum=MAX_BATCH_SIZE,
            step=1,
            value=DEFAULT_BATCH_SIZE,
            label="Batch Size",
        )
    return model, device, batch_size


def _create_processing_options_ui() -> tuple[gr.Dropdown, gr.Slider]:
    """Helper function to create processing options UI components.

    Returns:
        tuple[gr.Dropdown, gr.Slider]: The dtype dropdown and chunk length slider.
    """
    with gr.Accordion("Processing Options", open=False):
        dtype = gr.Dropdown(
            choices=["float16", "float32"],
            value="float16",
            label="Precision",
            info="Lower precision (float16) is faster but may be less accurate",
        )
        chunk_length = gr.Slider(
            minimum=10,
            maximum=60,
            step=5,
            value=30,
            label="Processing Chunk Length (seconds)",
            info=(
                "Length of audio segments for model processing. "
                "Longer chunks may be more accurate but use more memory"
            ),
        )
    return dtype, chunk_length


def _parse_optional_float(value: str | float | int | None) -> float | None:
    """Parse an optional numeric input from the WebUI.

    Args:
        value: Raw component value, which may be blank.

    Returns:
        The parsed float value, or ``None`` when left blank.

    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _toggle_stabilization_advanced(enabled: bool) -> dict[str, object]:
    """Show or hide advanced stabilization controls.

    Args:
        enabled: Whether the advanced controls should be visible.

    Returns:
        A Gradio update payload that toggles visibility.
    """
    return gr.update(visible=enabled)


def _create_stabilization_ui(
    *,
    default_stabilize: bool = WEBUI_DEFAULT_STABILIZE,
    default_demucs: bool = WEBUI_DEFAULT_DEMUCS,
    default_vad: bool = WEBUI_DEFAULT_VAD,
    default_vad_threshold: float = WEBUI_DEFAULT_VAD_THRESHOLD,
    default_subtitle_sync: bool = WEBUI_DEFAULT_SUBTITLE_SYNC,
    default_suppress_ts_tokens: bool = WEBUI_DEFAULT_SUPPRESS_TS_TOKENS,
    default_gap_padding: str = WEBUI_DEFAULT_GAP_PADDING,
    default_adjust_gaps: bool = WEBUI_DEFAULT_ADJUST_GAPS,
    default_nonspeech_skip: float | None = WEBUI_DEFAULT_NONSPEECH_SKIP,
) -> tuple[
    gr.Checkbox,
    gr.Checkbox,
    gr.Checkbox,
    gr.Checkbox,
    gr.Slider,
    gr.Checkbox,
    gr.Textbox,
    gr.Checkbox,
    gr.Textbox,
    gr.Accordion,
]:
    """Helper function to create timestamp stabilization UI components.

    Returns:
        The base stabilization controls, advanced controls, and the advanced
        accordion container for visibility toggling.
    """
    with gr.Accordion("Timestamp Stabilization", open=False):
        stabilize = gr.Checkbox(
            value=default_stabilize,
            label="Enable word-level stabilization (--stabilize)",
            info=(
                "Recommended for movie/TV subtitle timing. Start with "
                "Timestamp Type = word, VAD on (0.35), Demucs off."
            ),
        )
        demucs = gr.Checkbox(
            value=default_demucs, label="Use Demucs noise reduction (--demucs)"
        )
        vad = gr.Checkbox(value=default_vad, label="Enable VAD (--vad)")
        subtitle_sync = gr.Checkbox(
            value=default_subtitle_sync,
            label="Enable subtitle sync with ALASS",
            info="Synchronizes generated SRT output to video media when available.",
        )
        vad_threshold = gr.Slider(
            minimum=0.1,
            maximum=0.9,
            step=0.05,
            value=default_vad_threshold,
            label="VAD Threshold (--vad-threshold)",
        )
        with gr.Accordion(
            "Advanced Stabilization",
            open=False,
            visible=default_stabilize,
        ) as advanced_stabilization:
            suppress_ts_tokens = gr.Checkbox(
                value=default_suppress_ts_tokens,
                label="Suppress timestamp tokens in silence",
                info="Helps reduce words appearing before they are actually spoken.",
            )
            gap_padding = gr.Textbox(
                value=default_gap_padding,
                label="Gap padding",
                info="Padding passed to stable-ts to reduce early word starts.",
            )
            adjust_gaps = gr.Checkbox(
                value=default_adjust_gaps,
                label="Adjust segment gaps",
                info="Tightens subtitle blocks around detected non-speech gaps.",
            )
            nonspeech_skip = gr.Textbox(
                value=(
                    ""
                    if default_nonspeech_skip is None
                    else str(default_nonspeech_skip)
                ),
                label="Skip non-speech spans >= (seconds)",
                placeholder="Leave blank to disable",
                info=(
                    "Optional. Skip long non-speech regions entirely to reduce "
                    "timing drift and hallucinations."
                ),
            )
    return (
        stabilize,
        demucs,
        vad,
        subtitle_sync,
        vad_threshold,
        suppress_ts_tokens,
        gap_padding,
        adjust_gaps,
        nonspeech_skip,
        advanced_stabilization,
    )


def _create_task_config_ui(
    default_timestamp_type: TimestampType = WEBUI_DEFAULT_TIMESTAMP_TYPE,
    default_task: TaskType = WEBUI_DEFAULT_TASK,
) -> tuple[gr.Radio, gr.Textbox, gr.Radio]:
    """Helper function to create task configuration UI components.

    Returns:
        tuple[gr.Radio, gr.Textbox, gr.Radio]: Timestamp type, language, and
        task controls.
    """
    with gr.Accordion("Task Configuration", open=True):
        timestamp_type = gr.Radio(
            choices=["chunk", "word"],
            label="Timestamp Type",
            value=default_timestamp_type,
            info=(
                "For movie/TV subtitle timing, 'word' works best with "
                "stabilization enabled."
            ),
        )
        language = gr.Textbox(
            value=DEFAULT_LANGUAGE,
            label="Language in ISO format (use 'None' for auto detection)",
            placeholder="en, fr, de, etc.",
        )
        task = gr.Radio(
            choices=["transcribe", "translate"],
            label="Task",
            value=default_task,
        )
    return timestamp_type, language, task


def _create_file_handling_ui() -> tuple[gr.Checkbox, gr.Textbox]:
    """Helper function to create file handling UI components.

    Returns:
        tuple[gr.Checkbox, gr.Textbox]: Save transcriptions toggle and save
        directory input.
    """
    with gr.Accordion("File Handling", open=False):
        save_transcriptions = gr.Checkbox(
            value=True, label="Save transcriptions to disk"
        )
        temp_uploads_dir = gr.Textbox(
            value=DEFAULT_TRANSCRIPTS_DIR,
            label="Save directory",
            info="Directory to save transcription results",
        )
    return save_transcriptions, temp_uploads_dir


def _process_transcription_request_wrapper(
    audio_paths: list[str],
    model_name: str,
    device: str,
    batch_size: int,
    timestamp_type: TimestampType,
    language: str,
    task: TaskType,
    dtype: str,
    whisper_chunk_length: int,
    # Stabilization params
    stabilize: bool,
    demucs: bool,
    vad: bool,
    subtitle_sync: bool,
    vad_threshold: float,
    suppress_ts_tokens: bool,
    gap_padding: str,
    adjust_gaps: bool,
    nonspeech_skip: str,
    save_transcriptions: bool,
    temp_uploads_dir: str,
    progress: gr.Progress | None = None,
) -> tuple[object, ...]:
    """Wrapper to adapt Gradio inputs to process_transcription_request.

    Returns:
        tuple: The outputs expected by the Gradio click handler (text,
        JSON, generated files, and download button updates).
    """
    if progress is None:
        progress = gr.Progress()
    transcription_cfg = TranscriptionConfig(
        model=model_name,
        device=device,
        batch_size=batch_size,
        timestamp_type=timestamp_type,
        language=language,
        task=task,
        dtype=dtype,
        chunk_length=whisper_chunk_length,
        chunk_duration=None,
        chunk_overlap=None,
    )
    file_handling_cfg = FileHandlingConfig(
        save_transcriptions=save_transcriptions, temp_uploads_dir=temp_uploads_dir
    )
    # Inject stabilization options
    transcription_cfg.stabilize = stabilize
    transcription_cfg.demucs = demucs
    transcription_cfg.vad = vad
    transcription_cfg.subtitle_sync = subtitle_sync
    transcription_cfg.vad_threshold = vad_threshold
    transcription_cfg.suppress_ts_tokens = suppress_ts_tokens
    transcription_cfg.gap_padding = gap_padding
    transcription_cfg.adjust_gaps = adjust_gaps
    transcription_cfg.nonspeech_skip = _parse_optional_float(nonspeech_skip)
    return process_transcription_request(
        audio_paths=audio_paths,
        transcription_config=transcription_cfg,
        file_handling_config=file_handling_cfg,
        progress_tracker=progress,
    )


def create_ui_components(
    *,
    default_model: str = DEFAULT_MODEL,
    default_stabilize: bool = WEBUI_DEFAULT_STABILIZE,
    default_demucs: bool = WEBUI_DEFAULT_DEMUCS,
    default_vad: bool = WEBUI_DEFAULT_VAD,
    default_vad_threshold: float = WEBUI_DEFAULT_VAD_THRESHOLD,
    default_subtitle_sync: bool = WEBUI_DEFAULT_SUBTITLE_SYNC,
    default_suppress_ts_tokens: bool = WEBUI_DEFAULT_SUPPRESS_TS_TOKENS,
    default_gap_padding: str = WEBUI_DEFAULT_GAP_PADDING,
    default_adjust_gaps: bool = WEBUI_DEFAULT_ADJUST_GAPS,
    default_nonspeech_skip: float | None = WEBUI_DEFAULT_NONSPEECH_SKIP,
) -> gr.Blocks:  # pylint: disable=too-many-locals
    """Create and return Gradio UI components with all parameters.

    Returns:
        gr.Blocks: The configured Gradio Blocks interface instance.
    """
    with gr.Blocks(title="Insanely Fast Whisper - Local WebUI") as demo:
        gr.Markdown("# 🎙️ Insanely Fast Whisper - Local WebUI")
        gr.Markdown(
            "Transcribe or translate audio and video files using Whisper "
            "models directly in your browser."
        )

        with gr.Row():
            with gr.Column(scale=2):
                # Audio input
                audio_input = gr.File(
                    label="Upload Audio File(s)",
                    type="filepath",
                    file_count="multiple",
                    file_types=list(SUPPORTED_UPLOAD_FORMATS),
                )

                # Model configuration
                model, device, batch_size = _create_model_config_ui(default_model)

                # Processing options
                dtype, chunk_length = _create_processing_options_ui()

                # Timestamp stabilization options
                (
                    stabilize_opt,
                    demucs_opt,
                    vad_opt,
                    subtitle_sync_opt,
                    vad_threshold_opt,
                    suppress_ts_tokens_opt,
                    gap_padding_opt,
                    adjust_gaps_opt,
                    nonspeech_skip_opt,
                    advanced_stabilization,
                ) = _create_stabilization_ui(
                    default_stabilize=default_stabilize,
                    default_demucs=default_demucs,
                    default_vad=default_vad,
                    default_subtitle_sync=default_subtitle_sync,
                    default_vad_threshold=default_vad_threshold,
                    default_suppress_ts_tokens=default_suppress_ts_tokens,
                    default_gap_padding=default_gap_padding,
                    default_adjust_gaps=default_adjust_gaps,
                    default_nonspeech_skip=default_nonspeech_skip,
                )

                # Task configuration
                timestamp_type, language, task = _create_task_config_ui(
                    default_timestamp_type=WEBUI_DEFAULT_TIMESTAMP_TYPE,
                    default_task=WEBUI_DEFAULT_TASK,
                )

                # File handling
                save_transcriptions, temp_uploads_dir = _create_file_handling_ui()

                submit_btn = gr.Button("Transcribe", variant="primary")

            with gr.Column(scale=3):
                # Outputs
                with gr.Tabs():
                    with gr.TabItem("Transcription / Summary"):
                        transcription_output = gr.Textbox(
                            label="Transcription Result / Processing Summary",
                            lines=15,
                            interactive=False,
                        )
                    with gr.TabItem("Raw JSON / Details"):
                        json_output = gr.JSON(
                            label="Raw JSON Output / Detailed Results", visible=True
                        )

                generated_files = gr.Textbox(
                    label="Generated Files",
                    lines=6,
                    visible=False,
                    interactive=False,
                )

                with gr.Row():
                    download_txt_btn = gr.DownloadButton(
                        "Download TXT", visible=False, interactive=False
                    )
                    download_srt_btn = gr.DownloadButton(
                        "Download SRT", visible=False, interactive=False
                    )
                    download_json_btn = gr.DownloadButton(
                        "Download JSON", visible=False, interactive=False
                    )
                download_zip_btn = gr.DownloadButton(
                    "Download All as ZIP", visible=False, interactive=False
                )

        # Event handling
        stabilize_opt.change(
            fn=_toggle_stabilization_advanced,
            inputs=[stabilize_opt],
            outputs=[advanced_stabilization],
        )
        submit_btn.click(
            fn=_process_transcription_request_wrapper,
            inputs=[
                audio_input,
                model,
                device,
                batch_size,
                timestamp_type,
                language,
                task,
                dtype,
                chunk_length,
                # Stabilization options (match wrapper order)
                stabilize_opt,
                demucs_opt,
                vad_opt,
                subtitle_sync_opt,
                vad_threshold_opt,
                suppress_ts_tokens_opt,
                gap_padding_opt,
                adjust_gaps_opt,
                nonspeech_skip_opt,
                save_transcriptions,
                temp_uploads_dir,
            ],
            outputs=[
                transcription_output,
                json_output,
                generated_files,
                download_zip_btn,
                download_txt_btn,
                download_srt_btn,
                download_json_btn,
            ],
            api_name="transcribe_audio_v2",
        )

    demo.queue(default_concurrency_limit=1, max_size=8, api_open=True)
    return demo
