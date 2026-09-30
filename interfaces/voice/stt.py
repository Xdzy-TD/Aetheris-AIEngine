"""
Speech-to-Text wrapper — offline transcription via faster-whisper.

Provides a thin interface over the ``faster-whisper`` library (a
CTranslate2 reimplementation of OpenAI Whisper) for fully offline
speech recognition.  This enables hands-free operator interaction
with the Aetheris system in the field.

Selected differentiator: Operator Experience.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def _is_faster_whisper_available() -> bool:
    try:
        from faster_whisper import WhisperModel  # noqa: F401
        return True
    except ImportError:
        return False


class SpeechToText:
    """Offline speech-to-text using faster-whisper.

    Args:
        model_size: Whisper model size (``"tiny"``, ``"base"``, ``"small"``,
                    ``"medium"``, ``"large-v3"``).
        device:     ``"cpu"`` or ``"cuda"``.
        compute_type: ``"int8"``, ``"float16"``, or ``"float32"``.
        language:   Language code (e.g. ``"en"``). None = auto-detect.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = "en",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def load(self) -> None:
        """Load the Whisper model."""
        if not _is_faster_whisper_available():
            logger.warning(
                "stt_faster_whisper_not_installed",
                hint="pip install faster-whisper",
            )
            return

        from faster_whisper import WhisperModel

        logger.info(
            "stt_loading_model",
            model=self.model_size,
            device=self.device,
        )
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("stt_model_loaded")

    def transcribe(
        self,
        audio_path: str | Path,
        beam_size: int = 5,
    ) -> dict[str, Any]:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to WAV/MP3/M4A audio file.
            beam_size:  Beam search width.

        Returns:
            Dict with ``text``, ``language``, ``segments``, ``duration``.
        """
        if self._model is None:
            self.load()

        if self._model is None:
            return self._fallback_response(audio_path)

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("stt_transcribing", path=str(audio_path))

        segments, info = self._model.transcribe(
            str(audio_path),
            beam_size=beam_size,
            language=self.language,
            vad_filter=True,
        )

        # Collect segments
        all_segments = []
        full_text_parts = []
        for seg in segments:
            all_segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "avg_logprob": round(seg.avg_logprob, 3),
            })
            full_text_parts.append(seg.text.strip())

        full_text = " ".join(full_text_parts)

        logger.info(
            "stt_transcription_complete",
            text_length=len(full_text),
            segments=len(all_segments),
            language=info.language,
        )

        return {
            "text": full_text,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration": round(info.duration, 2),
            "segments": all_segments,
        }

    @staticmethod
    def _fallback_response(audio_path: str | Path) -> dict[str, Any]:
        """Return a placeholder when the model isn't available."""
        return {
            "text": "[STT unavailable — install faster-whisper]",
            "language": "unknown",
            "language_probability": 0.0,
            "duration": 0.0,
            "segments": [],
        }

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
