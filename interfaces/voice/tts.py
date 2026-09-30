"""
Text-to-Speech wrapper — offline synthesis via Piper TTS.

Provides a thin interface over the Piper TTS engine for fully offline
voice output.  Complements the STT module for a complete hands-free
operator experience.

Selected differentiator: Operator Experience.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def _is_piper_available() -> bool:
    """Check if piper CLI is on PATH."""
    return shutil.which("piper") is not None


def _is_piper_python_available() -> bool:
    """Check if piper-tts Python package is installed."""
    try:
        import piper  # noqa: F401
        return True
    except ImportError:
        return False


class TextToSpeech:
    """Offline text-to-speech using Piper TTS.

    Supports two backends:
    1. Piper CLI (``piper`` binary on PATH).
    2. Piper Python package (``piper-tts``).

    Args:
        model_path:  Path to a Piper ONNX voice model.
        voice:       Voice name (used to auto-download if model_path is None).
        output_dir:  Directory for generated audio files.
        sample_rate: Output sample rate in Hz.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        voice: str = "en_US-lessac-medium",
        output_dir: str | Path | None = None,
        sample_rate: int = 22050,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.voice = voice
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        self.sample_rate = sample_rate
        self._backend: str | None = None

        self._detect_backend()

    def _detect_backend(self) -> None:
        """Detect available TTS backend."""
        if _is_piper_available():
            self._backend = "cli"
            logger.info("tts_backend_cli")
        elif _is_piper_python_available():
            self._backend = "python"
            logger.info("tts_backend_python")
        else:
            logger.warning(
                "tts_no_backend",
                hint="Install Piper CLI or pip install piper-tts",
            )

    def speak(
        self,
        text: str,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Synthesise speech from text.

        Args:
            text:        Text to speak.
            output_path: Where to save the WAV file. Auto-generated if None.

        Returns:
            Dict with ``audio_path``, ``text``, ``duration_estimate``,
            ``backend``.
        """
        if not text.strip():
            return {
                "audio_path": "",
                "text": "",
                "duration_estimate": 0.0,
                "backend": "none",
            }

        if output_path is None:
            output_path = self.output_dir / f"aetheris_tts_{hash(text) & 0xFFFFFF:06x}.wav"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._backend == "cli":
            return self._speak_cli(text, output_path)
        elif self._backend == "python":
            return self._speak_python(text, output_path)
        else:
            return self._fallback(text, output_path)

    def _speak_cli(self, text: str, output_path: Path) -> dict[str, Any]:
        """Synthesise via Piper CLI."""
        cmd = ["piper", "--output_file", str(output_path)]

        if self.model_path:
            cmd.extend(["--model", str(self.model_path)])
        else:
            cmd.extend(["--model", self.voice])

        try:
            result = subprocess.run(
                cmd,
                input=text,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.error("tts_cli_failed", stderr=result.stderr[:200])
                return self._fallback(text, output_path)

            # Rough duration estimate (150 words/min ≈ 2.5 words/sec)
            duration = len(text.split()) / 2.5

            logger.info("tts_synthesised", path=str(output_path), words=len(text.split()))
            return {
                "audio_path": str(output_path),
                "text": text,
                "duration_estimate": round(duration, 2),
                "backend": "cli",
            }
        except Exception as exc:
            logger.error("tts_cli_error", error=str(exc))
            return self._fallback(text, output_path)

    def _speak_python(self, text: str, output_path: Path) -> dict[str, Any]:
        """Synthesise via Piper Python package."""
        try:
            import wave

            from piper import PiperVoice

            if self.model_path:
                voice = PiperVoice.load(str(self.model_path))
            else:
                # Use built-in voice downloading
                voice = PiperVoice.load(self.voice)

            with wave.open(str(output_path), "wb") as wav_file:
                voice.synthesize(text, wav_file)

            duration = len(text.split()) / 2.5

            logger.info("tts_synthesised", path=str(output_path))
            return {
                "audio_path": str(output_path),
                "text": text,
                "duration_estimate": round(duration, 2),
                "backend": "python",
            }
        except Exception as exc:
            logger.error("tts_python_error", error=str(exc))
            return self._fallback(text, output_path)

    @staticmethod
    def _fallback(text: str, output_path: Path) -> dict[str, Any]:
        """Return metadata without synthesising audio."""
        return {
            "audio_path": "",
            "text": text,
            "duration_estimate": len(text.split()) / 2.5,
            "backend": "unavailable",
        }

    @property
    def is_available(self) -> bool:
        return self._backend is not None
