"""TTS (Text-to-Speech) via Piper TTS.

Piper TTS is the de facto standard for offline TTS on ARM64 devices.
It runs entirely on CPU with <300ms first-audio latency.

Piper binary must be installed on the system:
    wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz
    tar xzf piper_arm64.tar.gz -C /usr/local/bin

Voice models are downloaded separately (~65-150 MB each):
    https://huggingface.co/rhasspy/piper-voices
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class TextToSpeech:
    """Piper TTS wrapper.

    Parameters
    ----------
    piper_path : str
        Path to the ``piper`` binary.
    voice_model : str
        Path to a Piper voice model ``.onnx`` file.
    voice_config : str or None
        Path to the voice config ``.json`` (auto-detected if None).
    """

    def __init__(
        self,
        piper_path: str = "piper",
        voice_model: str = "models/tts/zh_CN-huayan-medium.onnx",
        voice_config: str | None = None,
    ) -> None:
        self._piper = piper_path
        self._voice_model = Path(voice_model)
        self._voice_config = voice_config or str(self._voice_model.with_suffix(".json"))
        self._use_mock = not Path(piper_path).exists() and piper_path != "piper"
        if not self._voice_model.exists():
            self._use_mock = True

    def speak(self, text: str) -> np.ndarray | None:
        """Convert text to speech audio.

        Parameters
        ----------
        text : str
            Text to synthesize (Chinese or English).

        Returns
        -------
        np.ndarray or None
            1-D float32 audio samples at 22050 Hz, or None on failure.
        """
        if self._use_mock:
            return self._mock_speak(text)

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name

            cmd = [
                self._piper,
                "--model", str(self._voice_model),
                "--config", self._voice_config,
                "--output_file", wav_path,
            ]
            proc = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=10,
            )

            if proc.returncode != 0:
                logger.error(f"Piper TTS failed: {proc.stderr.decode()}")
                return None

            # Read WAV output
            import wave
            with wave.open(wav_path, "rb") as wf:
                n_frames = wf.getnframes()
                audio = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
                audio = audio.astype(np.float32) / 32768.0

            Path(wav_path).unlink(missing_ok=True)
            return audio

        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

    def speak_to_file(self, text: str, output_path: str) -> bool:
        """Synthesize speech and save to a WAV file.

        Returns True on success.
        """
        try:
            cmd = [
                self._piper,
                "--model", str(self._voice_model),
                "--config", self._voice_config,
                "--output_file", output_path,
            ]
            proc = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=15,
            )
            return proc.returncode == 0
        except Exception as e:
            logger.error(f"TTS file write failed: {e}")
            return False

    @staticmethod
    def _mock_speak(text: str) -> np.ndarray:
        """Generate a short silence array as mock TTS output."""
        logger.info(f"[Mock TTS] would speak: {text[:80]}")
        # Return 0.5s of silence
        return np.zeros(int(22050 * 0.5), dtype=np.float32)
