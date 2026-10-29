import io
import wave
import zipfile

import numpy as np
import pytest

from src.models.base import InferenceResult
from src.pipeline import esc50_eval
from src.utils.config import load_config


def wav_bytes(seconds=2, rate=44100):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.zeros(seconds * rate, dtype="<i2").tobytes())
    return buffer.getvalue()


def test_wav_resampling_and_confirmation():
    audio = esc50_eval.read_wav(wav_bytes())
    assert audio.dtype == np.float32 and len(audio) == 32000

    class Model:
        labels = iter(["Knock", "Knock", "Silence"])

        def infer(self, chunk):
            return InferenceResult(task="sed", label=next(self.labels), confidence=0.8)

    found = esc50_eval.detect_clip(Model(), np.zeros(32000, dtype=np.float32), 0.5, 2)
    assert found == [{"label": "Knock", "confidence": 0.8, "offset_sec": 1.0}]


def test_archive_selection_and_metrics(tmp_path, monkeypatch):
    archive = tmp_path / "ESC-50-master.zip"
    csv_data = "filename,fold,target,category,esc10,src_file,take\n"
    with zipfile.ZipFile(archive, "w") as zipped:
        for idx, category in enumerate((*esc50_eval.POSITIVE, *esc50_eval.NEGATIVE)):
            filename = f"1-example-{idx}.wav"
            csv_data += f"{filename},1,{idx},{category},False,example,A\n"
            zipped.writestr("ESC-50-master/audio/" + filename, wav_bytes())
        zipped.writestr("ESC-50-master/meta/esc50.csv", csv_data)
    config = load_config("configs/windows_sound_events.yaml")
    config["sed"]["model_path"] = str(archive)
    monkeypatch.setattr(esc50_eval.YAMNetSED, "load", lambda self: None)
    monkeypatch.setattr(esc50_eval.YAMNetSED, "unload", lambda self: None)
    monkeypatch.setattr(esc50_eval, "detect_clip", lambda model, audio, interval_sec, confirm_windows: [])
    report = esc50_eval.evaluate_archive(archive, config, samples_per_class=1)
    assert report["summary"]["positive_clips"] == 3
    assert report["summary"]["negative_clips"] == 10
    assert report["summary"]["correct_target_detected"] == 0
    assert len(report["files"]) == 13
    with pytest.raises(ValueError, match="samples_per_class"):
        esc50_eval.evaluate_archive(archive, config, samples_per_class=9)
