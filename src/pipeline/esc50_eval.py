"""Reproducible clip-level evaluation of the existing YAMNet event settings."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import wave
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from src.models.sed import YAMNetSED
from src.pipeline.media_import import SAMPLE_RATE, WINDOW_SAMPLES
from src.pipeline.sed_confirmation import SEDConfirmation
from src.utils.config import load_config

# ESC-50 has no doorbell or ordinary speech category.
POSITIVE = {
    "door_wood_knock": "Knock",
    "clock_alarm": "Alarm",
    "siren": "Siren",
}
NEGATIVE = (
    "keyboard_typing", "door_wood_creaks", "clock_tick", "church_bells",
    "car_horn", "glass_breaking", "footsteps", "mouse_click",
    "vacuum_cleaner", "rain",
)


def read_wav(data: bytes) -> np.ndarray:
    """Decode ESC-50 PCM WAV and resample to the model's 16 kHz input."""
    with wave.open(io.BytesIO(data), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("Expected mono 16-bit PCM WAV")
        sample_rate = wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    divisor = math.gcd(sample_rate, SAMPLE_RATE)
    return resample_poly(samples, SAMPLE_RATE // divisor, sample_rate // divisor).astype(np.float32)


def detect_clip(model: YAMNetSED, audio: np.ndarray, interval_sec: float, confirm_windows: int) -> list[dict]:
    gate = SEDConfirmation(confirm_windows)
    stride = max(1, round(interval_sec * SAMPLE_RATE))
    found = []
    for start in range(0, len(audio) - WINDOW_SAMPLES + 1, stride):
        result = model.infer(audio[start : start + WINDOW_SAMPLES])
        if gate.update(result):
            found.append({
                "label": result.label,
                "confidence": round(result.confidence, 4),
                "offset_sec": round((start + WINDOW_SAMPLES / 2) / SAMPLE_RATE, 3),
            })
    return found


def evaluate_archive(archive: Path, config: dict, fold: int = 1, samples_per_class: int = 5) -> dict:
    if fold not in range(1, 6) or samples_per_class < 1 or samples_per_class > 8:
        raise ValueError("fold must be 1..5 and samples_per_class must be 1..8")
    sed_cfg = config["sed"]
    if not sed_cfg["enabled"] or sed_cfg.get("interval_sec", 0) <= 0:
        raise ValueError("SED must be enabled with a positive interval_sec")
    if not Path(sed_cfg["model_path"]).is_file():
        raise FileNotFoundError(f"YAMNet model not found: {sed_cfg['model_path']}")
    model = YAMNetSED(
        model_path=sed_cfg["model_path"], labels_path=sed_cfg["labels_path"],
        threshold=sed_cfg["threshold"], target_classes=sed_cfg.get("target_classes", []),
        backend=config["inference"]["backend"], use_gpu=config["inference"]["use_gpu"],
    )
    rows = []
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        meta_names = [name for name in names if name.endswith("/meta/esc50.csv")]
        if len(meta_names) != 1:
            raise ValueError("Expected one ESC-50 meta/esc50.csv in archive")
        prefix = meta_names[0].removesuffix("meta/esc50.csv")
        metadata = list(csv.DictReader(io.StringIO(zipped.read(meta_names[0]).decode("utf-8-sig"))))
        categories = set(POSITIVE) | set(NEGATIVE)
        selected = {}
        for row in sorted(metadata, key=lambda item: item["filename"]):
            if int(row["fold"]) != fold or row["category"] not in categories:
                continue
            selected.setdefault(row["category"], [])
            if len(selected[row["category"]]) < samples_per_class:
                selected[row["category"]].append(row)
        if any(len(selected.get(category, [])) != samples_per_class for category in categories):
            raise ValueError("Archive does not contain the requested ESC-50 fold/categories")
        model.load()
        try:
            for category in (*POSITIVE, *NEGATIVE):
                for row in selected[category]:
                    audio = read_wav(zipped.read(prefix + "audio/" + row["filename"]))
                    detections = detect_clip(
                        model, audio, sed_cfg["interval_sec"], sed_cfg.get("confirm_windows", 1)
                    )
                    expected = POSITIVE.get(category)
                    labels = {item["label"] for item in detections}
                    rows.append({
                        "filename": row["filename"], "category": category,
                        "expected": expected, "detected": detections,
                        "correct_target_detected": expected in labels if expected else None,
                        "any_target_detected": bool(labels),
                    })
        finally:
            model.unload()
    by_category = {}
    for category in (*POSITIVE, *NEGATIVE):
        group = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "clips": len(group),
            "detected_correct_target": sum(row["correct_target_detected"] is True for row in group),
            "clips_with_any_target": sum(row["any_target_detected"] for row in group),
        }
    positives = [row for row in rows if row["expected"]]
    negatives = [row for row in rows if not row["expected"]]
    return {
        "dataset": "ESC-50", "fold": fold, "samples_per_class": samples_per_class,
        "settings": {"threshold": sed_cfg["threshold"], "confirm_windows": sed_cfg.get("confirm_windows", 1),
                     "interval_sec": sed_cfg["interval_sec"], "target_classes": sed_cfg.get("target_classes", [])},
        "summary": {
            "positive_clips": len(positives),
            "correct_target_detected": sum(row["correct_target_detected"] for row in positives),
            "negative_clips": len(negatives),
            "negative_clips_with_target": sum(row["any_target_detected"] for row in negatives),
            "detected_label_counts": dict(Counter(item["label"] for row in rows for item in row["detected"])),
        },
        "by_category": by_category, "files": rows,
        "limitations": [
            "Clip-level presence only; ESC-50 has no event timestamps, "
            "so localization and event-level recall are not measured.",
            "ESC-50 has no doorbell or ordinary speech class; those target/control conditions remain untested.",
            "AudioSet pretraining may overlap this dataset; "
            "results are not an independent deployment accuracy estimate.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate existing YAMNet settings on ESC-50 directly from ZIP")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/windows_sound_events.yaml"))
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate_archive(args.archive, load_config(args.config), args.fold, args.samples_per_class)
    output = args.output or Path("reports") / f"esc50_fold{args.fold}_{datetime.now():%Y%m%d_%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "by_category": report["by_category"],
                      "report": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
