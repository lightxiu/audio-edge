"""Explore ESC-50 errors with one model pass and replayed decision rules."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from src.models.sed import YAMNetSED
from src.pipeline.esc50_eval import NEGATIVE, POSITIVE, read_wav
from src.pipeline.media_import import SAMPLE_RATE, WINDOW_SAMPLES
from src.utils.config import load_config


def replay(scores: list[np.ndarray], labels: dict[int, str], targets: set[str], threshold: float,
           confirm_windows: int) -> set[str]:
    """Apply the production global-top/threshold/confirmation rules to cached scores."""
    previous = None
    consecutive = 0
    detected = set()
    for row in scores:
        idx = int(np.argmax(row))
        label = labels[idx]
        current = label if label in targets and float(row[idx]) >= threshold else None
        consecutive = consecutive + 1 if current and current == previous else (1 if current else 0)
        previous = current
        if current and consecutive >= confirm_windows:
            detected.add(current)
    return detected


def compare_archive(archive: Path, config: dict) -> dict:
    sed = config["sed"]
    if not Path(sed["model_path"]).is_file():
        raise FileNotFoundError(sed["model_path"])
    model = YAMNetSED(
        model_path=sed["model_path"], labels_path=sed["labels_path"], threshold=sed["threshold"],
        target_classes=sed["target_classes"], backend=config["inference"]["backend"],
        use_gpu=config["inference"]["use_gpu"],
    )
    baseline_targets = set(sed["target_classes"]) - {"Alarm clock"}
    variants = {
        "original_four_class_2_windows": (baseline_targets, 2),
        "add_alarm_clock_2_windows": (baseline_targets | {"Alarm clock"}, 2),
        "add_alarm_clock_1_window": (baseline_targets | {"Alarm clock"}, 1),
    }
    results = []
    with zipfile.ZipFile(archive) as zipped:
        meta = [name for name in zipped.namelist() if name.endswith("/meta/esc50.csv")]
        if len(meta) != 1:
            raise ValueError("Expected ESC-50 metadata")
        prefix = meta[0].removesuffix("meta/esc50.csv")
        rows = list(csv.DictReader(io.StringIO(zipped.read(meta[0]).decode("utf-8-sig"))))
        rows = sorted((row for row in rows if row["category"] in set(POSITIVE) | set(NEGATIVE)),
                      key=lambda row: row["filename"])
        stride = max(1, round(sed["interval_sec"] * SAMPLE_RATE))
        model.load()
        try:
            for row in rows:
                audio = read_wav(zipped.read(prefix + "audio/" + row["filename"]))
                scores = [model.class_scores(audio[start : start + WINDOW_SAMPLES])
                          for start in range(0, len(audio) - WINDOW_SAMPLES + 1, stride)]
                counts = Counter(model._labels[int(np.argmax(item))] for item in scores)
                decisions = {name: sorted(replay(scores, model._labels, targets, sed["threshold"], windows))
                             for name, (targets, windows) in variants.items()}
                results.append({"filename": row["filename"], "category": row["category"],
                                "top_class_windows": dict(counts.most_common(5)), "detected": decisions})
        finally:
            model.unload()
    by_variant = {}
    for name in variants:
        by_category = {}
        for category in (*POSITIVE, *NEGATIVE):
            group = [row for row in results if row["category"] == category]
            expected = {POSITIVE[category]} if category in POSITIVE else set()
            if category == "clock_alarm" and name != "original_four_class_2_windows":
                expected.add("Alarm clock")
            by_category[category] = {
                "clips": len(group),
                "correct": sum(bool(set(row["detected"][name]) & expected) for row in group) if expected else None,
                "any_target": sum(bool(row["detected"][name]) for row in group),
            }
        by_variant[name] = by_category
    missed_alarm = [row for row in results if row["category"] == "clock_alarm"
                    and not row["detected"]["original_four_class_2_windows"]]
    return {
        "settings": {"threshold": sed["threshold"], "interval_sec": sed["interval_sec"],
                     "confirm_windows": sed.get("confirm_windows", 1), "target_classes": sed["target_classes"]},
        "by_variant": by_variant,
        "missed_alarm_top_class_windows": dict(Counter({
            label: sum(row["top_class_windows"].get(label, 0) for row in missed_alarm)
            for row in missed_alarm for label in row["top_class_windows"]
        }).most_common(15)),
        "files": results,
        "note": "Exploratory comparison on previously inspected ESC-50 clips; not held-out validation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose existing ESC-50 candidate rules")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/windows_sound_events.yaml"))
    args = parser.parse_args()
    result = compare_archive(args.archive, load_config(args.config))
    dest = Path("reports") / f"esc50_diagnose_{datetime.now():%Y%m%d_%H%M%S}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"by_variant": result["by_variant"], "missed_alarm_top_class_windows":
                      result["missed_alarm_top_class_windows"], "report": str(dest)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
