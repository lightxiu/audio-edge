"""Offline audio inference and same-file video evidence extraction."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from src.models.sed import YAMNetSED
from src.output.review import ReviewStore
from src.pipeline.sed_confirmation import SEDConfirmation
from src.utils.config import load_config

SAMPLE_RATE = 16000
WINDOW_SAMPLES = SAMPLE_RATE
MAX_DURATION_SEC = 600


def recording_epoch(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("--recorded-at must include a timezone offset, e.g. 2026-10-29T14:00:00+08:00")
    return parsed.timestamp()


def _binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"{name} not found; install FFmpeg and put ffmpeg/ffprobe on PATH")
    return path


def probe_media(path: Path) -> dict:
    command = [_binary("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    response = subprocess.run(command, capture_output=True, check=True, timeout=30)
    data = json.loads(response.stdout)
    duration = float(data["format"]["duration"])
    if not math.isfinite(duration) or not 0 < duration <= MAX_DURATION_SEC:
        raise ValueError(f"Media duration must be between 0 and {MAX_DURATION_SEC} seconds")
    return data


def align_audio(samples: np.ndarray, audio_start: float, container_start: float) -> np.ndarray:
    """Align decoded samples with media time zero using stream start PTS."""
    offset = round((audio_start - container_start) * SAMPLE_RATE)
    if offset >= 0:
        return np.pad(samples, (offset, 0)) if offset else samples
    return samples[-offset:]


def decode_audio(path: Path, stream_index: int, duration: float) -> np.ndarray:
    command = [
        _binary("ffmpeg"), "-v", "error", "-nostdin", "-i", str(path),
        "-map", f"0:{stream_index}", "-t", str(duration), "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    response = subprocess.run(command, capture_output=True, check=True, timeout=180)
    return np.frombuffer(response.stdout, dtype="<f4").copy()


def extract_video(path: Path, stream_index: int, center_sec: float, duration: float, dest: Path) -> None:
    start = max(0.0, center_sec - 2.0)
    clip_duration = min(5.0, duration - start)
    command = [
        _binary("ffmpeg"), "-v", "error", "-nostdin", "-i", str(path),
        "-ss", str(start), "-t", str(clip_duration), "-map", f"0:{stream_index}", "-an",
        "-vf", "scale=640:360:force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-c:v", "mpeg4", "-q:v", "5", "-y", str(dest),
    ]
    subprocess.run(command, capture_output=True, check=True, timeout=120)


def import_media(path: Path, recorded_at: str, config: dict, source_id: str | None = None) -> dict:
    """Store SED candidates with absolute timestamps and optional same-file clips."""
    epoch = recording_epoch(recorded_at)
    path = Path(path).resolve(strict=True)
    media = probe_media(path)
    streams = media["streams"]
    audio_stream = next((stream for stream in streams if stream["codec_type"] == "audio"), None)
    video_stream = next((stream for stream in streams if stream["codec_type"] == "video"), None)
    if audio_stream is None:
        raise ValueError("No audio stream found; video-only C++ events can use python -m dashboard")
    sed_cfg = config["sed"]
    if not sed_cfg["enabled"]:
        raise ValueError("SED must be enabled for offline media import")
    review_cfg = config["review"]
    store = ReviewStore(review_cfg["db_path"], review_cfg["evidence_dir"])
    duration = float(media["format"]["duration"])
    container_start = float(media["format"].get("start_time") or 0)
    audio_start = float(audio_stream.get("start_time") or container_start)
    audio = align_audio(decode_audio(path, audio_stream["index"], duration), audio_start, container_start)
    model_path = Path(sed_cfg["model_path"])
    if not model_path.is_file():
        raise FileNotFoundError(f"YAMNet model not found: {model_path}")
    model = YAMNetSED(
        model_path=model_path, labels_path=sed_cfg["labels_path"],
        threshold=sed_cfg["threshold"], target_classes=sed_cfg.get("target_classes", []),
        backend=config["inference"]["backend"], use_gpu=config["inference"]["use_gpu"],
    )
    confirmation = SEDConfirmation(sed_cfg.get("confirm_windows", 1))
    stride = max(1, round(sed_cfg["interval_sec"] * SAMPLE_RATE))
    cooldown = sed_cfg.get("cooldown_sec", 3.0)
    last_emit: dict[str, float] = {}
    saved = []
    model.load()
    try:
        for start in range(0, len(audio) - WINDOW_SAMPLES + 1, stride):
            window = audio[start : start + WINDOW_SAMPLES]
            result = model.infer(window)
            center_sec = (start + WINDOW_SAMPLES / 2) / SAMPLE_RATE
            if center_sec > duration:
                break
            if not confirmation.update(result) or center_sec - last_emit.get(result.label, -math.inf) < cooldown:
                continue
            last_emit[result.label] = center_sec
            result.timestamp = epoch + center_sec
            candidate_id = store.add(result, window, SAMPLE_RATE)
            clip_available = False
            if video_stream is not None:
                clip_name = f"{candidate_id}.mp4"
                try:
                    extract_video(path, video_stream["index"], center_sec, duration, store.evidence_dir / clip_name)
                    store.add_media_evidence(candidate_id, source_id or path.stem, center_sec, clip_name)
                    clip_available = True
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    print(f"Video clip unavailable for {candidate_id}: {exc}")
            saved.append({"id": candidate_id, "label": result.label, "offset_sec": center_sec, "video": clip_available})
    finally:
        model.unload()
    return {"source_id": source_id or path.stem, "duration_sec": duration, "has_video": video_stream is not None,
            "candidates": saved}


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a local media recording without opening a microphone")
    parser.add_argument("media", type=Path)
    parser.add_argument(
        "--recorded-at", required=True,
        help="Media timeline zero with timezone, e.g. 2026-10-29T14:00:00+08:00",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/windows_sound_events.yaml"))
    parser.add_argument("--source-id", default=None)
    args = parser.parse_args()
    result = import_media(args.media, args.recorded_at, load_config(args.config), args.source_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
