"""Generate a local-LLM inspection report from emitted audio events.

The report path is intentionally small: structured aggregation happens in
Python, while Ollama is used only for wording and recommendations.  If the
local model is unavailable, a deterministic template still produces a useful
report and never interrupts the inference pipeline.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ReportResult:
    """Generated report and the source used to produce it."""

    text: str
    source: str
    stats: dict[str, Any]
    path: Path


def summarize_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Reduce raw emitted events to compact, prompt-friendly statistics."""
    rows = list(events)
    by_task: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "labels": {}})
    label_counts: Counter[str] = Counter()
    confidences: list[float] = []
    timestamps: list[float] = []

    for event in rows:
        task = str(event.get("task", "unknown"))
        label = str(event.get("label", "unknown"))
        by_task[task]["count"] += 1
        by_task[task]["labels"][label] = by_task[task]["labels"].get(label, 0) + 1
        label_counts[label] += 1
        if event.get("confidence") is not None:
            confidences.append(float(event["confidence"]))
        if event.get("timestamp") is not None:
            timestamps.append(float(event["timestamp"]))

    def local_time(timestamp: float) -> str:
        instant = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=timestamp)
        return instant.astimezone().isoformat(timespec="seconds")

    return {
        "event_count": len(rows),
        "task_counts": {task: data["count"] for task, data in sorted(by_task.items())},
        "events_by_task": {task: data["labels"] for task, data in sorted(by_task.items())},
        "top_labels": [
            {"label": label, "count": count}
            for label, count in label_counts.most_common(10)
        ],
        "mean_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
        "time_range": {
            "start": local_time(min(timestamps)) if timestamps else None,
            "end": local_time(max(timestamps)) if timestamps else None,
        },
    }


class EventReportGenerator:
    """Generate an event report with Ollama and a local fallback."""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        output_path: str | Path = "reports/latest.md",
        timeout_sec: float = 45.0,
        run_id: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.output_path = Path(output_path)
        self.timeout_sec = timeout_sec
        self.run_id = run_id or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")

    def generate(
        self,
        events: Iterable[dict[str, Any]],
        started_at: float | None = None,
        ended_at: float | None = None,
    ) -> ReportResult:
        """Generate and persist a report without raising on LLM failures."""
        stats = summarize_events(events)
        if started_at is not None and ended_at is not None:
            stats["monitoring"] = {
                "start": datetime.fromtimestamp(started_at, timezone.utc).astimezone().isoformat(timespec="seconds"),
                "end": datetime.fromtimestamp(ended_at, timezone.utc).astimezone().isoformat(timespec="seconds"),
                "duration_sec": round(max(0.0, ended_at - started_at), 1),
            }
        fallback = self._fallback(stats)
        source = "template"

        if stats["event_count"] and set(stats["task_counts"]) != {"vad"}:
            try:
                text = self._call_ollama(stats)
                if text.strip():
                    if self._has_unsupported_claims(text):
                        logger.warning("Ollama report made unsupported accuracy claims; using template report")
                        text = fallback
                    else:
                        source = "ollama"
                else:
                    text = fallback
            except Exception as exc:
                logger.warning("Ollama report unavailable ({}); using template report", exc)
                text = fallback
        else:
            text = fallback

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        stem = self.output_path.stem
        suffix = self.output_path.suffix or ".md"
        for index in range(1, 1001):
            extra = f"_{index}" if index > 1 else ""
            path = self.output_path.with_name(f"{stem}_{self.run_id}{extra}{suffix}")
            try:
                with path.open("x", encoding="utf-8") as report_file:
                    report_file.write(text.rstrip() + "\n")
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError(f"No available report filename for {self.output_path}")
        logger.info("Inspection report written to {} (source={})", path, source)
        return ReportResult(text=text, source=source, stats=stats, path=path)

    @staticmethod
    def _has_unsupported_claims(text: str) -> bool:
        import re

        unsupported = r"没有.{0,6}(误判|漏判)|能够准确|准确地|准确率.{0,8}(高|良好)|无需.{0,8}(调整|优化)"
        return bool(re.search(unsupported, text))

    def _call_ollama(self, stats: dict[str, Any]) -> str:
        import requests

        prompt_stats = dict(stats)
        if set(stats["task_counts"]) == {"vad"}:
            prompt_stats.pop("mean_confidence", None)

        prompt = (
            "你是边缘设备音频监测系统的报告模块。请只根据给定 JSON 写一份简洁的中文巡检报告。\n"
            "必须包含：监测概况、主要事件、风险判断、建议动作。\n"
            "monitoring 才是完整采集的起止与时长；time_range 只是首末事件时间，不能当作巡检时长。\n"
            "时间是带时区的实际时间，只能照录，不能改写或猜测。\n"
            "事件计数是状态变化次数，不是任务执行次数；VAD 的 confidence 是语音概率，silence 的低概率不是低准确率。\n"
            "没有标注真值不能推断误判率或准确率；仅有 VAD 事件不能判断业务风险，也不要建议调高阈值或重训模型。\n"
            "没有证据的地点、原因、风险和处置动作写‘暂无证据’，不要编造。使用正常 Markdown，不要套用多余的加粗符号。\n"
            f"JSON:\n{json.dumps(prompt_stats, ensure_ascii=False, indent=2)}"
        )
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        return str(response.json().get("response", ""))

    @staticmethod
    def _fallback(stats: dict[str, Any]) -> str:
        monitoring = stats.get("monitoring")
        duration = (
            f"采集时间 {monitoring['start']} 至 {monitoring['end']}，共 {monitoring['duration_sec']} 秒；"
            if monitoring else ""
        )
        if not stats["event_count"]:
            return f"# 音频巡检报告\n\n## 监测概况\n{duration}本次监测未产生可用事件记录。\n"
        labels = "、".join(f"{item['label']}（{item['count']}次）" for item in stats["top_labels"])
        return (
            "# 音频巡检报告\n\n"
            f"## 监测概况\n{duration}共记录 {stats['event_count']} 条事件。\n\n"
            f"## 主要事件\n{labels or '暂无可识别事件'}。\n\n"
            "## 风险判断\n仅凭当前事件不能判断业务风险或模型准确率；暂无额外证据。\n\n"
            "## 建议动作\n结合现场情况复核事件，再确定是否需要后续处置。\n"
        )


def event_record(result: Any) -> dict[str, Any]:
    """Convert an InferenceResult-like object to a JSON-safe event record."""
    return {
        "task": result.task,
        "label": result.label,
        "confidence": round(float(result.confidence), 4),
        "timestamp": float(result.timestamp),
        "latency_ms": round(float(result.latency_ms), 2),
    }
