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
            "start": min(timestamps) if timestamps else None,
            "end": max(timestamps) if timestamps else None,
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
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.output_path = Path(output_path)
        self.timeout_sec = timeout_sec

    def generate(self, events: Iterable[dict[str, Any]]) -> ReportResult:
        """Generate and persist a report without raising on LLM failures."""
        stats = summarize_events(events)
        fallback = self._fallback(stats)
        source = "template"

        if stats["event_count"]:
            try:
                text = self._call_ollama(stats)
                if text.strip():
                    source = "ollama"
                else:
                    text = fallback
            except Exception as exc:
                logger.warning("Ollama report unavailable (%s); using template report", exc)
                text = fallback
        else:
            text = fallback

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(text.rstrip() + "\n", encoding="utf-8")
        logger.info("Inspection report written to %s (source=%s)", self.output_path, source)
        return ReportResult(text=text, source=source, stats=stats)

    def _call_ollama(self, stats: dict[str, Any]) -> str:
        import requests

        prompt = (
            "你是边缘设备音频监测系统的报告模块。请只根据给定 JSON 写一份简洁的中文巡检报告。\n"
            "必须包含：监测概况、主要事件、风险判断、建议动作；没有证据的内容写‘暂无证据’，不要编造时间、地点或原因。\n"
            f"JSON:\n{json.dumps(stats, ensure_ascii=False, indent=2)}"
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
        if not stats["event_count"]:
            return "# 音频巡检报告\n\n## 监测概况\n本次监测未产生可用事件记录。\n"
        labels = "、".join(f"{item['label']}（{item['count']}次）" for item in stats["top_labels"])
        confidence = stats["mean_confidence"]
        confidence_text = f"平均置信度 {confidence:.3f}" if confidence is not None else "置信度未知"
        return (
            "# 音频巡检报告\n\n"
            f"## 监测概况\n共记录 {stats['event_count']} 条事件，{confidence_text}。\n\n"
            f"## 主要事件\n{labels or '暂无可识别事件'}。\n\n"
            "## 风险判断\n当前报告基于模型输出统计，需结合现场情况复核；暂无额外风险证据。\n\n"
            "## 建议动作\n对高频或高置信度事件进行现场复核，并根据业务阈值配置告警策略。\n"
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
