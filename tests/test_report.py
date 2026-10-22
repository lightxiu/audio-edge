from pathlib import Path

from agent.report import EventReportGenerator, summarize_events


def test_summarize_events_counts_tasks_and_labels():
    stats = summarize_events([
        {"task": "sed", "label": "Dog", "confidence": 0.8, "timestamp": 2},
        {"task": "sed", "label": "Dog", "confidence": 0.6, "timestamp": 3},
        {"task": "asc", "label": "Park", "confidence": 0.9, "timestamp": 4},
    ])

    assert stats["event_count"] == 3
    assert stats["task_counts"] == {"asc": 1, "sed": 2}
    assert stats["events_by_task"]["sed"]["Dog"] == 2
    assert stats["mean_confidence"] == 0.767


def test_report_uses_deterministic_fallback_when_no_events(tmp_path: Path):
    result = EventReportGenerator(output_path=tmp_path / "report.md").generate([])

    assert result.source == "template"
    assert "未产生可用事件" in result.text
    assert (tmp_path / "report.md").exists()


def test_report_falls_back_when_ollama_is_unavailable(tmp_path: Path):
    result = EventReportGenerator(
        base_url="http://localhost:1",
        output_path=tmp_path / "report.md",
        timeout_sec=0.1,
    ).generate([{"task": "sed", "label": "Dog", "confidence": 0.8, "timestamp": 2}])

    assert result.source == "template"
    assert "Dog" in result.text
