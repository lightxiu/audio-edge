from datetime import datetime, timedelta, timezone
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
    expected = (datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=2)).astimezone()
    assert stats["time_range"]["start"] == expected.isoformat(timespec="seconds")


def test_report_uses_deterministic_fallback_when_no_events(tmp_path: Path):
    result = EventReportGenerator(output_path=tmp_path / "report.md", run_id="first").generate([])

    assert result.source == "template"
    assert "未产生可用事件" in result.text
    assert result.path == tmp_path / "report_first.md"
    assert result.path.exists()


def test_report_falls_back_when_ollama_is_unavailable(tmp_path: Path):
    result = EventReportGenerator(
        base_url="http://localhost:1",
        output_path=tmp_path / "report.md",
        timeout_sec=0.1,
        run_id="second",
    ).generate([{"task": "sed", "label": "Dog", "confidence": 0.8, "timestamp": 2}])

    assert result.source == "template"
    assert "Dog" in result.text
    assert result.path == tmp_path / "report_second.md"


def test_report_keeps_previous_runs_and_existing_base_file(tmp_path: Path):
    base = tmp_path / "report.md"
    base.write_text("legacy report", encoding="utf-8")
    generator = EventReportGenerator(output_path=base, run_id="same_run")

    first = generator.generate([])
    second = generator.generate([])

    assert base.read_text(encoding="utf-8") == "legacy report"
    assert first.path.name == "report_same_run.md"
    assert second.path.name == "report_same_run_2.md"
    assert first.path.read_text(encoding="utf-8") == second.path.read_text(encoding="utf-8")


def test_report_prompt_disallows_vad_risk_inference(monkeypatch, tmp_path: Path):
    captured = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "测试报告"}

    def post(url, json, timeout):
        captured.update(json)
        return Response()

    monkeypatch.setattr("requests.post", post)
    result = EventReportGenerator(output_path=tmp_path / "report.md", run_id="prompt").generate(
        [{"task": "sed", "label": "Doorbell", "confidence": 0.9, "timestamp": 1790000000},
         {"task": "vad", "label": "speech", "confidence": 0.9, "timestamp": 1790000001}]
    )

    assert result.source == "ollama"
    assert "状态变化次数" in captured["prompt"]
    assert "不能判断业务风险" in captured["prompt"]
    assert datetime.fromtimestamp(1790000000).astimezone().isoformat(timespec="seconds") in captured["prompt"]


def test_monitoring_duration_is_not_first_to_last_event_span(monkeypatch, tmp_path: Path):
    prompts = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "监测到语音事件，风险暂无证据。"}

    def post(url, json, timeout):
        prompts.append(json["prompt"])
        return Response()

    monkeypatch.setattr("requests.post", post)
    result = EventReportGenerator(output_path=tmp_path / "report.md").generate(
        [{"task": "sed", "label": "Doorbell", "confidence": .8, "timestamp": 1790000010},
         {"task": "vad", "label": "silence", "confidence": .1, "timestamp": 1790000020}],
        started_at=1790000000,
        ended_at=1790000030,
    )

    assert result.stats["monitoring"]["duration_sec"] == 30
    assert result.stats["time_range"]["start"] != result.stats["monitoring"]["start"]
    assert '"duration_sec": 30' in prompts[0]
    assert "不能当作巡检时长" in prompts[0]


def test_report_rejects_unsupported_accuracy_claim(monkeypatch, tmp_path: Path):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "没有出现误判或漏判，无需调整。"}

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: Response())
    result = EventReportGenerator(output_path=tmp_path / "report.md").generate(
        [{"task": "sed", "label": "Doorbell", "confidence": .8, "timestamp": 1790000010}]
    )
    assert result.source == "template"
    assert "不能判断业务风险" in result.text


def test_vad_only_report_stays_factual_without_ollama(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("called")))
    result = EventReportGenerator(output_path=tmp_path / "report.md").generate(
        [{"task": "vad", "label": "speech", "confidence": .8, "timestamp": 1790000010}],
        started_at=1790000000,
        ended_at=1790000030,
    )
    assert result.source == "template"
    assert "30 秒" in result.text
    assert "1 条事件" in result.text
