"""Tests for pipeline orchestration components.

These tests use mock models to verify the orchestrator, scheduler,
and event aggregator work correctly without real hardware.
"""

import time

import numpy as np
import pytest

from src.models.base import InferenceResult
from src.models.kws import MockKWS
from src.pipeline.aggregator import AggregatorConfig, EventAggregator
from src.pipeline.scheduler import Scheduler

# --- Scheduler Tests ---


class TestScheduler:
    """Tests for the inference task scheduler."""

    def test_add_task(self):
        s = Scheduler()
        s.add_task("kws", interval_sec=0.1)
        assert "kws" in s.tasks
        assert s.tasks["kws"].interval_sec == 0.1

    def test_task_default_not_due(self):
        s = Scheduler()
        s.add_task("kws", interval_sec=1.0)
        # Task with 1s interval should not be due immediately
        # (it was just created, last_run = 0)
        assert s.is_due("kws")

    def test_task_due_after_interval(self):
        s = Scheduler()
        s.add_task("kws", interval_sec=0.01)
        s.mark_run("kws")
        assert not s.is_due("kws")
        time.sleep(0.02)
        assert s.is_due("kws")

    def test_speech_gating(self):
        s = Scheduler()
        s.add_task("kws", interval_sec=0.01, requires_speech=True)
        assert not s.is_due("kws", speech_active=False)
        assert s.is_due("kws", speech_active=True)

    def test_disabled_task(self):
        s = Scheduler()
        s.add_task("kws", interval_sec=0.01)
        s.disable("kws")
        assert not s.is_due("kws")

    def test_get_due_tasks(self):
        s = Scheduler()
        s.add_task("kws", interval_sec=0.01, requires_speech=True)
        s.add_task("sed", interval_sec=0.01, requires_speech=False)
        s.add_task("asc", interval_sec=0.01, requires_speech=False)

        # During speech
        due = s.get_due_tasks(speech_active=True)
        assert "kws" in due
        assert "sed" in due
        assert "asc" in due

        # During silence
        due = s.get_due_tasks(speech_active=False)
        assert "kws" not in due
        assert "sed" in due
        assert "asc" in due

    def test_reset(self):
        s = Scheduler()
        s.add_task("kws", interval_sec=0.01)
        s.mark_run("kws")
        assert not s.is_due("kws")
        s.reset()
        assert s.is_due("kws")

    def test_nonexistent_task(self):
        s = Scheduler()
        assert not s.is_due("nonexistent")


# --- Event Aggregator Tests ---


class TestEventAggregator:
    """Tests for event deduplication and throttling."""

    @pytest.fixture
    def agg(self):
        return EventAggregator(
            AggregatorConfig(
                kws_cooldown_sec=0.1,
                sed_cooldown_sec=0.2,
                asc_hysteresis_sec=0.3,
            )
        )

    def test_kws_first_fire(self, agg):
        """First keyword detection should pass through."""
        result = InferenceResult(task="kws", label="hey_computer", confidence=0.95)
        assert agg.should_emit(result)

    def test_kws_cooldown(self, agg):
        """Same keyword within cooldown should be suppressed."""
        result = InferenceResult(task="kws", label="hey_computer", confidence=0.95)
        agg.should_emit(result)  # First fire
        # Immediate repeat should be suppressed
        assert not agg.should_emit(result)

    def test_kws_different_keyword(self, agg):
        """Different keywords should not share cooldown."""
        r1 = InferenceResult(task="kws", label="hey_computer", confidence=0.95)
        r2 = InferenceResult(task="kws", label="stop", confidence=0.90)
        agg.should_emit(r1)
        assert agg.should_emit(r2)  # Different keyword, should pass

    def test_no_keyword_filtered(self, agg):
        """no_keyword results should always be suppressed."""
        result = InferenceResult(task="kws", label="no_keyword", confidence=0.1)
        assert not agg.should_emit(result)

    def test_vad_always_passes(self, agg):
        """VAD state changes should always pass through."""
        result = InferenceResult(task="vad", label="speech", confidence=0.9)
        assert agg.should_emit(result)

    def test_sed_cooldown(self, agg):
        """SED events should respect per-class cooldown."""
        r1 = InferenceResult(task="sed", label="siren", confidence=0.8)
        r2 = InferenceResult(task="sed", label="siren", confidence=0.85)
        assert agg.should_emit(r1)
        assert not agg.should_emit(r2)  # Within cooldown

    def test_asc_hysteresis(self, agg):
        """Scene change should require persistence."""
        r1 = InferenceResult(task="asc", label="outdoor", confidence=0.8)
        r2 = InferenceResult(task="asc", label="indoor", confidence=0.7)

        # First outdoor detection starts timer
        assert not agg.should_emit(r1)
        # Scene hasn't persisted
        assert not agg.should_emit(r2)

    def test_reset(self, agg):
        """Reset should clear all state."""
        result = InferenceResult(task="kws", label="test", confidence=0.9)
        agg.should_emit(result)
        assert not agg.should_emit(result)  # Cooldown active
        agg.reset()
        assert agg.should_emit(result)  # Cooldown cleared


# --- Mock KWS Tests ---


class TestMockKWS:
    """Tests for the mock keyword spotter."""

    def test_load(self):
        kws = MockKWS()
        kws.load()
        assert kws.is_loaded

    def test_infer_returns_result(self):
        kws = MockKWS()
        kws.load()
        audio = np.random.randn(16000).astype(np.float32)
        result = kws.infer(audio)
        assert isinstance(result, InferenceResult)
        assert result.task == "kws"
        assert result.label in ["no_keyword", "hey_computer", "stop", "go"]

    def test_sample_rate(self):
        kws = MockKWS()
        assert kws.sample_rate == 16000

    def test_custom_keywords(self):
        kws = MockKWS(keywords=["hello", "world"])
        kws.load()
        assert "hello" in kws.keywords
        assert "world" in kws.keywords
