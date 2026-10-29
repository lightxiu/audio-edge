import numpy as np

from src.pipeline.esc50_diagnose import replay
from src.utils.config import load_config


def test_replay_same_global_top_with_threshold_and_confirmation():
    labels = {0: "Speech", 1: "Alarm", 2: "Alarm clock"}
    scores = [
        np.array([0.8, 0.5, 0.2]),
        np.array([0.1, 0.2, 0.7]),
        np.array([0.1, 0.2, 0.8]),
    ]
    assert replay(scores, labels, {"Alarm"}, 0.25, 1) == set()
    assert replay(scores, labels, {"Alarm", "Alarm clock"}, 0.25, 2) == {"Alarm clock"}
    assert replay(scores[:2], labels, {"Alarm clock"}, 0.25, 2) == set()


def test_replay_rejects_below_threshold_and_resets():
    scores = [np.array([0.1, 0.3]), np.array([0.4, 0.2]), np.array([0.1, 0.5])]
    assert replay(scores, {0: "Speech", 1: "Knock"}, {"Knock"}, 0.25, 2) == set()


def test_windows_event_config_retains_confirmation():
    config = load_config("configs/windows_sound_events.yaml")
    assert "Alarm clock" in config["sed"]["target_classes"]
    assert config["sed"]["confirm_windows"] == 2
