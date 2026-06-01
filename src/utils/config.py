"""Configuration management using OmegaConf / YAML."""

from pathlib import Path
from typing import Any

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


# Default configuration values
DEFAULT_CONFIG: dict[str, Any] = {
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "block_size": 480,  # 30ms @ 16kHz
        "buffer_duration_sec": 3.0,
        "device": None,  # None = system default
    },
    "vad": {
        "enabled": True,
        "model_path": "models/vad/silero_vad.onnx",
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 300,
        "speech_pad_ms": 200,
    },
    "kws": {
        "enabled": True,
        "model_path": "models/kws/sherpa-kws.onnx",
        "keywords_path": "models/kws/keywords.txt",
        "window_sec": 1.0,
        "stride_sec": 0.1,
        "cooldown_sec": 1.5,
    },
    "sed": {
        "enabled": True,
        "model_path": "models/sed/yamnet.onnx",
        "labels_path": "models/sed/yamnet_class_map.csv",
        "interval_sec": 1.0,
        "threshold": 0.3,
        "target_classes": [],  # Empty = all classes
    },
    "asc": {
        "enabled": True,
        "model_path": "models/asc/ast-finetuned.onnx",
        "labels_path": "models/asc/scene_labels.txt",
        "interval_sec": 2.0,
        "min_duration_sec": 2.0,  # hysteresis: only report if scene persists
    },
    "inference": {
        "backend": "onnx",  # "onnx" or "tensorrt"
        "trt_engine_dir": "models/engines",
        "num_workers": 3,
        "use_gpu": True,
    },
    "output": {
        "console": True,
        "jsonl_path": None,  # None = disabled
        "mqtt_broker": None,
        "mqtt_topic": "audio-edge/events",
    },
    "logging": {
        "level": "INFO",
        "format": "text",  # "text" or "json"
        "file": None,  # None = stderr only
    },
}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from a YAML file, merging with defaults.

    Args:
        config_path: Path to YAML config file. If None, returns defaults.

    Returns:
        Merged configuration dictionary.
    """
    config = _deep_copy(DEFAULT_CONFIG)

    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return config

        with open(config_path, encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}

        if user_config:
            _deep_merge(config, user_config)
            logger.info(f"Loaded config from {config_path}")

    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save configuration to a YAML file.

    Args:
        config: Configuration dictionary.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Saved config to {path}")


def _deep_copy(d: dict) -> dict:
    """Simple deep copy via YAML round-trip (handles nested dicts/lists cleanly)."""
    import copy

    return copy.deepcopy(d)


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
