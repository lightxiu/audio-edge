"""Model download and cache management.

Handles downloading pretrained ONNX models from their source URLs,
verifying file integrity (SHA256), and caching them locally.

Supported models:
  - Silero VAD (MIT)
  - sherpa-onnx KWS (Apache 2.0)
  - YAMNet (Apache 2.0)
  - AST-finetuned-audioset (MIT)
"""

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Default model directory
MODEL_DIR = Path("models")

# --- Model registry ---
# Each entry defines: local path, download URL, expected SHA256, license


@dataclass
class ModelInfo:
    """Metadata for a downloadable model."""

    name: str  # Human-readable name
    local_path: str  # Relative path under models/
    url: str  # Download URL
    sha256: Optional[str] = None  # Expected SHA256 (None = skip verification)
    description: str = ""
    license: str = ""


# Registry of all models used by audio-edge
MODEL_REGISTRY: list[ModelInfo] = [
    ModelInfo(
        name="Silero VAD v5",
        local_path="vad/silero_vad.onnx",
        url="https://github.com/snakers4/silero-vad/raw/v5.1/src/silero_vad/data/silero_vad.onnx",
        sha256=None,  # No stable hash from upstream
        description="Voice Activity Detection — speech/silence classifier",
        license="MIT",
    ),
    ModelInfo(
        name="YAMNet class map",
        local_path="sed/yamnet_class_map.csv",
        url="https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
        sha256=None,
        description="AudioSet class labels (521 classes) for YAMNet",
        license="Apache 2.0",
    ),
]

# Optional models (downloaded on demand)
OPTIONAL_MODELS: list[ModelInfo] = [
    ModelInfo(
        name="sherpa-onnx KWS",
        local_path="kws/sherpa-kws.onnx",
        url="",  # Filled at download time via sherpa-onnx API
        sha256=None,
        description="Keyword spotting model (Zipformer-based)",
        license="Apache 2.0",
    ),
    ModelInfo(
        name="AST-finetuned-audioset",
        local_path="asc/ast-finetuned.onnx",
        url="",  # Filled at download time or exported from HF
        sha256=None,
        description="Audio Scene Classification — 527-class AudioSet fine-tuned",
        license="MIT",
    ),
]


def download_model(info: ModelInfo, models_dir: str | Path = MODEL_DIR) -> Path:
    """Download a single model file if not already cached.

    Args:
        info: ModelInfo with URL and local path.
        models_dir: Root models directory.

    Returns:
        Local path to the downloaded model.

    Raises:
        RuntimeError: If download fails.
    """
    models_dir = Path(models_dir)
    local_path = models_dir / info.local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        logger.info(f"Model already cached: {local_path}")
        if info.sha256 and not _verify_sha256(local_path, info.sha256):
            logger.warning(f"SHA256 mismatch for {local_path}, re-downloading...")
            local_path.unlink()
        else:
            return local_path

    if not info.url:
        raise ValueError(
            f"No download URL for {info.name}. "
            f"This model must be obtained manually or via a specialized script."
        )

    logger.info(f"Downloading {info.name} ({info.license})...")
    logger.info(f"  URL: {info.url}")
    logger.info(f"  To: {local_path}")

    try:
        _download_with_progress(info.url, local_path)
    except Exception as e:
        # Clean up partial download
        if local_path.exists():
            local_path.unlink()
        raise RuntimeError(f"Failed to download {info.name}: {e}") from e

    # Verify
    if info.sha256:
        if not _verify_sha256(local_path, info.sha256):
            local_path.unlink()
            raise RuntimeError(f"SHA256 verification failed for {info.name}")

    file_size_mb = local_path.stat().st_size / (1024 * 1024)
    logger.info(f"Downloaded {info.name}: {file_size_mb:.1f} MB")
    return local_path


def download_all(models_dir: str | Path = MODEL_DIR) -> list[Path]:
    """Download all required models.

    Args:
        models_dir: Root models directory.

    Returns:
        List of downloaded file paths.
    """
    downloaded = []

    for info in MODEL_REGISTRY:
        try:
            path = download_model(info, models_dir)
            downloaded.append(path)
        except ValueError as e:
            logger.warning(f"Skipping {info.name}: {e}")
        except RuntimeError as e:
            logger.error(str(e))

    return downloaded


def get_model_path(
    model_name: str,
    models_dir: str | Path = MODEL_DIR,
) -> Path:
    """Get the local path for a registered model.

    Args:
        model_name: Subdirectory name (e.g., "vad", "kws", "sed", "asc").
        models_dir: Root models directory.

    Returns:
        Expected local path.
    """
    models_dir = Path(models_dir)
    return models_dir / model_name


def _download_with_progress(url: str, dest: Path) -> None:
    """Download a file with a simple progress indicator."""
    import sys

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        """Callback for urlretrieve to show progress."""
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 // total_size)
            bar_len = 30
            filled = bar_len * percent // 100
            bar = "█" * filled + "░" * (bar_len - filled)
            sys.stderr.write(f"\r  [{bar}] {percent:3d}%")
            sys.stderr.flush()

    urllib.request.urlretrieve(url, str(dest), _progress)
    sys.stderr.write("\n")
    sys.stderr.flush()


def _verify_sha256(path: Path, expected: str) -> bool:
    """Verify file SHA256 hash."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    actual = sha.hexdigest()
    return actual == expected
