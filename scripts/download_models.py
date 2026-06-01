#!/usr/bin/env python3
"""Download all required ONNX models for audio-edge.

Usage:
    python scripts/download_models.py              # Download all required models
    python scripts/download_models.py --list       # List available models
    python scripts/download_models.py --model vad  # Download a specific model
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_loader import (
    MODEL_DIR,
    MODEL_REGISTRY,
    OPTIONAL_MODELS,
    ModelInfo,
    download_model,
    download_all,
)
from src.utils.logging import setup_logging


def cmd_list() -> None:
    """List all registered models."""
    print("\n=== Required Models ===\n")
    for m in MODEL_REGISTRY:
        status = "✓" if (MODEL_DIR / m.local_path).exists() else "○"
        print(f"  {status} {m.name}")
        print(f"    Path:     {m.local_path}")
        print(f"    License:  {m.license}")
        print(f"    URL:      {m.url}")
        print()

    print("=== Optional Models ===\n")
    for m in OPTIONAL_MODELS:
        status = "✓" if (MODEL_DIR / m.local_path).exists() else "○"
        print(f"  {status} {m.name}")
        print(f"    Path:     {m.local_path}")
        print(f"    License:  {m.license}")
        print(f"    Note:     Must be obtained manually or via specialized script")
        print()


def cmd_download(model_name: str | None = None) -> None:
    """Download models."""
    if model_name:
        # Find by local_path prefix
        found = False
        for m in MODEL_REGISTRY + OPTIONAL_MODELS:
            if m.local_path.startswith(model_name):
                try:
                    download_model(m)
                    found = True
                except (ValueError, RuntimeError) as e:
                    print(f"Error: {e}", file=sys.stderr)
                break

        if not found:
            print(f"Unknown model: {model_name}")
            print("Run with --list to see available models.")
            sys.exit(1)
    else:
        print(f"Downloading models to {MODEL_DIR.absolute()}...\n")
        downloaded = download_all()

        print(f"\nDone! Downloaded {len(downloaded)} model(s).")

        # Note about optional models
        missing = [m for m in OPTIONAL_MODELS if not (MODEL_DIR / m.local_path).exists()]
        if missing:
            print(f"\nOptional models not downloaded ({len(missing)}):")
            for m in missing:
                print(f"  - {m.name}: {m.description}")
            print("\nThese will be handled by their respective model wrappers.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download audio-edge models")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--model", type=str, default=None, help="Download specific model")
    args = parser.parse_args()

    setup_logging(level="INFO")

    if args.list:
        cmd_list()
    else:
        cmd_download(args.model)


if __name__ == "__main__":
    main()
