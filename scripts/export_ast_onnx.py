#!/usr/bin/env python3
"""Export AST (Audio Spectrogram Transformer) from HuggingFace to ONNX.

Downloads the MIT/ast-finetuned-audioset model from HuggingFace and exports
it to ONNX format for use with the audio-edge ASC inference pipeline.

Model:  MIT/ast-finetuned-audioset (AudioSet, 527 classes)
Input:  mel spectrogram  [batch, 128, N_frames]  (dynamic frame count)
Output: logits           [batch, 527]

Usage:
    python scripts/export_ast_onnx.py
    python scripts/export_ast_onnx.py --output models/asc/ast-finetuned.onnx --frames 1024

Requirements:
    pip install torch transformers onnx

Reference:
    AST: https://huggingface.co/MIT/ast-finetuned-audioset
    License: MIT
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

HF_MODEL_ID = "MIT/ast-finetuned-audioset"
DEFAULT_NUM_FRAMES = 1024  # Original AST expects ~10s @ 16kHz (128 mel × 1024 frames)
NUM_MELS = 128
NUM_CLASSES = 527

# AudioSet scene-related class labels (subset of 527)
# Extracted from the AudioSet ontology for acoustic scene classification
AUDIOSET_SCENE_LABELS = [
    "Indoor",
    "Outdoor",
    "Office",
    "Street",
    "Cafe/Restaurant",
    "Home",
    "Park",
    "Public transport",
    "Car",
    "Market",
    "Library",
    "Station",
    "Airport",
    "Hospital",
    "School",
    "Church",
    "Factory",
    "Construction site",
    "Beach",
    "Forest",
    "Urban",
    "Rural",
    "Subway",
    "Bus",
    "Train",
    "Bar",
    "Restaurant",
    "Shopping mall",
    "Stadium",
    "Theater",
]


def export_ast_onnx(
    output_path: str | Path,
    num_frames: int = DEFAULT_NUM_FRAMES,
    dynamic_frames: bool = True,
    opset: int = 14,
) -> Path:
    """Export AST model to ONNX.

    Args:
        output_path: Destination for the ONNX file.
        num_frames: Target number of mel frames (used for export tracing).
        dynamic_frames: If True, export with dynamic frame count axis.
        opset: ONNX opset version.

    Returns:
        Path to the exported model.

    Raises:
        ImportError: If torch or transformers are not installed.
        RuntimeError: If export fails.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info(f"ONNX model already exists: {output_path}")
        return output_path

    try:
        import torch
        from transformers import ASTForAudioClassification
    except ImportError as e:
        raise ImportError(
            "torch and transformers are required. "
            "Install with: pip install torch transformers onnx"
        ) from e

    logger.info(f"Loading AST model from HuggingFace: {HF_MODEL_ID}")
    model = ASTForAudioClassification.from_pretrained(
        HF_MODEL_ID,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    model.eval()

    # Create dummy input: mel spectrogram
    # AST expects: [batch, max_length, num_mel_bins] = [1, num_frames, 128]
    # Note: HuggingFace AST uses (batch, time, freq) convention
    dummy_input = torch.randn(1, num_frames, NUM_MELS, dtype=torch.float32)

    logger.info(f"  Input shape:  {dummy_input.shape} (batch, frames, mel_bands)")
    logger.info(f"  Dynamic frames: {dynamic_frames}")

    # Dynamic axes for variable-length input
    if dynamic_frames:
        dynamic_axes = {
            "input_values": {0: "batch", 1: "frames"},
            "logits": {0: "batch"},
        }
    else:
        dynamic_axes = {
            "input_values": {0: "batch"},
            "logits": {0: "batch"},
        }

    logger.info("Exporting to ONNX...")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  ONNX model saved: {output_path}")
    logger.info(f"  Size: {file_size_mb:.1f} MB")

    return output_path


def save_scene_labels(output_dir: Path) -> Path:
    """Save scene classification labels to a text file.

    The exported AST model outputs logits for all 527 AudioSet classes.
    We save a subset of scene-related labels as the default scene set,
    but the full AudioSet class map should also be available.

    Args:
        output_dir: Directory to save scene_labels.txt.

    Returns:
        Path to the labels file.
    """
    labels_path = output_dir / "scene_labels.txt"
    if labels_path.exists():
        logger.info(f"Scene labels already exist: {labels_path}")
        return labels_path

    with open(labels_path, "w", encoding="utf-8") as f:
        for label in AUDIOSET_SCENE_LABELS:
            f.write(label + "\n")

    logger.info(f"  Saved {len(AUDIOSET_SCENE_LABELS)} scene labels: {labels_path}")
    return labels_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Export AST to ONNX")
    parser.add_argument(
        "--output", type=str, default="models/asc/ast-finetuned.onnx", help="Output ONNX model path"
    )
    parser.add_argument(
        "--frames", type=int, default=DEFAULT_NUM_FRAMES, help="Mel frames for export tracing"
    )
    parser.add_argument(
        "--static-frames", action="store_true", help="Export with fixed frame count (no dynamic axis)"
    )
    parser.add_argument(
        "--opset", type=int, default=14, help="ONNX opset version"
    )
    args = parser.parse_args()

    setup_logging(level="INFO")

    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Export ONNX model
    logger.info("=" * 50)
    logger.info("Exporting AST ONNX model")
    logger.info("=" * 50)

    try:
        export_ast_onnx(
            output_path=args.output,
            num_frames=args.frames,
            dynamic_frames=not args.static_frames,
            opset=args.opset,
        )
        logger.info("AST ONNX export complete!")
    except ImportError as e:
        logger.error(str(e))
        logger.info(
            "\nTo export AST manually:\n"
            "  1. pip install torch transformers onnx\n"
            "  2. python scripts/export_ast_onnx.py\n"
            "\nOr download a pre-exported ONNX from HuggingFace model hub."
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)

    # Step 2: Save scene labels
    logger.info("=" * 50)
    logger.info("Saving scene labels")
    logger.info("=" * 50)
    save_scene_labels(output_dir)

    logger.info(f"\nDone!")
    logger.info(f"  Model:  {output_path}")
    logger.info(f"  Labels: {output_dir / 'scene_labels.txt'}")


if __name__ == "__main__":
    main()
