#!/usr/bin/env python3
"""Build TensorRT engines from ONNX models for Jetson Orin NX.

Converts ONNX models to optimized TensorRT FP16 engines, achieving
2-5x speedup over ONNX Runtime on Jetson hardware.

Usage:
    python scripts/build_trt_engines.py              # Build all models
    python scripts/build_trt_engines.py --model vad  # Build specific model
    python scripts/build_trt_engines.py --fp32       # FP32 precision (debug)

Prerequisites:
    - JetPack with TensorRT installed
    - ONNX models downloaded (python scripts/download_models.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Model configurations for TRT build
# Each entry: (onnx_path, engine_path, input_shapes, fp16_safe)
MODEL_CONFIGS = {
    "vad": {
        "onnx": "models/vad/silero_vad.onnx",
        "engine": "models/engines/silero_vad_fp16.trt",
        "inputs": {
            "input": (1, 512),
            "state": (2, 1, 128),
            "sr": (1,),
        },
        "fp16": True,
    },
    "kws": {
        "onnx": "models/kws/encoder.onnx",
        "engine": "models/engines/kws_encoder_fp16.trt",
        "inputs": {},  # Dynamic — build separately
        "fp16": True,
        "note": "KWS has multiple sub-models (encoder, decoder, joiner). Build each separately.",
    },
    "sed": {
        "onnx": "models/sed/yamnet.onnx",
        "engine": "models/engines/yamnet_fp16.trt",
        "inputs": {
            "input_1": (1, 96, 64),  # mel spectrogram
        },
        "fp16": True,
    },
    "asc": {
        "onnx": "models/asc/ast-finetuned.onnx",
        "engine": "models/engines/ast_fp16.trt",
        "inputs": {
            "input": (1, 128, 100),
        },
        "fp16": True,
    },
}


def build_trt_engine(
    onnx_path: str,
    engine_path: str,
    input_shapes: dict[str, tuple[int, ...]],
    use_fp16: bool = True,
    workspace_gb: int = 2,
) -> bool:
    """Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path: Path to ONNX model.
        engine_path: Output path for .trt engine.
        input_shapes: Dict of input_name → (min, opt, max) or (shape,) tuples.
        use_fp16: Enable FP16 precision.
        workspace_gb: Max GPU memory for engine build (GB).

    Returns:
        True if build succeeded.
    """
    try:
        import tensorrt as trt
    except ImportError:
        logger.error(
            "TensorRT Python bindings not found. Ensure JetPack is installed: sudo apt install python3-libnvinfer"
        )
        return False

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)

    if not onnx_path.exists():
        logger.warning(f"ONNX model not found: {onnx_path}")
        return False

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building TensorRT engine: {onnx_path.name} → {engine_path.name}")
    logger.info(f"  FP16: {use_fp16}, Workspace: {workspace_gb}GB")

    logger_ = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger_)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, logger_)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"  ONNX parse error: {parser.get_error(i)}")
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1024**3))

    if use_fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("  FP16 mode enabled")
        else:
            logger.warning("  FP16 not supported on this platform, using FP32")

    if input_shapes:
        profile = builder.create_optimization_profile()
        for input_name, shapes in input_shapes.items():
            if len(shapes) == 3:
                profile.set_shape(input_name, shapes[0], shapes[1], shapes[2])
            else:
                # Single shape → min = opt = max
                shape = shapes if isinstance(shapes, tuple) else tuple(shapes)
                profile.set_shape(input_name, shape, shape, shape)
        config.add_optimization_profile(profile)

    # Build engine
    import time

    t0 = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    build_time = time.time() - t0

    if serialized_engine is None:
        logger.error(f"  Engine build failed for {onnx_path.name}")
        return False

    # Write engine to disk
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    size_mb = engine_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Built {engine_path.name}: {size_mb:.1f}MB in {build_time:.1f}s")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build TensorRT engines from ONNX models")
    parser.add_argument("--model", type=str, default=None, help="Build specific model (vad/kws/sed/asc)")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 instead of FP16")
    parser.add_argument("--workspace", type=int, default=2, help="GPU workspace in GB")
    args = parser.parse_args()

    setup_logging(level="INFO")

    logger.info("=" * 50)
    logger.info("Building TensorRT Engines")
    logger.info("=" * 50)

    if args.model:
        configs = [(args.model, MODEL_CONFIGS[args.model])]
    else:
        configs = MODEL_CONFIGS.items()

    success_count = 0
    for name, cfg in configs:
        onnx_path = cfg["onnx"]
        engine_path = cfg["engine"]

        if not Path(onnx_path).exists():
            logger.warning(f"Skipping {name}: ONNX model not found at {onnx_path}")
            continue

        if "note" in cfg:
            logger.info(f"Note for {name}: {cfg['note']}")

        use_fp16 = cfg.get("fp16", True) and not args.fp32

        if build_trt_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            input_shapes=cfg.get("inputs", {}),
            use_fp16=use_fp16,
            workspace_gb=args.workspace,
        ):
            success_count += 1

    logger.info(f"\nDone! {success_count}/{len(configs)} engines built.")


if __name__ == "__main__":
    main()
