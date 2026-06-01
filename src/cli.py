"""audio-edge CLI entry point.

Usage:
    audio-edge run                      # Run with default config
    audio-edge run --config configs/jetson_trt.yaml
    audio-edge list-devices             # List audio input devices
    audio-edge test-audio               # Test audio capture (RMS meter)
"""

import signal
import sys
from pathlib import Path

import typer

from src.utils.config import load_config
from src.utils.logging import setup_logging

app = typer.Typer(
    name="audio-edge",
    help="Multi-task real-time audio intelligence for Jetson edge devices",
    add_completion=False,
)


@app.command()
def run(
    config_path: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Path to YAML config file"
    ),
    mock: bool = typer.Option(False, "--mock/--no-mock", help="Use mock audio capture (silence generator)"),
    duration: float = typer.Option(0, "--duration", "-d", help="Run for N seconds (0 = until Ctrl+C)"),
):
    """Run the audio-edge inference pipeline."""
    # Load config
    cfg = load_config(config_path)
    cfg["_mock_audio"] = mock

    # Setup logging
    log_cfg = cfg["logging"]
    setup_logging(
        level=log_cfg["level"],
        fmt=log_cfg["format"],
        log_file=log_cfg.get("file"),
    )

    # Import here to avoid loading models at import time
    from src.pipeline.orchestrator import Orchestrator

    orch = Orchestrator(cfg)

    # Handle Ctrl+C gracefully
    def _sigint(signum, frame):
        orch.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint)

    try:
        orch.run(duration_sec=duration)
    except KeyboardInterrupt:
        pass


@app.command()
def list_devices():
    """List available audio input devices."""
    from src.capture.device import list_input_devices

    devices = list_input_devices()
    if not devices:
        typer.echo("No audio input devices found!")
        raise typer.Exit(1)

    typer.echo(f"\nFound {len(devices)} audio input device(s):\n")
    for d in devices:
        default = " [DEFAULT]" if d.is_default else ""
        typer.echo(
            f"  [{d.index}] {d.name}{default}\n"
            f"      API: {d.hostapi}, Channels: {d.max_input_channels}, "
            f"Sample Rate: {d.default_sample_rate:.0f} Hz"
        )


@app.command()
def test_audio(
    device: str | None = typer.Option(None, "--device", "-d", help="Audio device name"),
    sample_rate: int = typer.Option(16000, "--sample-rate", "-r", help="Sample rate in Hz"),
    duration: float = typer.Option(0, "--duration", "-t", help="Duration in seconds (0=until Ctrl+C)"),
    mock: bool = typer.Option(False, "--mock", help="Use mock capture (silence)"),
):
    """Test audio capture with an RMS VU meter."""
    import time

    import numpy as np

    from src.capture.stream import AudioCapture, MockAudioCapture

    capture_cls = MockAudioCapture if mock else AudioCapture
    capture = capture_cls(sample_rate=sample_rate, device=device)

    typer.echo("\nAudio Capture Test")
    typer.echo(f"  Device: {capture.device_name}")
    typer.echo(f"  Sample Rate: {capture.sample_rate} Hz")
    typer.echo(f"  Mode: {'MOCK' if mock else 'LIVE'}")
    typer.echo("  Press Ctrl+C to stop.\n")

    capture.start()
    start_time = time.time()

    try:
        while True:
            chunk_size = int(sample_rate * 0.1)
            audio = capture.read(chunk_size)

            if len(audio) > 0:
                rms = float(np.sqrt(np.mean(audio**2)))
                db = 20 * np.log10(max(rms, 1e-10))
                bar_len = int(np.clip((db + 60) / 3, 0, 20))
                bar = "█" * bar_len + "░" * (20 - bar_len)

                elapsed = time.time() - start_time
                typer.echo(
                    f"\r[{elapsed:6.1f}s] RMS: {rms:.4f} | {db:6.1f} dB | [{bar}]",
                    nl=False,
                )

            if duration > 0 and (time.time() - start_time) >= duration:
                break

    except KeyboardInterrupt:
        typer.echo("\n\nStopping...")
    finally:
        capture.stop()

    typer.echo("Done.")


@app.command()
def download_models():
    """Download required ONNX models."""
    from src.models.model_loader import download_all
    from src.utils.logging import setup_logging

    setup_logging(level="INFO")
    download_all()


def main():
    app()


if __name__ == "__main__":
    main()
