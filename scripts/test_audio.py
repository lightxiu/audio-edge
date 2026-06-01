#!/usr/bin/env python3
"""Test audio capture — prints live RMS levels to verify the mic is working.

Usage:
    python scripts/test_audio.py              # Use default device
    python scripts/test_audio.py --device "USB"  # Search by name
    python scripts/test_audio.py --duration 10    # Run for 10 seconds

Press Ctrl+C to stop.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.capture.stream import AudioCapture, MockAudioCapture
from src.utils.logging import setup_logging


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test audio capture loop")
    parser.add_argument("--device", type=str, default=None, help="Audio device name or index")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate in Hz")
    parser.add_argument("--duration", type=float, default=0, help="Duration in seconds (0=until Ctrl+C)")
    parser.add_argument("--mock", action="store_true", help="Use mock capture (silence generator)")
    args = parser.parse_args()

    setup_logging(level="INFO")

    # Create capture
    capture_cls = MockAudioCapture if args.mock else AudioCapture
    capture = capture_cls(
        sample_rate=args.sample_rate,
        device=args.device,
    )

    print(f"\n{'=' * 60}")
    print("Audio Capture Test")
    print(f"{'=' * 60}")
    print(f"Device: {capture.device_name}")
    print(f"Sample Rate: {capture.sample_rate} Hz")
    print(f"Mode: {'MOCK (silence)' if args.mock else 'LIVE'}")
    print("Press Ctrl+C to stop.\n")

    capture.start()

    start_time = time.time()
    try:
        while True:
            # Read ~100ms of audio
            chunk_size = int(capture.sample_rate * 0.1)
            audio = capture.read(chunk_size)

            if len(audio) > 0:
                # Compute RMS (root mean square) as volume indicator
                rms = float(np.sqrt(np.mean(audio**2)))
                db = 20 * np.log10(max(rms, 1e-10))

                # Simple VU meter
                bar_len = int(np.clip((db + 60) / 3, 0, 20))
                bar = "█" * bar_len + "░" * (20 - bar_len)

                elapsed = time.time() - start_time
                print(f"\r[{elapsed:6.1f}s] RMS: {rms:.4f} | {db:6.1f} dB | [{bar}]", end="", flush=True)

            # Check duration
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                break

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        capture.stop()

    print("Done.")


if __name__ == "__main__":
    main()
