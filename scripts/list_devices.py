#!/usr/bin/env python3
"""List available audio input devices on the current system.

Use this to find your USB sound card device name for configuration.
"""

import sys
from pathlib import Path

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capture.device import list_input_devices


def main():
    devices = list_input_devices()

    if not devices:
        print("No audio input devices found!")
        print("Check that a microphone or USB sound card is connected.")
        sys.exit(1)

    print(f"Found {len(devices)} audio input device(s):\n")
    print(f"{'Index':<6} {'Default':<8} {'Name':<50} {'API':<15} {'Max Ch':<8} {'Sample Rate'}")
    print("-" * 100)

    for d in devices:
        default_mark = "✓" if d.is_default else ""
        print(
            f"{d.index:<6} "
            f"{default_mark:<8} "
            f"{d.name:<50} "
            f"{d.hostapi:<15} "
            f"{d.max_input_channels:<8} "
            f"{d.default_sample_rate:.0f} Hz"
        )

    print("\n---")
    print("Use the device name (or index) in your config YAML:")
    print("  audio:")
    print(f'    device: "{devices[0].name}"  # or index: {devices[0].index}')


if __name__ == "__main__":
    main()
