"""Audio device enumeration and selection utilities."""

from dataclasses import dataclass

import sounddevice as sd


@dataclass
class AudioDevice:
    """Lightweight representation of an audio input device."""

    index: int
    name: str
    hostapi: str
    max_input_channels: int
    default_sample_rate: float
    is_default: bool = False


def list_input_devices() -> list[AudioDevice]:
    """Enumerate all available audio input devices.

    Returns:
        List of AudioDevice objects for input-capable devices.
    """
    devices = []
    default_device = sd.default.device[0] if sd.default.device else None

    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] <= 0:
            continue

        hostapi = sd.query_hostapis(dev["hostapi"])["name"]
        devices.append(
            AudioDevice(
                index=idx,
                name=dev["name"],
                hostapi=hostapi,
                max_input_channels=dev["max_input_channels"],
                default_sample_rate=dev["default_samplerate"],
                is_default=(idx == default_device),
            )
        )

    return devices


def find_device(identifier: str | int | None = None) -> AudioDevice | None:
    """Find an audio input device by name substring or index.

    Args:
        identifier: Device name substring, numeric index (as int or str),
                    or None for default device.

    Returns:
        Matching AudioDevice, or None if not found.
    """
    if identifier is None:
        # Use system default
        default_idx = sd.default.device[0]
        if default_idx is not None:
            return _device_from_index(default_idx)
        devices = list_input_devices()
        return devices[0] if devices else None

    if isinstance(identifier, int):
        return _device_from_index(identifier)

    # Try as integer string
    try:
        return _device_from_index(int(identifier))
    except ValueError:
        pass

    # Search by name substring
    identifier_lower = identifier.lower()
    for dev in list_input_devices():
        if identifier_lower in dev.name.lower():
            return dev

    return None


def _device_from_index(index: int) -> AudioDevice | None:
    """Get AudioDevice by index, returning None if invalid."""
    try:
        dev = sd.query_devices(index)
        if dev["max_input_channels"] <= 0:
            return None
        hostapi = sd.query_hostapis(dev["hostapi"])["name"]
        return AudioDevice(
            index=index,
            name=dev["name"],
            hostapi=hostapi,
            max_input_channels=dev["max_input_channels"],
            default_sample_rate=dev["default_samplerate"],
            is_default=(sd.default.device and sd.default.device[0] == index),
        )
    except (sd.PortAudioError, IndexError):
        return None
