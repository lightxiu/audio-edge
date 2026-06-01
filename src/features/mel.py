"""Mel spectrogram and MFCC feature extraction.

Shared feature extraction used by KWS, SED, and ASC models.
Compute once, feed to all downstream tasks to avoid redundant computation.
"""

from dataclasses import dataclass

import numpy as np
from scipy.fft import rfft
from scipy.signal import get_window


@dataclass
class MelSpectrogram:
    """Computed mel spectrogram with metadata."""

    features: np.ndarray  # Shape: (n_mels, n_frames)
    n_mels: int
    n_fft: int
    hop_length: int
    sample_rate: int
    duration_sec: float


@dataclass
class MFCC:
    """Computed MFCC features with metadata."""

    features: np.ndarray  # Shape: (n_mfcc, n_frames)
    n_mfcc: int
    n_fft: int
    hop_length: int
    sample_rate: int


def mel_filter_bank(
    n_mels: int = 64,
    n_fft: int = 1024,
    sample_rate: int = 16000,
    f_min: float = 80.0,
    f_max: float = 7600.0,
) -> np.ndarray:
    """Create a mel filter bank matrix.

    Args:
        n_mels: Number of mel bands.
        n_fft: FFT size.
        sample_rate: Audio sample rate.
        f_min: Minimum frequency in Hz.
        f_max: Maximum frequency in Hz.

    Returns:
        Filter bank matrix of shape (n_mels, n_fft // 2 + 1).
    """
    n_freqs = n_fft // 2 + 1

    # Convert frequencies to mel scale
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)

    # Create equally spaced mel points
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    # Convert to FFT bin indices
    bin_indices = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # Create filter bank
    filters = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for m in range(n_mels):
        f_start = bin_indices[m]
        f_center = bin_indices[m + 1]
        f_end = bin_indices[m + 2]

        # Rising slope
        if f_center > f_start:
            filters[m, f_start:f_center] = (np.arange(f_start, f_center) - f_start) / (f_center - f_start)

        # Falling slope
        if f_end > f_center:
            filters[m, f_center:f_end] = (f_end - np.arange(f_center, f_end)) / (f_end - f_center)

    return filters


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 160,  # 10ms @ 16kHz
    f_min: float = 80.0,
    f_max: float = 7600.0,
    power: float = 2.0,
    window: str = "hann",
) -> MelSpectrogram:
    """Compute mel spectrogram from raw audio.

    Args:
        audio: 1-D float32 audio array.
        sample_rate: Sample rate in Hz.
        n_mels: Number of mel bands.
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        f_min: Minimum frequency.
        f_max: Maximum frequency (Nyquist = sample_rate / 2).
        power: Power for magnitude spectrogram (2.0 = energy).
        window: Window function name.

    Returns:
        MelSpectrogram with features of shape (n_mels, n_frames).
    """
    if audio.ndim != 1:
        audio = audio.squeeze()

    audio = audio.astype(np.float32)

    # STFT
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    if n_frames <= 0:
        # Audio shorter than n_fft — pad
        audio = np.pad(audio, (0, n_fft - len(audio)), mode="constant")
        n_frames = 1

    # Create window
    win = get_window(window, n_fft, fftbins=True).astype(np.float32)

    # Pre-allocate spectrogram
    magnitude = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start : start + n_fft] * win
        spec = np.abs(rfft(frame)) ** power
        magnitude[:, i] = spec

    # Apply mel filter bank
    mel_filters = mel_filter_bank(n_mels, n_fft, sample_rate, f_min, f_max)
    mel_spec = np.dot(mel_filters, magnitude)

    # Convert to log scale (dB)
    mel_spec = np.log(np.maximum(mel_spec, 1e-10))

    return MelSpectrogram(
        features=mel_spec,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        sample_rate=sample_rate,
        duration_sec=len(audio) / sample_rate,
    )


def compute_mfcc(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 160,
    f_min: float = 80.0,
    f_max: float = 7600.0,
) -> MFCC:
    """Compute MFCC features from raw audio.

    MFCC = DCT of log mel spectrogram.

    Args:
        audio: 1-D float32 audio array.
        sample_rate: Sample rate in Hz.
        n_mfcc: Number of MFCC coefficients (excluding 0th).
        n_mels: Number of mel bands (intermediate representation).
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        f_min: Minimum frequency.
        f_max: Maximum frequency.

    Returns:
        MFCC with features of shape (n_mfcc, n_frames).
    """
    mel = compute_mel_spectrogram(
        audio=audio,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
    )

    # DCT type-II on log mel spectrogram
    mfcc_features = _dct_type2(mel.features, n_mfcc=n_mfcc)

    return MFCC(
        features=mfcc_features,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        sample_rate=sample_rate,
    )


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
    """Convert Hz to mel scale."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    """Convert mel scale to Hz."""
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _dct_type2(x: np.ndarray, n_mfcc: int = 40) -> np.ndarray:
    """Compute DCT Type-II along the mel axis.

    Args:
        x: Input of shape (n_mels, n_frames).
        n_mfcc: Number of DCT coefficients to keep.

    Returns:
        DCT coefficients of shape (n_mfcc, n_frames).
    """
    n_mels, n_frames = x.shape
    n = np.arange(n_mels)
    k = np.arange(1, n_mfcc + 1).reshape(-1, 1)
    dct_matrix = np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))
    return np.dot(dct_matrix, x)
