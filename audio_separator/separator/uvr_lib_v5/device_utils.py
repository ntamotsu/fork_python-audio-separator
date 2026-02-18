"""Device capability helpers for complex spectral operations."""

from functools import lru_cache
import os
import torch


_FORCE_CPU_COMPLEX = os.getenv("AUDIO_SEPARATOR_FORCE_CPU_COMPLEX", "").strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=32)
def _supports_complex_stft_istft(device_type: str, device_index: int) -> bool:
    """Return whether the target device can run complex STFT/ISTFT end-to-end."""
    if device_type in {"cpu", "cuda"}:
        return True

    try:
        device = torch.device(f"{device_type}:{device_index}") if device_index >= 0 else torch.device(device_type)
        sample_length = 1024
        n_fft = 256
        hop_length = 64
        x = torch.randn(1, sample_length, device=device)
        window = torch.hann_window(n_fft, device=device)
        z = torch.stft(x, n_fft=n_fft, hop_length=hop_length, window=window, center=True, return_complex=True)
        _ = torch.istft(z, n_fft=n_fft, hop_length=hop_length, window=window, center=True, length=sample_length)
        return True
    except Exception:
        return False


def should_fallback_to_cpu_for_complex_ops(device: torch.device) -> bool:
    """
    Decide whether to run complex spectral ops on CPU for this device.

    Set `AUDIO_SEPARATOR_FORCE_CPU_COMPLEX=1` to force legacy CPU fallback behavior.
    """
    if _FORCE_CPU_COMPLEX:
        return True

    device_index = -1 if device.index is None else int(device.index)
    return not _supports_complex_stft_istft(device.type, device_index)

