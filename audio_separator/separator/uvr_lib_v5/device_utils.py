"""Device capability helpers for complex spectral operations."""

import os
from functools import lru_cache

import torch


def _probe_complex_scatter_add(spectrum: torch.Tensor) -> None:
    """Exercise the complex scatter operation used by MelBand RoFormer."""
    source = spectrum[:, :2, :2]
    indices = torch.zeros(source.shape, dtype=torch.long, device=source.device)
    torch.zeros_like(source).scatter_add_(1, indices, source)


@lru_cache(maxsize=32)
def _supports_complex_spectral_ops(device_type: str, device_index: int) -> bool:
    """Return whether a device can execute the complex operations used by the models."""
    if device_type in {"cpu", "cuda"}:
        return True

    # DirectML cannot represent complex tensors. Avoid probing unsupported
    # operations on its out-of-tree backend slot.
    if device_type == "privateuseone":
        return False

    try:
        device = torch.device(f"{device_type}:{device_index}") if device_index >= 0 else torch.device(device_type)
        sample_length = 1024
        n_fft = 256
        hop_length = 64
        sample = torch.randn(1, sample_length, device=device)
        window = torch.hann_window(n_fft, device=device)
        spectrum = torch.stft(sample, n_fft=n_fft, hop_length=hop_length, window=window, center=True, return_complex=True)
        spectrum = torch.view_as_complex(torch.view_as_real(spectrum).contiguous()) * torch.ones_like(spectrum)
        _probe_complex_scatter_add(spectrum)
        torch.istft(spectrum, n_fft=n_fft, hop_length=hop_length, window=window, center=True, length=sample_length)
        return True
    except Exception:
        return False


def should_fallback_to_cpu_for_complex_ops(device: torch.device) -> bool:
    """Return whether complex spectral operations should use the legacy CPU path."""
    if os.environ.get("AUDIO_SEPARATOR_FORCE_CPU_COMPLEX"):
        return True

    device_index = -1 if device.index is None else int(device.index)
    return not _supports_complex_spectral_ops(device.type, device_index)


def should_accumulate_on_device(device: torch.device) -> bool:
    """Keep full-track accumulators on MPS, where CPU and GPU share unified memory."""
    return device.type == "mps"
