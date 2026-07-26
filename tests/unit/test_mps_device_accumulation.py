from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from audio_separator.separator.architectures.demucs_separator import DemucsSeparator
from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator


def _demucs_separator(device: torch.device) -> DemucsSeparator:
    separator = object.__new__(DemucsSeparator)
    separator.logger = Mock()
    separator.torch_device = device
    separator.demucs_model_instance = Mock()
    separator.shifts = 0
    separator.segments_enabled = True
    separator.overlap = 0.25
    return separator


def _mix() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((2, 128), dtype=np.float32)


class _FakeMDXCModel:
    num_target_instruments = 2

    def __init__(self, expected_device: torch.device):
        self.expected_device = expected_device

    def __call__(self, batch):
        assert batch.device.type == self.expected_device.type
        return batch.unsqueeze(1).repeat(1, 2, 1, 1)


def _mdxc_separator(device: torch.device) -> MDXCSeparator:
    separator = object.__new__(MDXCSeparator)
    separator.logger = Mock()
    separator.torch_device = device
    separator.model_run = _FakeMDXCModel(device)
    separator.model_data_cfgdict = SimpleNamespace(
        training=SimpleNamespace(instruments=["first", "second"], target_instrument=None),
        inference=SimpleNamespace(dim_t=5),
        audio=SimpleNamespace(hop_length=2),
    )
    separator.pitch_shift = 0
    separator.is_roformer = False
    separator.segment_size = 5
    separator.overlap = 2
    separator.batch_size = 1
    separator.is_primary_stem_main_target = False
    return separator


def test_demucs_keeps_full_track_input_on_cpu_for_cpu_inference():
    separator = _demucs_separator(torch.device("cpu"))

    def fake_apply_model(*, mix, **kwargs):
        assert mix.device.type == "cpu"
        return torch.zeros(1, 4, 2, mix.shape[-1], device=mix.device)

    with patch("audio_separator.separator.architectures.demucs_separator.apply_model", side_effect=fake_apply_model):
        result = separator.demix_demucs(_mix())

    assert result.shape == (4, 2, 128)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_demucs_keeps_full_track_input_on_mps():
    separator = _demucs_separator(torch.device("mps"))

    def fake_apply_model(*, mix, **kwargs):
        assert mix.device.type == "mps"
        return torch.zeros(1, 4, 2, mix.shape[-1], device=mix.device)

    with patch("audio_separator.separator.architectures.demucs_separator.apply_model", side_effect=fake_apply_model):
        result = separator.demix_demucs(_mix())

    assert result.shape == (4, 2, 128)


@pytest.mark.parametrize("device_type", ["cpu", "mps"])
def test_mdxc_chunk_buffers_share_the_selected_accumulation_device(device_type):
    if device_type == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    device = torch.device(device_type)
    separator = _mdxc_separator(device)
    allocated_devices = []
    torch_zeros = torch.zeros

    def tracked_zeros(*args, **kwargs):
        tensor = torch_zeros(*args, **kwargs)
        allocated_devices.append(tensor.device.type)
        return tensor

    with patch("audio_separator.separator.architectures.mdxc_separator.torch.zeros", side_effect=tracked_zeros):
        result = separator.demix(_mix(), override_model_segment_size=True)

    assert set(result) == {"first", "second"}
    assert all(stem.shape == (2, 128) for stem in result.values())
    assert allocated_devices
    assert set(allocated_devices) == {device_type}
