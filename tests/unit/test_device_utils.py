from unittest.mock import patch

import pytest
import torch

from audio_separator.separator.uvr_lib_v5 import device_utils


@pytest.fixture(autouse=True)
def clear_device_capability_cache():
    device_utils._supports_complex_spectral_ops.cache_clear()
    yield
    device_utils._supports_complex_spectral_ops.cache_clear()


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_standard_devices_are_supported_without_a_runtime_probe(device_type):
    with patch.object(device_utils.torch, "device") as device_constructor:
        assert device_utils._supports_complex_spectral_ops(device_type, -1) is True

    device_constructor.assert_not_called()


def test_directml_short_circuits_without_a_runtime_probe():
    with patch.object(device_utils.torch, "device") as device_constructor:
        assert device_utils._supports_complex_spectral_ops("privateuseone", -1) is False

    device_constructor.assert_not_called()


def test_probe_failure_preserves_cpu_fallback():
    with patch.object(device_utils.torch, "device", side_effect=RuntimeError("unsupported")):
        assert device_utils._supports_complex_spectral_ops("mps", -1) is False


def test_force_cpu_environment_flag_overrides_capability_probe(monkeypatch):
    monkeypatch.setenv("AUDIO_SEPARATOR_FORCE_CPU_COMPLEX", "1")
    with patch.object(device_utils, "_supports_complex_spectral_ops") as probe:
        assert device_utils.should_fallback_to_cpu_for_complex_ops(torch.device("cpu")) is True

    probe.assert_not_called()


def test_fallback_decision_uses_cached_capability_probe(monkeypatch):
    monkeypatch.delenv("AUDIO_SEPARATOR_FORCE_CPU_COMPLEX", raising=False)
    with patch.object(device_utils, "_supports_complex_spectral_ops", return_value=True) as probe:
        assert device_utils.should_fallback_to_cpu_for_complex_ops(torch.device("mps")) is False

    probe.assert_called_once_with("mps", -1)


@pytest.mark.parametrize(
    ("device_type", "expected"),
    [("cpu", False), ("cuda", False), ("privateuseone", False), ("mps", True)],
)
def test_device_accumulation_is_limited_to_mps(device_type, expected):
    assert device_utils.should_accumulate_on_device(torch.device(device_type)) is expected


def test_probe_rejects_a_device_without_complex_scatter_support():
    cpu_device = torch.device("cpu")
    with (
        patch.object(device_utils.torch, "device", return_value=cpu_device),
        patch.object(device_utils, "_probe_complex_scatter_add", side_effect=RuntimeError("unsupported")),
    ):
        assert device_utils._supports_complex_spectral_ops("mps", -1) is False


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_mps_probe_returns_a_boolean():
    assert isinstance(device_utils._supports_complex_spectral_ops("mps", -1), bool)
