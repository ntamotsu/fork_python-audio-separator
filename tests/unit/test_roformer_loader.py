"""Behavior tests for the concrete Roformer loader."""

from unittest.mock import Mock, patch

import pytest

from audio_separator.separator.roformer.roformer_loader import RoformerLoader
from audio_separator.separator.roformer.model_loading_result import ImplementationVersion


def test_detect_model_type_ignores_parent_directory():
    """Path-only detection must use the checkpoint basename."""
    loader = RoformerLoader()

    model_type = loader.detect_model_type("/tmp/bs_roformer_experiment/mel_band_roformer_karaoke.ckpt")

    assert model_type == "mel_band_roformer"


def test_load_model_reports_ambiguous_checkpoint_basename_as_failure_result():
    """Concrete loader errors use one result contract for config and filename ambiguity."""
    loader = RoformerLoader()

    result = loader.load_model(
        "/tmp/bs_roformer_mel_band_roformer.ckpt",
        {"dim": 384, "depth": 12},
    )

    assert result.success is False
    assert "both BS and MelBand Roformer tokens" in result.error_message


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "dereverb_big_mbr_ep_362.ckpt",
        "dereverb_super_big_mbr_ep_346.ckpt",
        "dereverb_echo_mbr_fused.ckpt",
    ],
)
def test_load_model_uses_nested_config_for_registry_model_without_family_in_filename(checkpoint_name):
    """The concrete loader must select MelBand from config without using fallback."""
    loader = RoformerLoader()
    model = Mock()
    loader._create_mel_band_roformer = Mock(return_value=model)
    loader._create_bs_roformer = Mock()
    config = {
        "model": {
            "dim": 384,
            "depth": 12,
            "num_bands": 60,
        }
    }

    result = loader.load_model(
        f"/tmp/bs_roformer_experiment/{checkpoint_name}",
        config,
    )

    assert result.success is True
    assert result.implementation_used is ImplementationVersion.NEW
    assert result.model_info["model_type"] == "mel_band_roformer"
    loader._create_mel_band_roformer.assert_called_once()
    loader._create_bs_roformer.assert_not_called()


def test_load_model_uses_nested_bs_config_despite_mel_parent_directory():
    """The BS-Roformer-SW config remains authoritative in a misleading directory."""
    loader = RoformerLoader()
    model = Mock()
    loader._create_bs_roformer = Mock(return_value=model)
    loader._create_mel_band_roformer = Mock()
    config = {
        "model": {
            "dim": 384,
            "depth": 12,
            "freqs_per_bands": (2, 4, 8, 16, 32, 64),
        }
    }

    result = loader.load_model(
        "/tmp/mel_band_roformer_experiment/BS-Roformer-SW.ckpt",
        config,
    )

    assert result.success is True
    assert result.implementation_used is ImplementationVersion.NEW
    assert result.model_info["model_type"] == "bs_roformer"
    loader._create_bs_roformer.assert_called_once()
    loader._create_mel_band_roformer.assert_not_called()


@patch("torch.load", return_value={})
@patch("audio_separator.separator.uvr_lib_v5.roformer.mel_band_roformer.MelBandRoformer")
def test_legacy_fallback_keeps_family_resolved_from_params_config(mel_constructor, _torch_load):
    """Legacy fallback must not run a second, narrower family detector."""
    loader = RoformerLoader()
    loader._load_with_new_implementation = Mock(side_effect=RuntimeError("new loader failed"))
    mel_constructor.return_value = Mock()
    config = {
        "params": {
            "dim": 384,
            "depth": 12,
            "num_bands": 60,
        }
    }

    result = loader.load_model("/tmp/model.ckpt", config)

    assert result.success is True
    assert result.implementation_used is ImplementationVersion.FALLBACK
    mel_constructor.assert_called_once_with(dim=384, depth=12, num_bands=60)


@patch("torch.load", return_value={})
@patch("audio_separator.separator.uvr_lib_v5.roformer.mel_band_roformer.MelBandRoformer")
def test_legacy_fallback_uses_normalized_parameter_aliases(mel_constructor, _torch_load):
    """Alias support must remain valid if the new loader falls back to a direct constructor."""
    loader = RoformerLoader()
    loader._load_with_new_implementation = Mock(side_effect=RuntimeError("new loader failed"))
    mel_constructor.return_value = Mock()
    config = {
        "params": {
            "dim": 384,
            "depth": 12,
            "n_mels": 60,
        }
    }

    result = loader.load_model("/tmp/model.ckpt", config)

    assert result.success is True
    mel_constructor.assert_called_once_with(dim=384, depth=12, num_bands=60)


@patch("torch.load", return_value={})
@patch("audio_separator.separator.uvr_lib_v5.roformer.mel_band_roformer.MelBandRoformer")
def test_legacy_fallback_merges_supported_nested_sections(mel_constructor, _torch_load):
    """Fallback constructor input must match the normalizer's nested section support."""
    loader = RoformerLoader()
    loader._load_with_new_implementation = Mock(side_effect=RuntimeError("new loader failed"))
    mel_constructor.return_value = Mock()
    config = {
        "architecture": {
            "dim": 384,
            "depth": 12,
        },
        "params": {
            "num_bands": 60,
        },
    }

    result = loader.load_model("/tmp/model.ckpt", config)

    assert result.success is True
    mel_constructor.assert_called_once_with(dim=384, depth=12, num_bands=60)
