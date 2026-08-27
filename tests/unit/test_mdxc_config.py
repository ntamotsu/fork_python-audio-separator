from unittest.mock import Mock, patch

import pytest
import torch
from ml_collections import ConfigDict

from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator
from audio_separator.separator.common_separator import CommonSeparator


def _make_separator(arch_config, inference_config=None, is_roformer_model=True):
    separator = MDXCSeparator.__new__(MDXCSeparator)
    separator.logger = Mock()
    separator.is_roformer_model = is_roformer_model
    separator.model_data = {
        "inference": inference_config or {},
        "training": {"target_instrument": "Vocals"},
    }
    separator.torch_device = torch.device("cpu")
    separator.torch_device_cpu = torch.device("cpu")

    def load_model():
        separator.model_data_cfgdict = ConfigDict(separator.model_data)

    with patch.object(CommonSeparator, "__init__", return_value=None), patch.object(MDXCSeparator, "load_model", side_effect=load_model):
        MDXCSeparator.__init__(separator, {}, arch_config)

    return separator


def test_mdxc_inference_defaults_come_from_model_config():
    separator = _make_separator({"overlap": None, "batch_size": None}, {"num_overlap": 2, "batch_size": 4})

    assert separator.overlap == 2
    assert separator.batch_size == 4


def test_mdxc_explicit_inference_options_override_model_config():
    separator = _make_separator({"overlap": 8, "batch_size": 1}, {"num_overlap": 2, "batch_size": 4})

    assert separator.overlap == 8
    assert separator.batch_size == 1


def test_mdxc_inference_defaults_fall_back_for_older_configs():
    separator = _make_separator({"overlap": None, "batch_size": None})

    assert separator.overlap == 8
    assert separator.batch_size == 1


def test_mdxc_null_inference_values_use_fallbacks():
    separator = _make_separator({"overlap": None, "batch_size": None}, {"num_overlap": None, "batch_size": None})

    assert separator.overlap == 8
    assert separator.batch_size == 1


def test_mdxc_step_size_seconds_is_stored():
    separator = _make_separator({"overlap": None, "batch_size": None, "step_size_seconds": 1.5})

    assert separator.step_size_seconds == 1.5


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_mdxc_rejects_invalid_step_size_seconds(value):
    with pytest.raises(ValueError, match="MDXC step size seconds must be finite and greater than zero"):
        _make_separator({"overlap": None, "batch_size": None, "step_size_seconds": value})


def test_mdxc_step_size_seconds_warns_when_overlap_is_also_explicit():
    separator = _make_separator({"overlap": 2, "batch_size": None, "step_size_seconds": 1.5})

    separator.logger.warning.assert_called_once_with(
        "Both MDXC overlap and step size seconds were specified; step size seconds takes precedence for RoFormer models."
    )


def test_mdxc_step_size_seconds_overrides_invalid_explicit_overlap_for_roformer():
    separator = _make_separator({"overlap": 0, "batch_size": None, "step_size_seconds": 1.5})

    assert separator.step_size_seconds == 1.5


def test_mdxc_step_size_seconds_is_ignored_for_non_roformer_models():
    separator = _make_separator(
        {"overlap": None, "batch_size": None, "step_size_seconds": 1.5},
        is_roformer_model=False,
    )

    separator.logger.warning.assert_called_once_with(
        "MDXC step size seconds is only supported for RoFormer models; ignoring it for this model."
    )


@pytest.mark.parametrize(
    ("arch_config", "inference_config", "message"),
    [
        ({"overlap": 0, "batch_size": 1}, {"num_overlap": 2}, "MDXC overlap"),
        ({"overlap": -1, "batch_size": 1}, {"num_overlap": 2}, "MDXC overlap"),
        ({"overlap": None, "batch_size": 1}, {"num_overlap": 0}, "MDXC overlap"),
        ({"overlap": None, "batch_size": 1}, {"num_overlap": -1}, "MDXC overlap"),
        ({"overlap": 2, "batch_size": 0}, {"batch_size": 1}, "MDXC batch size"),
        ({"overlap": 2, "batch_size": -1}, {"batch_size": 1}, "MDXC batch size"),
        ({"overlap": 2, "batch_size": None}, {"batch_size": 0}, "MDXC batch size"),
        ({"overlap": 2, "batch_size": None}, {"batch_size": -1}, "MDXC batch size"),
    ],
)
def test_mdxc_rejects_non_positive_inference_values(arch_config, inference_config, message):
    with pytest.raises(ValueError, match=message):
        _make_separator(arch_config, inference_config)
