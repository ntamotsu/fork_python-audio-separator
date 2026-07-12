from unittest.mock import Mock, patch

import pytest
import torch

from audio_separator.separator.architectures.vr_separator import VRSeparator
from audio_separator.separator.separator import Separator


def test_separator_reuses_same_model_instance():
    separator = object.__new__(Separator)
    separator.logger = Mock()
    separator.model_instance = object()
    separator._loaded_model_filename = "model.pth"

    with patch.object(separator, "download_model_files") as download_model_files:
        separator.load_model("model.pth")

    download_model_files.assert_not_called()


def test_separator_can_force_same_model_reload():
    separator = object.__new__(Separator)
    separator.logger = Mock()
    separator.model_instance = object()
    separator._loaded_model_filename = "model.pth"

    with patch.object(separator, "download_model_files", side_effect=RuntimeError("reload attempted")):
        with pytest.raises(RuntimeError, match="reload attempted"):
            separator.load_model("model.pth", force_reload=True)


def test_vr_separator_reuses_loaded_weights():
    separator = object.__new__(VRSeparator)
    separator.logger = Mock()
    separator.model_run = torch.nn.Identity()

    with patch("audio_separator.separator.architectures.vr_separator.nets.determine_model_capacity") as determine_model_capacity:
        separator._ensure_model_loaded(nn_arch_size=123821)

    determine_model_capacity.assert_not_called()
