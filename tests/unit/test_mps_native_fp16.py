from contextlib import nullcontext
from unittest.mock import Mock, patch

import torch

from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator
from audio_separator.separator.separator import Separator
from audio_separator.separator.uvr_lib_v5.roformer.mel_band_roformer import RMSNorm


class MelBandRoformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)


def make_separator(device="mps", use_autocast=True):
    separator = object.__new__(MDXCSeparator)
    separator.logger = Mock()
    separator.model_run = MelBandRoformer()
    separator.roformer_model_type = "mel_band_roformer"
    separator.torch_device = torch.device(device)
    separator.use_autocast = use_autocast
    separator.use_torch_compile = False
    separator.is_native_mps_fp16 = False
    return separator


def test_mel_band_roformer_uses_native_fp16_for_mps_autocast():
    separator = make_separator()

    separator._configure_model_precision()

    assert separator.is_native_mps_fp16 is True
    assert next(separator.model_run.parameters()).dtype == torch.float16


def test_native_fp16_is_not_enabled_on_cpu():
    separator = make_separator(device="cpu")

    separator._configure_model_precision()

    assert separator.is_native_mps_fp16 is False
    assert next(separator.model_run.parameters()).dtype == torch.float32


def test_native_fp16_respects_disabled_autocast():
    separator = make_separator(use_autocast=False)

    separator._configure_model_precision()

    assert separator.is_native_mps_fp16 is False
    assert next(separator.model_run.parameters()).dtype == torch.float32


def test_native_fp16_rms_norm_keeps_silence_finite():
    norm = RMSNorm(8).half()

    output = norm(torch.zeros(2, 8, dtype=torch.float16))

    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0


def test_separator_does_not_wrap_native_fp16_model_in_autocast():
    separator = object.__new__(Separator)
    separator.chunk_duration = None
    separator.logger = Mock()
    separator.normalization_threshold = 1.0
    separator.amplification_threshold = 0.0
    separator.use_autocast = True
    separator.torch_device = torch.device("mps")
    separator.model_instance = Mock(is_native_mps_fp16=True)
    separator.model_instance.separate.return_value = ["output.wav"]
    separator.print_uvr_vip_message = Mock()

    with patch("audio_separator.separator.separator.autocast_mode.autocast", return_value=nullcontext()) as autocast:
        output_files = separator._separate_file("input.wav")

    assert output_files == ["output.wav"]
    autocast.assert_not_called()


def test_regional_compile_wraps_repeated_transformers():
    separator = make_separator()
    separator.use_torch_compile = True
    separator._configure_model_precision()
    separator.model_run.layers = torch.nn.ModuleList(
        [torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])]
    )

    with patch.object(torch.nn.Module, "compile", autospec=True) as compile_module:
        separator._configure_model_compilation()

    assert separator.is_torch_compiled is True
    assert compile_module.call_count == 2


def test_regional_compile_requires_native_mps_fp16():
    separator = make_separator(device="cpu")
    separator.use_torch_compile = True
    separator._configure_model_precision()
    separator.model_run.layers = torch.nn.ModuleList(
        [torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])]
    )

    with patch.object(torch.nn.Module, "compile", autospec=True) as compile_module:
        separator._configure_model_compilation()

    assert separator.is_torch_compiled is False
    compile_module.assert_not_called()
