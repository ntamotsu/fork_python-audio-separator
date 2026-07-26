from unittest.mock import Mock, patch

import torch

from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator


def _separator(*, use_torch_compile=True, native_fp16=True):
    separator = object.__new__(MDXCSeparator)
    separator.logger = Mock()
    separator.use_torch_compile = use_torch_compile
    separator.is_native_mps_fp16 = native_fp16
    separator.model_run = Mock()
    separator.model_run.layers = torch.nn.ModuleList([torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])])
    return separator


def test_regional_compile_wraps_repeated_transformers():
    separator = _separator()

    with patch.object(torch.nn.Module, "compile", autospec=True) as compile_module:
        separator._configure_model_compilation()

    assert separator.is_torch_compiled is True
    assert compile_module.call_count == 2
    separator.logger.warning.assert_not_called()


def test_regional_compile_requires_native_mps_fp16():
    separator = _separator(native_fp16=False)

    with patch.object(torch.nn.Module, "compile", autospec=True) as compile_module:
        separator._configure_model_compilation()

    assert separator.is_torch_compiled is False
    compile_module.assert_not_called()
    separator.logger.warning.assert_called_once()


def test_disabled_regional_compile_is_silent():
    separator = _separator(use_torch_compile=False)

    with patch.object(torch.nn.Module, "compile", autospec=True) as compile_module:
        separator._configure_model_compilation()

    assert separator.is_torch_compiled is False
    compile_module.assert_not_called()
    separator.logger.warning.assert_not_called()


def test_regional_compile_falls_back_to_eager_when_compilation_fails():
    separator = _separator()

    with patch.object(torch.nn.Module, "compile", autospec=True, side_effect=RuntimeError("unsupported")):
        separator._configure_model_compilation()

    assert separator.is_torch_compiled is False
    separator.logger.warning.assert_called_once()
