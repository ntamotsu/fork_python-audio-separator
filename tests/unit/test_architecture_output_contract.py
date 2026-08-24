"""Architecture-level contracts for propagating final audio writer failures."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from audio_separator.separator.architectures.demucs_separator import DemucsSeparator
from audio_separator.separator.architectures.mdx_separator import MDXSeparator
from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator
from audio_separator.separator.architectures.vr_separator import VRSeparator
from audio_separator.separator.exceptions import AudioExportError


def architecture_double(separator_class):
    separator = Mock(spec=separator_class)
    separator.logger = logging.getLogger(__name__)
    separator.normalization_threshold = 0.9
    separator.amplification_threshold = 0.0
    separator.output_single_stem = None
    return separator


def test_mdx_does_not_return_planned_path_when_final_writer_fails():
    separator = architecture_double(MDXSeparator)
    separator.prepare_mix.return_value = np.zeros((2, 32), dtype=np.float32)
    separator.demix.side_effect = [np.zeros((2, 32), dtype=np.float32), np.zeros((2, 32), dtype=np.float32)]
    separator.primary_source = None
    separator.secondary_source = None
    separator.primary_stem_name = "Vocals"
    separator.secondary_stem_name = "Instrumental"
    separator.compensate = 1.0
    separator.invert_using_spec = False
    separator.get_stem_output_path.side_effect = ["input_(Instrumental).wav", "input_(Vocals).wav"]
    writer_error = AudioExportError("writer failed", path="input_(Instrumental).wav", backend="pydub")
    separator.final_process.side_effect = writer_error

    with pytest.raises(AudioExportError) as raised:
        MDXSeparator.separate(separator, "input.wav")

    assert raised.value is writer_error


def test_mdxc_multistem_does_not_return_planned_path_when_final_writer_fails():
    separator = architecture_double(MDXCSeparator)
    separator.prepare_mix.return_value = np.zeros((2, 32), dtype=np.float32)
    separator.sample_rate = 44100
    separator.override_model_segment_size = True
    separator.process_all_stems = True
    separator.primary_source = None
    separator.secondary_source = None
    separator.model_data_cfgdict = SimpleNamespace(
        training=SimpleNamespace(target_instrument=None, instruments=["Vocals", "Drums", "Bass"])
    )
    separator.demix.return_value = {
        "Vocals": np.zeros((2, 32), dtype=np.float32),
        "Drums": np.zeros((2, 32), dtype=np.float32),
        "Bass": np.zeros((2, 32), dtype=np.float32),
    }
    separator.get_stem_output_path.return_value = "input_(Vocals).wav"
    writer_error = AudioExportError("writer failed", path="input_(Vocals).wav", backend="soundfile")
    separator.final_process.side_effect = writer_error

    with pytest.raises(AudioExportError) as raised:
        MDXCSeparator.separate(separator, "input.wav")

    assert raised.value is writer_error


def test_demucs_does_not_return_planned_path_when_final_writer_fails():
    separator = architecture_double(DemucsSeparator)
    separator.output_single_stem = "Vocals"
    separator.prepare_mix.return_value = np.zeros((2, 32), dtype=np.float32)
    separator.model_path = "/models/htdemucs.yaml"
    separator.segment_size = "Default"
    separator.torch_device = Mock()
    separator.demix_demucs.return_value = np.zeros((4, 2, 32), dtype=np.float32)
    separator.get_stem_output_path.return_value = "input_(Vocals).wav"
    writer_error = AudioExportError("writer failed", path="input_(Vocals).wav", backend="pydub")
    separator.final_process.side_effect = writer_error
    model = Mock()

    with patch("audio_separator.separator.architectures.demucs_separator.HDemucs"), patch(
        "audio_separator.separator.architectures.demucs_separator.get_demucs_model", return_value=model
    ), patch("audio_separator.separator.architectures.demucs_separator.demucs_segments", return_value=model):
        with pytest.raises(AudioExportError) as raised:
            DemucsSeparator.separate(separator, "input.wav")

    assert raised.value is writer_error


def test_vr_does_not_return_planned_path_when_final_writer_fails(tmp_path):
    separator = architecture_double(VRSeparator)
    separator.output_single_stem = "Vocals"
    model_path = tmp_path / "vr.pth"
    model_path.write_bytes(b"model")
    separator.model_path = str(model_path)
    separator.model_params = SimpleNamespace(param={"bins": 1})
    separator.model_capacity = (1, 1)
    separator.is_vr_51_model = False
    separator.torch_device = Mock()
    separator.aggressiveness = 0
    separator.loading_mix.return_value = np.zeros((2, 32), dtype=np.float32)
    separator.inference_vr.return_value = (np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32))
    separator.spec_to_wav.return_value = np.zeros((2, 32), dtype=np.float32)
    separator.model_samplerate = 44100
    separator.primary_stem_name = "Vocals"
    separator.secondary_stem_name = "Instrumental"
    separator.model_name = "vr-test"
    separator.get_stem_output_path.return_value = "input_(Vocals).wav"
    writer_error = AudioExportError("writer failed", path="input_(Vocals).wav", backend="pydub")
    separator.final_process.side_effect = writer_error
    model = Mock()

    with patch("soundfile.info", return_value=SimpleNamespace(subtype="PCM_16")), patch(
        "audio_separator.separator.architectures.vr_separator.nets.determine_model_capacity", return_value=model
    ), patch("audio_separator.separator.architectures.vr_separator.torch.load", return_value={}):
        with pytest.raises(AudioExportError) as raised:
            VRSeparator.separate(separator, "input.wav")

    assert raised.value is writer_error
