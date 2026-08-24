"""Public separation error propagation and batch aggregation contracts."""

import logging
from unittest.mock import Mock, patch

import numpy as np
import pytest

from audio_separator.separator.exceptions import AudioExportError, BatchSeparationError, InvalidAudioDataError
from audio_separator.separator.separator import Separator


def separator_double():
    separator = Mock(spec=Separator)
    separator.torch_device = Mock()
    separator.model_instance = Mock()
    separator.model_filename = "model.ckpt"
    separator.logger = logging.getLogger(__name__)
    return separator


def test_single_file_separation_fails_fast_with_original_error():
    separator = separator_double()
    export_error = AudioExportError("writer failed", path="bad_(Vocals).wav", backend="soundfile")
    separator._separate_file.side_effect = export_error

    with pytest.raises(AudioExportError) as raised:
        Separator.separate(separator, "bad.wav")

    assert raised.value is export_error


def test_batch_processes_every_file_then_raises_aggregate_error():
    separator = separator_double()
    export_error = AudioExportError("writer failed", path="bad_(Vocals).wav", backend="soundfile")
    separator._separate_file.side_effect = [["good_(Vocals).wav"], export_error, ["later_(Vocals).wav"]]

    with pytest.raises(BatchSeparationError) as raised:
        Separator.separate(separator, ["good.wav", "bad.wav", "later.wav"])

    assert raised.value.successful_files == ["good_(Vocals).wav", "later_(Vocals).wav"]
    assert raised.value.failures == [("bad.wav", export_error)]
    assert separator._separate_file.call_count == 3


def test_single_file_cleanup_runs_when_model_separation_fails():
    separator = separator_double()
    separator.chunk_duration = None
    separator.use_autocast = False
    separator.normalization_threshold = 0.9
    separator.amplification_threshold = 0.0
    separator.print_uvr_vip_message = Mock()
    model_error = RuntimeError("inference failed")
    separator.model_instance.separate.side_effect = model_error

    with pytest.raises(RuntimeError) as raised:
        Separator._separate_file(separator, "bad.wav")

    assert raised.value is model_error
    separator.model_instance.clear_gpu_cache.assert_called_once_with()
    separator.model_instance.clear_file_specific_paths.assert_called_once_with()


def test_cleanup_failures_do_not_mask_writer_error_or_skip_later_cleanup():
    separator = separator_double()
    separator.chunk_duration = None
    separator.use_autocast = False
    separator.normalization_threshold = 0.9
    separator.amplification_threshold = 0.0
    separator.print_uvr_vip_message = Mock()
    writer_error = AudioExportError("writer failed", path="bad_(Vocals).wav", backend="soundfile")
    separator.model_instance.separate.side_effect = writer_error
    separator.model_instance.clear_gpu_cache.side_effect = RuntimeError("cache cleanup failed")

    with pytest.raises(AudioExportError) as raised:
        Separator._separate_file(separator, "bad.wav")

    assert raised.value is writer_error
    separator.model_instance.clear_gpu_cache.assert_called_once_with()
    separator.model_instance.clear_file_specific_paths.assert_called_once_with()


@patch("audio_separator.separator.audio_chunking.AudioChunker")
def test_chunked_separation_rejects_a_stem_missing_from_one_chunk(chunker_class, tmp_path):
    separator = separator_double()
    separator.output_dir = str(tmp_path)
    separator.output_format = "WAV"
    separator.chunk_duration = 10.0
    separator.model_instance.output_dir = str(tmp_path)
    chunker = chunker_class.return_value
    chunker.split_audio.return_value = [str(tmp_path / "chunk_0000.wav"), str(tmp_path / "chunk_0001.wav")]
    separator._separate_file.side_effect = [
        ["chunk_0000_(Vocals).wav", "chunk_0000_(Instrumental).wav"],
        ["chunk_0001_(Vocals).wav"],
    ]

    with pytest.raises(InvalidAudioDataError, match="missing"):
        Separator._process_with_chunking(separator, str(tmp_path / "input.wav"))

    chunker.merge_chunks.assert_not_called()


@patch("audio_separator.separator.audio_chunking.AudioChunker")
def test_chunked_separation_rejects_duplicate_stem_outputs_in_one_chunk(chunker_class, tmp_path):
    separator = separator_double()
    separator.output_dir = str(tmp_path)
    separator.output_format = "WAV"
    separator.chunk_duration = 10.0
    separator.model_instance.output_dir = str(tmp_path)
    chunker = chunker_class.return_value
    chunker.split_audio.return_value = [str(tmp_path / "chunk_0000.wav")]
    separator._separate_file.return_value = ["first_(Vocals).wav", "second_(Vocals).wav"]

    with pytest.raises(InvalidAudioDataError, match="duplicate"):
        Separator._process_with_chunking(separator, str(tmp_path / "input.wav"))

    chunker.merge_chunks.assert_not_called()


@patch("audio_separator.separator.audio_chunking.AudioChunker")
def test_chunked_separation_uses_stable_indexes_when_stem_names_are_absent(chunker_class, tmp_path):
    separator = separator_double()
    separator.output_dir = str(tmp_path)
    separator.output_format = "WAV"
    separator.chunk_duration = 10.0
    separator.model_instance.output_dir = str(tmp_path)
    chunker = chunker_class.return_value
    chunker.split_audio.return_value = [str(tmp_path / "chunk_0000.wav"), str(tmp_path / "chunk_0001.wav")]
    separator._separate_file.side_effect = [
        ["chunk_0000-first.wav", "chunk_0000-second.wav"],
        ["chunk_0001-first.wav", "chunk_0001-second.wav"],
    ]

    output_files = Separator._process_with_chunking(separator, str(tmp_path / "input.wav"))

    assert len(output_files) == 2
    assert chunker.merge_chunks.call_count == 2


def test_ensemble_batch_processes_every_file_then_raises_aggregate_error():
    separator = separator_double()
    separator.model_filename = ["model-a.ckpt", "model-b.ckpt"]
    export_error = AudioExportError("ensemble writer failed", path="bad_(Vocals).wav", backend="pydub")
    separator._separate_ensemble.side_effect = [["good_(Vocals).wav"], export_error, ["later_(Vocals).wav"]]

    with pytest.raises(BatchSeparationError) as raised:
        Separator.separate(separator, ["good.wav", "bad.wav", "later.wav"])

    assert raised.value.successful_files == ["good_(Vocals).wav", "later_(Vocals).wav"]
    assert raised.value.failures == [("bad.wav", export_error)]
    assert separator._separate_ensemble.call_count == 3


def test_batch_error_message_preserves_each_original_failure_message():
    export_error = AudioExportError("ffmpeg encoder exploded", path="bad_(Vocals).m4a", backend="pydub")

    error = BatchSeparationError(["good_(Vocals).wav"], [("bad.wav", export_error)])

    assert "bad.wav: ffmpeg encoder exploded" in str(error)


def test_batch_error_preserves_duplicate_input_paths_in_order():
    separator = separator_double()
    first_error = RuntimeError("first failure")
    second_error = RuntimeError("second failure")
    separator._separate_file.side_effect = [first_error, second_error]

    with pytest.raises(BatchSeparationError) as raised:
        Separator.separate(separator, ["same.wav", "same.wav"])

    assert raised.value.failures == [("same.wav", first_error), ("same.wav", second_error)]


def test_directory_scan_failure_is_aggregated_after_accessible_files_are_processed():
    separator = separator_double()
    scan_error = PermissionError("directory is unreadable")
    scan_error.filename = "/music/locked"

    def walk_directory(_path, onerror=None):
        yield "/music", [], ["good.wav"]
        if onerror is not None:
            onerror(scan_error)

    separator._separate_file.return_value = ["good_(Vocals).wav"]
    with patch("audio_separator.separator.separator.os.path.isdir", return_value=True), patch(
        "audio_separator.separator.separator.os.walk", side_effect=walk_directory
    ):
        with pytest.raises(BatchSeparationError) as raised:
            Separator.separate(separator, "/music")

    assert raised.value.successful_files == ["good_(Vocals).wav"]
    assert raised.value.failures == [("/music/locked", scan_error)]


def test_completed_stem_from_failed_input_remains_on_disk_but_is_not_a_batch_success(tmp_path):
    separator = separator_double()
    completed_stem = tmp_path / "bad_(Vocals).wav"
    writer_error = AudioExportError("second stem failed", path=str(tmp_path / "bad_(Instrumental).wav"), backend="pydub")

    def fail_after_first_stem(*_args, **_kwargs):
        completed_stem.write_bytes(b"complete audio file")
        raise writer_error

    separator._separate_file.side_effect = fail_after_first_stem

    with pytest.raises(BatchSeparationError) as raised:
        Separator.separate(separator, ["bad.wav"])

    assert raised.value.successful_files == []
    assert raised.value.failures == [("bad.wav", writer_error)]
    assert completed_stem.read_bytes() == b"complete audio file"


@patch("audio_separator.separator.separator.Ensembler")
@patch("audio_separator.separator.separator.librosa.load", return_value=(np.zeros((2, 32), dtype=np.float32), 44100))
def test_ensemble_fallback_writer_preserves_soundfile_error(load_audio, ensembler_class, tmp_path):
    separator = separator_double()
    separator.model_filename = ["model-a.ckpt", "model-b.ckpt"]
    separator.model_filenames = ["model-a.ckpt", "model-b.ckpt"]
    separator.model_instance = None
    separator.output_dir = str(tmp_path)
    separator.output_format = "WAV"
    separator.sample_rate = 44100
    separator.normalization_threshold = 0.9
    separator.amplification_threshold = 0.0
    separator.ensemble_algorithm = "avg_wave"
    separator.ensemble_weights = None
    separator.ensemble_preset = None
    separator._separate_file.return_value = ["input_(Vocals).wav"]
    ensembler_class.return_value.ensemble.return_value = np.zeros((2, 32), dtype=np.float32)
    backend_error = OSError("soundfile failed")

    with patch("soundfile.write", side_effect=backend_error):
        with pytest.raises(AudioExportError) as raised:
            Separator._separate_ensemble(separator, "input.wav")

    assert raised.value.backend == "soundfile"
    assert raised.value.__cause__ is backend_error
    assert raised.value.path.endswith(".wav")


@patch("audio_separator.separator.separator.Ensembler")
@patch("audio_separator.separator.separator.librosa.load", return_value=(np.zeros((2, 32), dtype=np.float32), 44100))
def test_ensemble_fallback_writer_rejects_nonfinite_audio_before_backend(load_audio, ensembler_class, tmp_path):
    separator = separator_double()
    separator.model_filename = ["model-a.ckpt", "model-b.ckpt"]
    separator.model_filenames = ["model-a.ckpt", "model-b.ckpt"]
    separator.model_instance = None
    separator.output_dir = str(tmp_path)
    separator.output_format = "WAV"
    separator.sample_rate = 44100
    separator.normalization_threshold = 0.9
    separator.amplification_threshold = 0.0
    separator.ensemble_algorithm = "avg_wave"
    separator.ensemble_weights = None
    separator.ensemble_preset = None
    separator._separate_file.return_value = ["input_(Vocals).wav"]
    ensembler_class.return_value.ensemble.return_value = np.full((2, 32), np.nan, dtype=np.float32)

    with patch("soundfile.write") as write_audio:
        with pytest.raises(InvalidAudioDataError, match="finite"):
            Separator._separate_ensemble(separator, "input.wav")

    write_audio.assert_not_called()


@patch("audio_separator.separator.separator.Ensembler")
@patch("audio_separator.separator.separator.librosa.load", return_value=(np.zeros((2, 32), dtype=np.float32), 44100))
def test_ensemble_model_writer_failure_is_not_returned_as_success(load_audio, ensembler_class, tmp_path):
    separator = separator_double()
    separator.model_filename = ["model-a.ckpt", "model-b.ckpt"]
    separator.model_filenames = ["model-a.ckpt", "model-b.ckpt"]
    separator.output_dir = str(tmp_path)
    separator.output_format = "WAV"
    separator.sample_rate = 44100
    separator.ensemble_algorithm = "avg_wave"
    separator.ensemble_weights = None
    separator.ensemble_preset = None
    separator._separate_file.return_value = ["input_(Vocals).wav"]
    ensembler_class.return_value.ensemble.return_value = np.zeros((2, 32), dtype=np.float32)
    writer_error = AudioExportError("ensemble writer failed", path=str(tmp_path / "output.wav"), backend="pydub")
    separator.model_instance.write_audio.side_effect = writer_error

    with pytest.raises(AudioExportError) as raised:
        Separator._separate_ensemble(separator, "input.wav")

    assert raised.value is writer_error
