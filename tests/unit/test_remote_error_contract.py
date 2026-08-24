"""Remote error serialization contracts."""

from audio_separator.remote.error_utils import format_remote_error
from audio_separator.separator.exceptions import AudioExportError, BatchSeparationError


def test_remote_error_text_includes_the_original_backend_cause():
    backend_error = OSError("ffmpeg exited with code 1")
    export_error = AudioExportError("Failed to export stem", path="song_(Vocals).m4a", backend="pydub")
    export_error.__cause__ = backend_error

    error_text = format_remote_error(export_error)

    assert "Failed to export stem" in error_text
    assert "ffmpeg exited with code 1" in error_text


def test_remote_batch_error_text_traverses_each_failure_cause():
    backend_error = OSError("libsndfile write failed")
    export_error = AudioExportError("Failed to export stem", path="song_(Vocals).wav", backend="soundfile")
    export_error.__cause__ = backend_error
    batch_error = BatchSeparationError([], [("song.wav", export_error)])

    error_text = format_remote_error(batch_error)

    assert "libsndfile write failed" in error_text
