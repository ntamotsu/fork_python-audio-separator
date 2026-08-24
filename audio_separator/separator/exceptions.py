"""Public exceptions raised by audio validation and separation output handling."""


def _restore_audio_export_error(message, path, backend):
    return AudioExportError(message, path=path, backend=backend)


def _restore_batch_separation_error(successful_files, failures):
    return BatchSeparationError(successful_files, failures)


class InvalidAudioDataError(ValueError):
    """Raised when a separator produces audio that cannot be exported safely."""


class AudioExportError(RuntimeError):
    """Raised when an audio backend or filesystem cannot publish an output file."""

    def __init__(self, message, *, path, backend):
        super().__init__(message)
        self.path = path
        self.backend = backend

    def __reduce__(self):
        return _restore_audio_export_error, (str(self), self.path, self.backend)


class BatchSeparationError(RuntimeError):
    """Raised after a batch finishes with failed inputs.

    ``successful_files`` contains files from inputs that returned normally. A
    fully published stem from an input that later failed is left on disk but is
    intentionally excluded because that input did not complete its full stem set.
    """

    def __init__(self, successful_files, failures):
        self.successful_files = list(successful_files)
        self.failures = list(failures.items()) if isinstance(failures, dict) else list(failures)
        failure_details = "; ".join(f"{path}: {error}" for path, error in self.failures)
        super().__init__(f"Separation failed for {len(self.failures)} input(s): {failure_details}")

    def __reduce__(self):
        return _restore_batch_separation_error, (self.successful_files, self.failures)
