import inspect

from audio_separator.separator import Separator


def test_use_torch_compile_is_appended_to_constructor_signature():
    parameters = list(inspect.signature(Separator.__init__).parameters)

    assert parameters == [
        "self",
        "log_level",
        "log_formatter",
        "model_file_dir",
        "output_dir",
        "output_format",
        "output_bitrate",
        "normalization_threshold",
        "amplification_threshold",
        "output_single_stem",
        "invert_using_spec",
        "sample_rate",
        "use_soundfile",
        "use_autocast",
        "use_directml",
        "chunk_duration",
        "mdx_params",
        "vr_params",
        "demucs_params",
        "mdxc_params",
        "ensemble_algorithm",
        "ensemble_weights",
        "ensemble_preset",
        "info_only",
        "use_torch_compile",
    ]
    assert inspect.signature(Separator.__init__).parameters["use_torch_compile"].default is False
