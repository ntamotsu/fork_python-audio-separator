import inspect

from audio_separator.separator import Separator


def test_new_constructor_option_is_appended_after_legacy_positional_parameters():
    legacy_parameters = [
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
        "info_only",
    ]

    assert list(inspect.signature(Separator).parameters) == [*legacy_parameters, "use_torch_compile"]
