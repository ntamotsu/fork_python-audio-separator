import inspect

from audio_separator.separator import Separator


def test_execution_options_are_appended_to_constructor_signature():
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
        "use_native_fp16",
    ]
    assert inspect.signature(Separator.__init__).parameters["use_torch_compile"].default is False
    assert inspect.signature(Separator.__init__).parameters["use_native_fp16"].default is False


def test_mdxc_step_size_seconds_default_is_disabled():
    mdxc_params = inspect.signature(Separator.__init__).parameters["mdxc_params"].default

    assert mdxc_params["step_size_seconds"] is None


def test_mdxc_step_size_seconds_is_preserved_in_separator_config():
    separator = Separator(mdxc_params={"step_size_seconds": 1.5}, info_only=True)

    assert separator.arch_specific_params["MDXC"]["step_size_seconds"] == 1.5
