import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest
import soundfile as sf

from tools import benchmark_apple_backends


class FakeRuntime:
    def __init__(self, name: str) -> None:
        self.name = name
        self.cleanup_count = 0

    def synchronize(self) -> None:
        pass

    def seed(self, _seed: int) -> None:
        pass

    def memory(self) -> dict[str, int]:
        return {"active_bytes": 0}

    def reset_peak_memory(self) -> None:
        pass

    def cleanup(self) -> None:
        self.cleanup_count += 1


class FakeSeparator:
    """順序とpath形式が不定なbackendを再現する。"""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.loaded_models: list[str] = []
        self.separation_calls: list[tuple[Path, dict[str, str]]] = []
        self.last_perf_metrics = {"inference_s": np.float32(0.25)}

    def load_model(self, model_filename: str) -> None:
        self.loaded_models.append(model_filename)

    def separate(self, source: str, custom_output_names: dict[str, str]) -> list[str]:
        self.separation_calls.append((Path(source), custom_output_names.copy()))
        returned_paths: list[str] = []
        for index, output_name in enumerate(custom_output_names.values()):
            output_path = self.output_dir / f"{output_name}.wav"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, np.zeros((32, 2), dtype=np.float32), 44_100)
            returned_paths.append(str(output_path) if index % 2 else output_path.name)
        return list(reversed(returned_paths))


def make_args(
    tmp_path: Path,
    *,
    backend: str = "mlx",
    mlx_deecho_reuse: bool = True,
) -> argparse.Namespace:
    audio_file = tmp_path / "input.wav"
    sf.write(audio_file, np.zeros((32, 2), dtype=np.float32), 44_100)
    return argparse.Namespace(
        audio_file=audio_file,
        backend=backend,
        mode="chain",
        models=list(benchmark_apple_backends.MODEL_SPECS),
        model_dir=tmp_path,
        output_dir=tmp_path / "output",
        json_output=None,
        warmup=0,
        repeats=1,
        seed=0,
        torch_compile=False,
        mlx_speed_mode="default",
        mlx_cache_clear_policy="aggressive",
        mlx_write_workers=1,
        mlx_demucs_batch_size=1,
        mlx_save_converted_safetensors=False,
        mlx_deecho_reuse=mlx_deecho_reuse,
        mdxc_segment_size=1101,
        mdxc_overlap=8,
        vr_window_size=320,
        vr_aggression=50,
        vr_tta=True,
        demucs_shifts=2,
        demucs_overlap=0.25,
    )


def test_map_named_outputs_is_independent_of_order_case_and_path_style(tmp_path):
    output_dir = tmp_path / "outputs"
    absolute_instrumental = tmp_path / "elsewhere" / "Kim_Instrumental.WAV"
    names = {"other": "kim_instrumental", "vocals": "kim_vocals"}

    mapped = benchmark_apple_backends.map_named_outputs(
        output_dir,
        names,
        ["kim_vocals.wav", str(absolute_instrumental)],
    )

    assert mapped == {
        "vocals": (output_dir / "kim_vocals.wav").resolve(),
        "other": absolute_instrumental.resolve(),
    }


@pytest.mark.parametrize(
    ("output_files", "message"),
    [
        (["vocals.wav"], "出力stemが不足しています: other"),
        (["vocals.wav", "vocals.wav", "other.wav"], "出力stemが重複しています: vocals"),
        (["vocals.wav", "other.wav", "noise.wav"], "想定外の出力です: noise.wav"),
    ],
)
def test_map_named_outputs_rejects_incomplete_or_ambiguous_results(
    tmp_path,
    output_files,
    message,
):
    names = {"vocals": "vocals", "other": "other"}

    with pytest.raises(RuntimeError, match=message):
        benchmark_apple_backends.map_named_outputs(tmp_path, names, output_files)


def test_parse_args_rejects_model_filter_in_chain_mode(tmp_path, capsys):
    audio_file = tmp_path / "input.wav"
    sf.write(audio_file, np.zeros((32, 2), dtype=np.float32), 44_100)

    with pytest.raises(SystemExit, match="2"):
        benchmark_apple_backends.parse_args(
            [
                str(audio_file),
                "--backend",
                "mps",
                "--mode",
                "chain",
                "--model",
                "kim",
                "--model-dir",
                str(tmp_path),
            ]
        )

    assert "--modelはmodelsモードだけで指定できます" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("reuse", "expected_deecho_loads", "load_model_called", "reuse_strategy"),
    [(True, 1, False, "runner_skip"), (False, 2, True, None)],
)
def test_mlx_chain_connects_named_stems_and_controls_deecho_reload(
    tmp_path,
    reuse,
    expected_deecho_loads,
    load_model_called,
    reuse_strategy,
):
    args = make_args(tmp_path, mlx_deecho_reuse=reuse)
    runtime = FakeRuntime("mlx")
    output_dir = args.output_dir / "mlx" / "chain" / "run-1"
    separator = FakeSeparator(output_dir)

    with patch.object(benchmark_apple_backends, "build_separator", return_value=separator):
        report = benchmark_apple_backends.run_chain_once(
            args,
            runtime,
            output_dir,
            retain_outputs=True,
        )

    deecho_filename = benchmark_apple_backends.MODEL_SPECS["deecho"].filename
    assert separator.loaded_models.count(deecho_filename) == expected_deecho_loads
    assert [source for source, _ in separator.separation_calls] == [
        args.audio_file,
        (output_dir / "01_kim_kim_instrumental.wav").resolve(),
        (output_dir / "01_kim_kim_vocals.wav").resolve(),
        (output_dir / "03_karaoke_karaoke_backing.wav").resolve(),
        (output_dir / "03_karaoke_karaoke_lead.wav").resolve(),
    ]
    last_stage = report["stages"][-1]
    assert last_stage["load_model_called"] is load_model_called
    assert last_stage["model_instance_reused"] is reuse
    assert last_stage["reuse_strategy"] == reuse_strategy
    assert all(stage["validation"]["valid"] for stage in report["stages"])
    assert runtime.cleanup_count == 1


def test_mps_chain_reports_internal_deecho_cache_reuse(tmp_path):
    args = make_args(tmp_path, backend="mps")
    runtime = FakeRuntime("mps")
    output_dir = args.output_dir / "mps" / "chain" / "run-1"
    separator = FakeSeparator(output_dir)

    with patch.object(benchmark_apple_backends, "build_separator", return_value=separator):
        report = benchmark_apple_backends.run_chain_once(
            args,
            runtime,
            output_dir,
            retain_outputs=True,
        )

    deecho_filename = benchmark_apple_backends.MODEL_SPECS["deecho"].filename
    assert separator.loaded_models.count(deecho_filename) == 2
    last_stage = report["stages"][-1]
    assert last_stage["load_model_called"] is True
    assert last_stage["model_instance_reused"] is True
    assert last_stage["reuse_strategy"] == "separator_cache"


def test_effective_mlx_settings_reports_profile_overrides():
    runtime = FakeRuntime("mlx")
    separator = Mock(
        performance_params={
            "speed_mode": "latency_safe_v3",
            "cache_clear_policy": "deferred",
            "write_workers": 2,
        },
        arch_specific_params={
            "Demucs": {"batch_size": 8},
            "MDXC": {"batch_size": 1},
            "VR": {"batch_size": 1},
            "ignored": {"overlap": 8},
        },
    )

    assert benchmark_apple_backends.effective_backend_settings(runtime, separator) == {
        "speed_mode": "latency_safe_v3",
        "cache_clear_policy": "deferred",
        "write_workers": 2,
        "architecture_batch_sizes": {"Demucs": 8, "MDXC": 1, "VR": 1},
    }
    assert benchmark_apple_backends.effective_backend_settings(FakeRuntime("mps"), separator) == {}


def test_git_worktree_state_separates_tracked_and_untracked_changes(tmp_path):
    completed = subprocess.CompletedProcess(
        args=["git"],
        returncode=0,
        stdout=" M tools/benchmark.py\n?? scratch.txt\n?? output/result.json\n",
        stderr="",
    )

    with (
        patch.object(benchmark_apple_backends.subprocess, "run", return_value=completed),
        patch.object(benchmark_apple_backends, "git_revision", return_value="abc123"),
    ):
        state = benchmark_apple_backends.git_worktree_state(tmp_path)

    assert state == {
        "revision": "abc123",
        "tracked_dirty": True,
        "untracked_file_count": 2,
    }


def test_importing_benchmark_module_does_not_import_backend_frameworks():
    script = """
import json
import sys
import tools.benchmark_apple_backends
print(json.dumps({name: name in sys.modules for name in ('torch', 'mlx', 'mlx.core')}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=benchmark_apple_backends.REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {"torch": False, "mlx": False, "mlx.core": False}
    assert completed.stderr == ""


def test_model_progress_is_written_only_to_stderr(tmp_path, capsys):
    args = make_args(tmp_path, backend="mlx")
    args.mode = "models"
    runtime = FakeRuntime("mlx")
    separator = Mock()
    outputs = {
        "other": tmp_path / "other.wav",
        "vocals": tmp_path / "vocals.wav",
    }

    with (
        patch.object(benchmark_apple_backends, "build_separator", return_value=separator),
        patch.object(benchmark_apple_backends, "timed_load", return_value=0.5),
        patch.object(
            benchmark_apple_backends,
            "timed_separation",
            return_value=(1.25, outputs, {"inference_s": 1.0}),
        ),
        patch.object(benchmark_apple_backends, "validate_outputs", return_value={"valid": True}),
        patch.object(benchmark_apple_backends, "audio_duration", return_value=2.0),
        patch.object(benchmark_apple_backends, "process_peak_rss_bytes", return_value=123),
    ):
        result = benchmark_apple_backends.benchmark_model(args, runtime, "kim")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "model": "kim",
        "iteration": 1,
        "seconds": 1.25,
        "backend_memory": {"active_bytes": 0},
        "backend_perf_metrics": {"inference_s": 1.0},
    }
    assert result["median_seconds"] == 1.25


def test_main_writes_one_final_json_to_stdout_and_progress_to_stderr(tmp_path, capsys):
    args = make_args(tmp_path, backend="mlx")
    args.mode = "models"
    args.models = ["kim"]
    json_output = tmp_path / "result.json"
    args.json_output = json_output
    runtime = FakeRuntime("mlx")

    def fake_benchmark_model(_args, _runtime, model_key):
        print(json.dumps({"progress_model": model_key}), file=sys.stderr)
        return {"model": model_key, "median_seconds": 1.0}

    with (
        patch.object(benchmark_apple_backends, "parse_args", return_value=args),
        patch.object(benchmark_apple_backends, "build_runtime", return_value=runtime),
        patch.object(benchmark_apple_backends, "environment_report", return_value={"python": "test"}),
        patch.object(benchmark_apple_backends, "settings_report", return_value={"backend": "mlx"}),
        patch.object(benchmark_apple_backends, "audio_file_info", return_value={"duration_seconds": 1.0}),
        patch.object(benchmark_apple_backends, "process_peak_rss_bytes", return_value=123),
        patch.object(benchmark_apple_backends, "benchmark_model", side_effect=fake_benchmark_model),
    ):
        assert benchmark_apple_backends.main([]) == 0

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert captured.out.count("\n") == 1 + json.dumps(report, ensure_ascii=False, indent=2).count("\n")
    assert captured.err == '{"progress_model": "kim"}\n'
    assert report["models"] == [{"model": "kim", "median_seconds": 1.0}]
    assert json.loads(json_output.read_text(encoding="utf-8")) == report
