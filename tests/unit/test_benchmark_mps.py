import argparse
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from tools import benchmark_mps


def make_args(tmp_path: Path, *, device: str = "mps", autocast: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        audio_file=tmp_path / "input.wav",
        models=["model.ckpt"],
        model_dir=tmp_path,
        output_dir=tmp_path / "output",
        json_output=None,
        device=device,
        warmup=0,
        repeats=1,
        seed=0,
        autocast=autocast,
        torch_compile=False,
        mdxc_segment_size=1101,
        mdxc_overlap=8,
        mdxc_batch_size=1,
        vr_batch_size=1,
        vr_window_size=320,
        vr_aggression=50,
        vr_tta=True,
        demucs_shifts=2,
        demucs_overlap=0.25,
    )


@pytest.mark.parametrize("requested_device", ["cpu", "auto"])
def test_main_warns_when_autocast_resolves_to_cpu(tmp_path, capsys, requested_device):
    args = make_args(tmp_path, device=requested_device)
    args.models = ["first.ckpt", "second.ckpt"]
    events = []
    original_warn_cpu_autocast = benchmark_mps.warn_cpu_autocast

    def fake_benchmark_model(_args, model, on_device_resolved):
        on_device_resolved(torch.device("cpu"))
        events.append("benchmark")
        return {"model": model, "device": "cpu"}

    def fake_warn_cpu_autocast():
        events.append("warning")
        original_warn_cpu_autocast()

    def fake_git_revision():
        events.append("revision")
        return "revision"

    with (
        patch.object(benchmark_mps, "parse_args", return_value=args),
        patch.object(benchmark_mps, "benchmark_model", side_effect=fake_benchmark_model),
        patch.object(benchmark_mps, "warn_cpu_autocast", side_effect=fake_warn_cpu_autocast),
        patch.object(benchmark_mps, "git_revision", side_effect=fake_git_revision),
    ):
        assert benchmark_mps.main() == 0

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["models"] == [
        {"model": "first.ckpt", "device": "cpu"},
        {"model": "second.ckpt", "device": "cpu"},
    ]
    assert captured.err.count("CPU + autocastはbfloat16推論です") == 1
    assert events == ["revision", "warning", "benchmark", "benchmark"]


def test_benchmark_progress_is_written_only_to_stderr(tmp_path, capsys):
    args = make_args(tmp_path)
    separator = Mock()
    separator.torch_device = torch.device("cpu")
    events = []
    separator.load_model.side_effect = lambda _model: events.append("load")
    on_device_resolved = Mock(side_effect=lambda _device: events.append("device"))

    with (
        patch.object(benchmark_mps, "build_separator", return_value=separator),
        patch.object(benchmark_mps, "seed_everything"),
        patch.object(benchmark_mps, "synchronize"),
        patch.object(benchmark_mps, "timed_separation", return_value=(1.25, ["stem.wav"])),
        patch.object(benchmark_mps, "mps_memory", return_value=None),
        patch.object(torch.backends.mps, "is_available", return_value=False),
    ):
        result = benchmark_mps.benchmark_model(args, "model.ckpt", on_device_resolved)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "model": "model.ckpt",
        "iteration": 1,
        "seconds": 1.25,
        "mps_memory": None,
    }
    assert result["median_seconds"] == 1.25
    on_device_resolved.assert_called_once_with(torch.device("cpu"))
    assert events == ["device", "load"]
