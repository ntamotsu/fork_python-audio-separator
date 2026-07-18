#!/usr/bin/env python3
"""PyTorch/MPS版とMLX版を同一条件で比較するベンチマーク。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
from importlib import metadata
import json
import logging
import os
import platform
import random
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

import numpy as np
import soundfile as sf


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    output_names: dict[str, str]


MODEL_SPECS = {
    "kim": ModelSpec(
        filename="mel_band_roformer_kim_ft2_bleedless_unwa.ckpt",
        output_names={"other": "kim_instrumental", "vocals": "kim_vocals"},
    ),
    "demucs": ModelSpec(
        filename="htdemucs.yaml",
        output_names={
            "bass": "demucs_bass",
            "drums": "demucs_drums",
            "other": "demucs_other",
            "vocals": "demucs_vocals",
        },
    ),
    "karaoke": ModelSpec(
        filename="mel_band_roformer_karaoke_gabox_v2.ckpt",
        output_names={"instrumental": "karaoke_backing", "vocals": "karaoke_lead"},
    ),
    "deecho": ModelSpec(
        filename="UVR-DeEcho-DeReverb.pth",
        output_names={"no reverb": "deecho_dry", "reverb": "deecho_reverb"},
    ),
}


class BackendRuntime(Protocol):
    name: str

    def synchronize(self) -> None: ...

    def seed(self, seed: int) -> None: ...

    def memory(self) -> dict[str, int | None]: ...

    def reset_peak_memory(self) -> None: ...

    def cleanup(self) -> None: ...


class MPSRuntime:
    name = "mps"

    def __init__(self) -> None:
        import torch

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPSは利用できません。")
        self.torch = torch

    def synchronize(self) -> None:
        self.torch.mps.synchronize()

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.torch.manual_seed(seed)
        self.torch.mps.manual_seed(seed)

    def memory(self) -> dict[str, int | None]:
        peak_memory = getattr(self.torch.mps, "peak_memory_stats", None)
        peak_bytes = None
        if callable(peak_memory):
            stats = peak_memory()
            peak_bytes = stats.get("allocation", {}).get("peak")
        result = {
            "current_allocated_bytes": self.torch.mps.current_allocated_memory(),
            "driver_allocated_bytes": self.torch.mps.driver_allocated_memory(),
            "peak_allocated_bytes": peak_bytes,
        }
        recommended = getattr(self.torch.mps, "recommended_max_memory", None)
        if callable(recommended):
            result["recommended_max_bytes"] = recommended()
        return result

    def reset_peak_memory(self) -> None:
        reset = getattr(self.torch.mps, "reset_peak_memory_stats", None)
        if callable(reset):
            reset()

    def cleanup(self) -> None:
        gc.collect()
        self.torch.mps.empty_cache()


class MLXRuntime:
    name = "mlx"

    def __init__(self) -> None:
        import mlx.core as mx

        self.mx = mx

    def synchronize(self) -> None:
        self.mx.synchronize()

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self.mx.random.seed(seed)

    def memory(self) -> dict[str, int | None]:
        return {
            "active_bytes": self.mx.get_active_memory(),
            "cache_bytes": self.mx.get_cache_memory(),
            "peak_bytes": self.mx.get_peak_memory(),
        }

    def reset_peak_memory(self) -> None:
        self.mx.reset_peak_memory()

    def cleanup(self) -> None:
        gc.collect()
        self.mx.clear_cache()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MPS版とMLX版のpublic separate()を、製品相当設定で計測します。",
    )
    parser.add_argument("audio_file", type=Path)
    parser.add_argument("--backend", choices=("mps", "mlx"), required=True)
    parser.add_argument("--mode", choices=("models", "chain"), default="models")
    parser.add_argument(
        "--model",
        action="append",
        choices=tuple(MODEL_SPECS),
        dest="models",
        help="modelsモードで計測するモデル。省略時は4モデル全て。",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-output"))
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--mlx-speed-mode",
        choices=("default", "latency_safe", "latency_safe_v2", "latency_safe_v3"),
        default="default",
    )
    parser.add_argument("--mlx-cache-clear-policy", choices=("aggressive", "deferred"), default="aggressive")
    parser.add_argument("--mlx-write-workers", type=int, default=1)
    parser.add_argument("--mlx-demucs-batch-size", type=int, default=1)
    parser.add_argument(
        "--mlx-deecho-reuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="chainの2回目のDeEchoで既存instanceを再利用します。",
    )
    parser.add_argument("--mdxc-segment-size", type=int, default=1101)
    parser.add_argument("--mdxc-overlap", type=int, default=8)
    parser.add_argument("--vr-window-size", type=int, default=320)
    parser.add_argument("--vr-aggression", type=int, default=50)
    parser.add_argument("--vr-tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--demucs-shifts", type=int, default=2)
    parser.add_argument("--demucs-overlap", type=float, default=0.25)
    args = parser.parse_args(argv)

    if args.warmup < 0:
        parser.error("--warmupは0以上にしてください。")
    if args.repeats < 1:
        parser.error("--repeatsは1以上にしてください。")
    if args.mlx_write_workers < 1 or args.mlx_demucs_batch_size < 1:
        parser.error("MLXのbatch sizeとwrite workersは1以上にしてください。")
    if not args.audio_file.is_file():
        parser.error(f"音源が見つかりません: {args.audio_file}")
    if not args.model_dir.is_dir():
        parser.error(f"モデルディレクトリが見つかりません: {args.model_dir}")
    args.models = args.models or list(MODEL_SPECS)
    return args


def build_runtime(backend: str) -> BackendRuntime:
    if backend == "mps":
        return MPSRuntime()
    return MLXRuntime()


def build_separator(args: argparse.Namespace, runtime: BackendRuntime, output_dir: Path) -> Any:
    common_kwargs = {
        "log_level": logging.WARNING,
        "model_file_dir": str(args.model_dir),
        "output_dir": str(output_dir),
        "output_format": "WAV",
        "normalization_threshold": 1.0,
        "amplification_threshold": 0.0,
        "mdxc_params": {
            "segment_size": args.mdxc_segment_size,
            "override_model_segment_size": True,
            "batch_size": 1,
            "overlap": args.mdxc_overlap,
            "pitch_shift": 0,
        },
        "vr_params": {
            "batch_size": 1,
            "window_size": args.vr_window_size,
            "aggression": args.vr_aggression,
            "enable_tta": args.vr_tta,
            "enable_post_process": False,
            "post_process_threshold": 0.2,
            "high_end_process": False,
        },
        "demucs_params": {
            "segment_size": "Default",
            "shifts": args.demucs_shifts,
            "overlap": args.demucs_overlap,
            "segments_enabled": True,
        },
    }

    if runtime.name == "mps":
        from audio_separator.separator import Separator

        separator = Separator(
            **common_kwargs,
            use_soundfile=True,
            use_autocast=True,
            use_torch_compile=args.torch_compile,
        )
        torch = runtime.torch  # type: ignore[attr-defined]
        separator.torch_device = torch.device("mps")
        separator.torch_device_cpu = torch.device("cpu")
        separator.torch_device_mps = separator.torch_device
        separator.onnx_execution_provider = ["CPUExecutionProvider"]
        return separator

    from mlx_audio_separator import Separator

    common_kwargs["demucs_params"]["batch_size"] = args.mlx_demucs_batch_size
    separator = Separator(
        **common_kwargs,
        performance_params={
            "speed_mode": args.mlx_speed_mode,
            "cache_clear_policy": args.mlx_cache_clear_policy,
            "write_workers": args.mlx_write_workers,
        },
        save_converted_safetensors=False,
    )
    strict_errors = getattr(separator, "_set_strict_separation_errors", None)
    if callable(strict_errors):
        strict_errors(True)
    return separator


def normalize_returned_outputs(output_dir: Path, output_files: list[str]) -> list[Path]:
    normalized = []
    for output_file in output_files:
        path = Path(output_file)
        normalized.append((path if path.is_absolute() else output_dir / path).resolve())
    return normalized


def map_named_outputs(
    output_dir: Path,
    custom_output_names: dict[str, str],
    output_files: list[str],
) -> dict[str, Path]:
    """返却順や相対/絶対pathに依存せず、指定したstemへ出力を対応付ける。"""
    expected_by_name = {f"{name}.wav".casefold(): stem for stem, name in custom_output_names.items()}
    result: dict[str, Path] = {}
    for path in normalize_returned_outputs(output_dir, output_files):
        stem = expected_by_name.get(path.name.casefold())
        if stem is None:
            raise RuntimeError(f"想定外の出力です: {path.name}")
        if stem in result:
            raise RuntimeError(f"出力stemが重複しています: {stem}")
        result[stem] = path

    missing = sorted(set(custom_output_names) - set(result))
    if missing:
        raise RuntimeError(f"出力stemが不足しています: {', '.join(missing)}")
    return result


def validate_outputs(outputs: dict[str, Path]) -> dict[str, Any]:
    details = {}
    valid = True
    for stem, path in outputs.items():
        detail: dict[str, Any] = {
            "path": str(path),
            "exists": path.is_file(),
            "nonempty": path.is_file() and path.stat().st_size > 0,
        }
        if detail["nonempty"]:
            try:
                info = sf.info(path)
                detail.update(
                    {
                        "decodable": True,
                        "sample_rate": info.samplerate,
                        "channels": info.channels,
                        "frames": info.frames,
                        "duration_seconds": info.duration,
                    }
                )
            except sf.SoundFileError as error:
                detail.update({"decodable": False, "decode_error": str(error)})
        else:
            detail["decodable"] = False
        valid = valid and detail["exists"] and detail["nonempty"] and detail["decodable"]
        details[stem] = detail
    return {"valid": valid, "outputs": details}


def timed_load(runtime: BackendRuntime, separator: Any, model_filename: str) -> float:
    runtime.synchronize()
    started_at = time.perf_counter()
    separator.load_model(model_filename)
    runtime.synchronize()
    return time.perf_counter() - started_at


def timed_separation(
    runtime: BackendRuntime,
    separator: Any,
    audio_file: Path,
    output_dir: Path,
    output_names: dict[str, str],
    seed: int,
) -> tuple[float, dict[str, Path], dict[str, Any] | None]:
    runtime.seed(seed)
    runtime.synchronize()
    started_at = time.perf_counter()
    output_files = separator.separate(str(audio_file), custom_output_names=output_names)
    runtime.synchronize()
    elapsed = time.perf_counter() - started_at
    if not output_files:
        raise RuntimeError("分離結果が0件でした。")
    outputs = map_named_outputs(output_dir, output_names, output_files)
    validation = validate_outputs(outputs)
    if not validation["valid"]:
        raise RuntimeError(f"出力検証に失敗しました: {validation}")
    perf_metrics = getattr(separator, "last_perf_metrics", None)
    return elapsed, outputs, json_safe(perf_metrics) if perf_metrics is not None else None


def benchmark_model(
    args: argparse.Namespace,
    runtime: BackendRuntime,
    model_key: str,
) -> dict[str, Any]:
    spec = MODEL_SPECS[model_key]
    output_dir = args.output_dir / args.backend / "models" / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    separator = build_separator(args, runtime, output_dir)
    runtime.reset_peak_memory()

    load_seconds = timed_load(runtime, separator, spec.filename)
    memory_after_load = runtime.memory()
    warmup_seconds = []
    for _ in range(args.warmup):
        elapsed, _, _ = timed_separation(runtime, separator, args.audio_file, output_dir, spec.output_names, args.seed)
        warmup_seconds.append(elapsed)

    runs = []
    final_outputs: dict[str, Path] = {}
    for iteration in range(1, args.repeats + 1):
        elapsed, final_outputs, perf_metrics = timed_separation(
            runtime, separator, args.audio_file, output_dir, spec.output_names, args.seed
        )
        run = {
            "iteration": iteration,
            "seconds": elapsed,
            "backend_memory": runtime.memory(),
            "backend_perf_metrics": perf_metrics,
        }
        runs.append(run)
        print(json.dumps({"model": model_key, **run}, ensure_ascii=False), file=sys.stderr, flush=True)

    seconds = [run["seconds"] for run in runs]
    result = {
        "model": model_key,
        "model_filename": spec.filename,
        "load_seconds": load_seconds,
        "memory_after_load": memory_after_load,
        "warmup_seconds": warmup_seconds,
        "runs": runs,
        "median_seconds": statistics.median(seconds),
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
        "realtime_factor": statistics.median(seconds) / audio_duration(args.audio_file),
        "outputs": {stem: str(path) for stem, path in final_outputs.items()},
        "validation": validate_outputs(final_outputs),
        "process_peak_rss_bytes": process_peak_rss_bytes(),
    }

    del separator
    runtime.cleanup()
    return result


def chain_output_names(prefix: str, spec: ModelSpec) -> dict[str, str]:
    return {stem: f"{prefix}_{name}" for stem, name in spec.output_names.items()}


def run_chain_stage(
    *,
    args: argparse.Namespace,
    runtime: BackendRuntime,
    separator: Any,
    stage_name: str,
    model_key: str,
    source: Path,
    output_dir: Path,
    load_model: bool = True,
) -> tuple[dict[str, Any], dict[str, Path]]:
    spec = MODEL_SPECS[model_key]
    output_names = chain_output_names(stage_name, spec)
    load_seconds = timed_load(runtime, separator, spec.filename) if load_model else 0.0
    separate_seconds, outputs, perf_metrics = timed_separation(runtime, separator, source, output_dir, output_names, args.seed)
    result = {
        "stage": stage_name,
        "model": model_key,
        "model_filename": spec.filename,
        "source": str(source),
        "model_loaded": load_model,
        "load_seconds": load_seconds,
        "separate_seconds": separate_seconds,
        "backend_memory": runtime.memory(),
        "backend_perf_metrics": perf_metrics,
        "outputs": {stem: str(path) for stem, path in outputs.items()},
        "validation": validate_outputs(outputs),
    }
    print(
        json.dumps(
            {
                "stage": stage_name,
                "load_seconds": load_seconds,
                "separate_seconds": separate_seconds,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )
    return result, outputs


def run_chain_once(
    args: argparse.Namespace,
    runtime: BackendRuntime,
    output_dir: Path,
    *,
    retain_outputs: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    separator = build_separator(args, runtime, output_dir)
    runtime.reset_peak_memory()
    runtime.synchronize()
    started_at = time.perf_counter()
    stages = []

    kim, kim_outputs = run_chain_stage(
        args=args,
        runtime=runtime,
        separator=separator,
        stage_name="01_kim",
        model_key="kim",
        source=args.audio_file,
        output_dir=output_dir,
    )
    stages.append(kim)
    demucs, _ = run_chain_stage(
        args=args,
        runtime=runtime,
        separator=separator,
        stage_name="02_demucs",
        model_key="demucs",
        source=kim_outputs["other"],
        output_dir=output_dir,
    )
    stages.append(demucs)
    karaoke, karaoke_outputs = run_chain_stage(
        args=args,
        runtime=runtime,
        separator=separator,
        stage_name="03_karaoke",
        model_key="karaoke",
        source=kim_outputs["vocals"],
        output_dir=output_dir,
    )
    stages.append(karaoke)
    backing, _ = run_chain_stage(
        args=args,
        runtime=runtime,
        separator=separator,
        stage_name="04_deecho_backing",
        model_key="deecho",
        source=karaoke_outputs["instrumental"],
        output_dir=output_dir,
    )
    stages.append(backing)
    reuse_mlx_deecho = runtime.name == "mlx" and args.mlx_deecho_reuse
    lead, _ = run_chain_stage(
        args=args,
        runtime=runtime,
        separator=separator,
        stage_name="05_deecho_lead",
        model_key="deecho",
        source=karaoke_outputs["vocals"],
        output_dir=output_dir,
        load_model=not reuse_mlx_deecho,
    )
    lead["reused_previous_model"] = reuse_mlx_deecho
    stages.append(lead)

    runtime.synchronize()
    total_seconds = time.perf_counter() - started_at
    timed_subtotal = sum(stage["load_seconds"] + stage["separate_seconds"] for stage in stages)
    result = {
        "total_seconds": total_seconds,
        "timed_stage_subtotal_seconds": timed_subtotal,
        "runner_overhead_seconds": total_seconds - timed_subtotal,
        "stages": stages,
        "backend_memory": runtime.memory(),
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "output_dir": str(output_dir) if retain_outputs else None,
        "outputs_retained": retain_outputs,
    }

    del separator
    runtime.cleanup()
    if not retain_outputs:
        shutil.rmtree(output_dir)
        for stage in result["stages"]:
            stage["outputs"] = {stem: Path(path).name for stem, path in stage["outputs"].items()}
            for detail in stage["validation"]["outputs"].values():
                detail["path"] = Path(detail["path"]).name
    return result


def benchmark_chain(args: argparse.Namespace, runtime: BackendRuntime) -> dict[str, Any]:
    chain_root = args.output_dir / args.backend / "chain"
    warmups = []
    for iteration in range(1, args.warmup + 1):
        run = run_chain_once(
            args,
            runtime,
            chain_root / f"warmup-{iteration}",
            retain_outputs=False,
        )
        warmups.append(run["total_seconds"])

    runs = []
    for iteration in range(1, args.repeats + 1):
        run = run_chain_once(
            args,
            runtime,
            chain_root / f"run-{iteration}",
            retain_outputs=iteration == args.repeats,
        )
        run["iteration"] = iteration
        runs.append(run)
        print(
            json.dumps(
                {"chain_iteration": iteration, "total_seconds": run["total_seconds"]},
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )

    seconds = [run["total_seconds"] for run in runs]
    return {
        "warmup_seconds": warmups,
        "runs": runs,
        "median_seconds": statistics.median(seconds),
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
        "realtime_factor": statistics.median(seconds) / audio_duration(args.audio_file),
    }


def audio_duration(path: Path) -> float:
    return sf.info(path).duration


def audio_file_info(path: Path) -> dict[str, Any]:
    info = sf.info(path)
    return {
        "path": str(path.resolve()),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration_seconds": info.duration,
        "format": info.format,
        "subtype": info.subtype,
        "size_bytes": path.stat().st_size,
    }


def process_peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def package_source(name: str) -> dict[str, str | None] | None:
    """editable installの元repositoryとrevisionを可能なら記録する。"""
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None
    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        return None
    try:
        direct_url = json.loads(direct_url_text)
        parsed = urlparse(direct_url["url"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None
    if parsed.scheme != "file":
        return {"url": direct_url.get("url"), "path": None, "git_revision": None}
    source_path = Path(unquote(parsed.path)).resolve()
    return {
        "url": direct_url.get("url"),
        "path": str(source_path),
        "git_revision": git_revision(source_path),
    }


def model_inventory(model_dir: Path) -> dict[str, dict[str, Any]]:
    inventory = {}
    for key, spec in MODEL_SPECS.items():
        path = model_dir / spec.filename
        inventory[key] = {
            "filename": spec.filename,
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
    return inventory


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def settings_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backend": args.backend,
        "mode": args.mode,
        "models": args.models,
        "model_dir": str(args.model_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "torch_compile": args.torch_compile,
        "mlx_speed_mode": args.mlx_speed_mode,
        "mlx_cache_clear_policy": args.mlx_cache_clear_policy,
        "mlx_write_workers": args.mlx_write_workers,
        "mlx_demucs_batch_size": args.mlx_demucs_batch_size,
        "mlx_deecho_reuse": args.mlx_deecho_reuse,
        "mdxc_segment_size": args.mdxc_segment_size,
        "mdxc_overlap": args.mdxc_overlap,
        "mdxc_batch_size": 1,
        "vr_batch_size": 1,
        "vr_window_size": args.vr_window_size,
        "vr_aggression": args.vr_aggression,
        "vr_tta": args.vr_tta,
        "vr_post_process": False,
        "vr_high_end_process": False,
        "demucs_segment_size": "Default",
        "demucs_shifts": args.demucs_shifts,
        "demucs_overlap": args.demucs_overlap,
        "output_format": "WAV",
        "normalization_threshold": 1.0,
        "amplification_threshold": 0.0,
    }


def environment_report() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_revision": git_revision(REPOSITORY_ROOT),
        "packages": {
            name: package_version(name)
            for name in (
                "audio-separator",
                "mlx-audio-separator",
                "torch",
                "torchvision",
                "mlx",
                "mlx-audio-io",
                "mlx-spectro",
                "numpy",
                "soundfile",
            )
        },
        "package_sources": {name: package_source(name) for name in ("audio-separator", "mlx-audio-separator")},
        "environment": {
            name: os.environ[name]
            for name in (
                "HOME",
                "PYTORCH_MPS_FAST_MATH",
                "PYTORCH_MPS_PREFER_METAL",
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                "PYTORCH_MPS_LOW_WATERMARK_RATIO",
            )
            if name in os.environ
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = build_runtime(args.backend)
    report: dict[str, Any] = {
        "environment": environment_report(),
        "settings": settings_report(args),
        "input": audio_file_info(args.audio_file),
        "model_inventory": model_inventory(args.model_dir),
        "process_rss_after_import_bytes": process_peak_rss_bytes(),
    }
    if args.mode == "models":
        report["models"] = [benchmark_model(args, runtime, model) for model in args.models]
    else:
        report["chain"] = benchmark_chain(args, runtime)

    serialized = json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False)
    print(serialized)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(f"{serialized}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
