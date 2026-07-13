#!/usr/bin/env python3
"""実モデルを使ってaudio-separatorの推論時間を計測する。"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import gc
import json
import logging
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audio_separator.separator import Separator


MPS_ENVIRONMENT_VARIABLES = (
    "PYTORCH_MPS_FAST_MATH",
    "PYTORCH_MPS_PREFER_METAL",
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="モデル読込とstem分離を分け、warm-up後の実行時間をJSONで記録します。",
    )
    parser.add_argument("audio_file", type=Path)
    parser.add_argument("--model", action="append", required=True, dest="models")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-output"))
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="mps")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--autocast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mdxc-segment-size", type=int, default=1101)
    parser.add_argument("--mdxc-overlap", type=int, default=8)
    parser.add_argument("--mdxc-batch-size", type=int, default=1)
    parser.add_argument("--vr-batch-size", type=int, default=1)
    parser.add_argument("--vr-window-size", type=int, default=320)
    parser.add_argument("--vr-aggression", type=int, default=50)
    parser.add_argument("--vr-tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--demucs-shifts", type=int, default=2)
    parser.add_argument("--demucs-overlap", type=float, default=0.25)
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmupは0以上にしてください。")
    if args.repeats < 1:
        parser.error("--repeatsは1以上にしてください。")
    if args.mdxc_batch_size < 1 or args.vr_batch_size < 1:
        parser.error("batch sizeは1以上にしてください。")
    if not args.audio_file.is_file():
        parser.error(f"音源が見つかりません: {args.audio_file}")
    if not args.model_dir.is_dir():
        parser.error(f"モデルディレクトリが見つかりません: {args.model_dir}")
    return args


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def mps_memory() -> dict[str, int] | None:
    if not torch.backends.mps.is_available():
        return None

    result = {
        "current_allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
    }
    recommended_max_memory = getattr(torch.mps, "recommended_max_memory", None)
    if recommended_max_memory is not None:
        result["recommended_max_bytes"] = recommended_max_memory()
    return result


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_separator(args: argparse.Namespace, model_output_dir: Path) -> Separator:
    separator = Separator(
        log_level=logging.WARNING,
        model_file_dir=str(args.model_dir),
        output_dir=str(model_output_dir),
        output_format="WAV",
        normalization_threshold=1.0,
        amplification_threshold=0.0,
        use_soundfile=True,
        use_autocast=args.autocast,
        use_torch_compile=args.torch_compile,
        mdxc_params={
            "segment_size": args.mdxc_segment_size,
            "override_model_segment_size": True,
            "batch_size": args.mdxc_batch_size,
            "overlap": args.mdxc_overlap,
            "pitch_shift": 0,
        },
        vr_params={
            "batch_size": args.vr_batch_size,
            "window_size": args.vr_window_size,
            "aggression": args.vr_aggression,
            "enable_tta": args.vr_tta,
            "enable_post_process": False,
            "post_process_threshold": 0.2,
            "high_end_process": False,
        },
        demucs_params={
            "segment_size": "Default",
            "shifts": args.demucs_shifts,
            "overlap": args.demucs_overlap,
            "segments_enabled": True,
        },
    )

    if args.device != "auto":
        requested_device = torch.device(args.device)
        if requested_device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPSは利用できません。")
        separator.torch_device = requested_device
        separator.torch_device_cpu = torch.device("cpu")
        separator.torch_device_mps = requested_device if requested_device.type == "mps" else None
        separator.onnx_execution_provider = ["CPUExecutionProvider"]
    return separator


def timed_separation(separator: Separator, audio_file: Path, seed: int) -> tuple[float, list[str]]:
    seed_everything(seed)
    synchronize(separator.torch_device)
    started_at = time.perf_counter()
    output_files = separator.separate(str(audio_file))
    synchronize(separator.torch_device)
    if not output_files:
        raise RuntimeError("分離結果が0件でした。直前のerror logを確認してください。")
    return time.perf_counter() - started_at, output_files


def benchmark_model(
    args: argparse.Namespace,
    model: str,
    on_device_resolved: Callable[[torch.device], None] | None = None,
) -> dict[str, Any]:
    model_output_dir = args.output_dir / Path(model).stem
    model_output_dir.mkdir(parents=True, exist_ok=True)
    separator = build_separator(args, model_output_dir)
    if on_device_resolved is not None:
        on_device_resolved(separator.torch_device)

    seed_everything(args.seed)
    synchronize(separator.torch_device)
    started_at = time.perf_counter()
    separator.load_model(model)
    synchronize(separator.torch_device)
    load_seconds = time.perf_counter() - started_at

    warmup_seconds = []
    for _ in range(args.warmup):
        elapsed, _ = timed_separation(separator, args.audio_file, args.seed)
        warmup_seconds.append(elapsed)

    runs = []
    output_files: list[str] = []
    for iteration in range(1, args.repeats + 1):
        elapsed, output_files = timed_separation(separator, args.audio_file, args.seed)
        run = {
            "iteration": iteration,
            "seconds": elapsed,
            "mps_memory": mps_memory(),
        }
        runs.append(run)
        print(json.dumps({"model": model, **run}, ensure_ascii=False), file=sys.stderr, flush=True)

    seconds = [run["seconds"] for run in runs]
    result = {
        "model": model,
        "device": str(separator.torch_device),
        "load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "runs": runs,
        "median_seconds": statistics.median(seconds),
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
        "output_files": output_files,
    }

    del separator
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return result


def warn_cpu_autocast() -> None:
    """CPU autocastがfp32基準にならないことをstderrへ警告する。"""
    print(
        "警告: CPU + autocastはbfloat16推論です。fp32基準との比較には--no-autocastを指定してください。",
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cpu_autocast_warned = False

    def handle_resolved_device(device: torch.device) -> None:
        nonlocal cpu_autocast_warned
        if args.autocast and device.type == "cpu" and not cpu_autocast_warned:
            warn_cpu_autocast()
            cpu_autocast_warned = True

    revision = git_revision()
    models = [benchmark_model(args, model, handle_resolved_device) for model in args.models]

    report = {
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "git_revision": revision,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "mps_environment": {name: os.environ[name] for name in MPS_ENVIRONMENT_VARIABLES if name in os.environ},
        },
        "settings": {
            "audio_file": str(args.audio_file.resolve()),
            "model_dir": str(args.model_dir.resolve()),
            "device": args.device,
            "autocast": args.autocast,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "torch_compile": args.torch_compile,
            "mdxc_segment_size": args.mdxc_segment_size,
            "mdxc_overlap": args.mdxc_overlap,
            "mdxc_batch_size": args.mdxc_batch_size,
            "vr_batch_size": args.vr_batch_size,
            "vr_window_size": args.vr_window_size,
            "vr_aggression": args.vr_aggression,
            "vr_tta": args.vr_tta,
            "demucs_shifts": args.demucs_shifts,
            "demucs_overlap": args.demucs_overlap,
        },
        "models": models,
    }

    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(f"{serialized}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
