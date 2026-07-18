#!/usr/bin/env python3
"""2組のstem出力を名前で対応付け、音声品質の差をJSONで報告する。"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


SUPPORTED_AUDIO_EXTENSIONS = frozenset({".aif", ".aiff", ".flac", ".ogg", ".wav"})
STEM_NAME_PATTERN = re.compile(r"\(([^()]+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基準側と比較側の音声をstem名で照合し、波形差をJSONで出力します。",
    )
    parser.add_argument(
        "--reference",
        nargs="+",
        required=True,
        type=Path,
        help="基準にする音声ファイルまたはディレクトリ。ディレクトリ内は再帰的に探索します。",
    )
    parser.add_argument(
        "--candidate",
        nargs="+",
        required=True,
        type=Path,
        help="比較する音声ファイルまたはディレクトリ。ディレクトリ内は再帰的に探索します。",
    )
    parser.add_argument("--json-output", type=Path, help="stdoutと同じJSONを保存する任意のパス。")
    return parser.parse_args()


def extract_stem_name(path: Path) -> str:
    """audio-separator形式の末尾の括弧をstem名として取り出す。"""
    matches = STEM_NAME_PATTERN.findall(path.stem)
    if matches:
        return matches[-1].strip()
    return path.stem.strip()


def normalize_stem_name(name: str) -> str:
    """大文字小文字や空白表現だけが違うstem名を同じキーにする。"""
    normalized = unicodedata.normalize("NFKC", name).casefold()
    return " ".join(normalized.split())


def discover_audio_files(paths: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_file():
            if path.suffix.casefold() not in SUPPORTED_AUDIO_EXTENSIONS:
                raise ValueError(f"対応していない音声形式です: {path}")
            files.add(path.resolve())
            continue
        if path.is_dir():
            files.update(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.casefold() in SUPPORTED_AUDIO_EXTENSIONS
            )
            continue
        raise ValueError(f"ファイルまたはディレクトリが見つかりません: {path}")

    if not files:
        raise ValueError("比較対象の音声ファイルが見つかりませんでした。")
    return sorted(files)


def map_files_by_stem(paths: list[Path], side: str) -> dict[str, tuple[str, Path]]:
    result: dict[str, tuple[str, Path]] = {}
    for path in discover_audio_files(paths):
        stem_name = extract_stem_name(path)
        stem_key = normalize_stem_name(stem_name)
        if not stem_key:
            raise ValueError(f"stem名を抽出できませんでした: {path}")
        if stem_key in result:
            previous = result[stem_key][1]
            raise ValueError(f"{side}側でstem名が重複しています ({stem_name}): {previous}, {path}")
        result[stem_key] = (stem_name, path)
    return result


def _safe_rms(audio: np.ndarray, finite: bool) -> float | None:
    if not finite or audio.size == 0:
        return None
    flat = audio.reshape(-1)
    return math.sqrt(float(np.dot(flat, flat)) / flat.size)


def audio_metadata(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    finite = bool(np.isfinite(audio).all())
    return {
        "sample_rate": sample_rate,
        "channels": int(audio.shape[1]),
        "frames": int(audio.shape[0]),
        "shape": list(audio.shape),
        "finite": finite,
        "rms": _safe_rms(audio, finite),
    }


def _ratio_db(numerator: float, denominator: float) -> float | None:
    """有限なdB値だけを返し、無限大はJSONのnullで表現する。"""
    if numerator <= 0.0 or denominator <= 0.0:
        return None
    return 10.0 * math.log10(numerator / denominator)


def calculate_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | bool | None]:
    if reference.shape != candidate.shape:
        raise ValueError("指標計算には同じshapeの波形が必要です。")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("指標計算にはfiniteな波形が必要です。")

    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    difference = candidate_flat - reference_flat
    sample_count = reference_flat.size
    if sample_count == 0:
        raise ValueError("空の波形は比較できません。")

    reference_energy = float(np.dot(reference_flat, reference_flat))
    candidate_energy = float(np.dot(candidate_flat, candidate_flat))
    error_energy = float(np.dot(difference, difference))
    absolute_difference = np.abs(difference)

    reference_sum = float(np.sum(reference_flat))
    candidate_sum = float(np.sum(candidate_flat))
    covariance = float(np.dot(reference_flat, candidate_flat)) - (reference_sum * candidate_sum / sample_count)
    reference_variance = max(reference_energy - (reference_sum * reference_sum / sample_count), 0.0)
    candidate_variance = max(candidate_energy - (candidate_sum * candidate_sum / sample_count), 0.0)
    correlation_denominator = math.sqrt(reference_variance * candidate_variance)
    correlation = covariance / correlation_denominator if correlation_denominator > 0.0 else None
    if correlation is not None:
        correlation = min(max(correlation, -1.0), 1.0)

    exact_match = error_energy == 0.0
    snr_db = _ratio_db(reference_energy, error_energy)
    relative_l2 = math.sqrt(error_energy / reference_energy) if reference_energy > 0.0 else None
    return {
        "snr_db": snr_db,
        "snr_infinite": exact_match and reference_energy > 0.0,
        "correlation": correlation,
        "mae": float(np.mean(absolute_difference)),
        "rmse": math.sqrt(error_energy / sample_count),
        "max_abs": float(np.max(absolute_difference)),
        "relative_l2": relative_l2,
        "reference_rms": math.sqrt(reference_energy / sample_count),
        "candidate_rms": math.sqrt(candidate_energy / sample_count),
    }


def compare_audio_files(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference, reference_rate = sf.read(reference_path, dtype="float64", always_2d=True)
    candidate, candidate_rate = sf.read(candidate_path, dtype="float64", always_2d=True)
    reference_info = audio_metadata(reference, reference_rate)
    candidate_info = audio_metadata(candidate, candidate_rate)
    checks = {
        "sample_rate_equal": reference_rate == candidate_rate,
        "channels_equal": reference.shape[1] == candidate.shape[1],
        "shape_equal": reference.shape == candidate.shape,
        "reference_finite": reference_info["finite"],
        "candidate_finite": candidate_info["finite"],
    }
    comparable = all(checks.values()) and reference.size > 0
    return {
        "reference": reference_info,
        "candidate": candidate_info,
        "checks": checks,
        "comparable": comparable,
        "metrics": calculate_metrics(reference, candidate) if comparable else None,
    }


def _optional_extreme(values: list[float | None], extreme: Any) -> float | None:
    finite_values = [value for value in values if value is not None]
    return extreme(finite_values) if finite_values else None


def summarize(stems: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = [stem["metrics"] for stem in stems.values() if stem["metrics"] is not None]
    return {
        "matched_stem_count": len(stems),
        "comparable_stem_count": len(metrics),
        "min_snr_db": _optional_extreme([metric["snr_db"] for metric in metrics], min),
        "min_correlation": _optional_extreme([metric["correlation"] for metric in metrics], min),
        "max_mae": _optional_extreme([metric["mae"] for metric in metrics], max),
        "max_rmse": _optional_extreme([metric["rmse"] for metric in metrics], max),
        "max_abs": _optional_extreme([metric["max_abs"] for metric in metrics], max),
        "max_relative_l2": _optional_extreme([metric["relative_l2"] for metric in metrics], max),
    }


def compare_collections(reference_paths: list[Path], candidate_paths: list[Path]) -> dict[str, Any]:
    reference_files = map_files_by_stem(reference_paths, "reference")
    candidate_files = map_files_by_stem(candidate_paths, "candidate")
    reference_keys = set(reference_files)
    candidate_keys = set(candidate_files)
    matched_keys = sorted(reference_keys & candidate_keys)

    stems: dict[str, dict[str, Any]] = {}
    for stem_key in matched_keys:
        reference_name, reference_path = reference_files[stem_key]
        candidate_name, candidate_path = candidate_files[stem_key]
        print(f"比較中: {stem_key}", file=sys.stderr, flush=True)
        stems[stem_key] = {
            "reference_stem_name": reference_name,
            "candidate_stem_name": candidate_name,
            "reference_file": str(reference_path),
            "candidate_file": str(candidate_path),
            **compare_audio_files(reference_path, candidate_path),
        }

    reference_only = sorted(reference_keys - candidate_keys)
    candidate_only = sorted(candidate_keys - reference_keys)
    all_comparable = all(stem["comparable"] for stem in stems.values())
    report = {
        "valid": bool(stems) and not reference_only and not candidate_only and all_comparable,
        "reference_only_stems": reference_only,
        "candidate_only_stems": candidate_only,
        "stems": stems,
        "summary": summarize(stems),
    }
    return report


def main() -> int:
    args = parse_args()
    try:
        report = compare_collections(args.reference, args.candidate)
    except (OSError, ValueError, sf.SoundFileError) as error:
        print(f"比較に失敗しました: {error}", file=sys.stderr, flush=True)
        report = {"valid": False, "error": str(error)}

    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    print(serialized)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(f"{serialized}\n", encoding="utf-8")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
