import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from tools import compare_stem_outputs


def write_audio(path: Path, audio: np.ndarray, sample_rate: int = 44_100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="FLOAT")


def test_extract_stem_name_uses_last_parenthesized_label():
    path = Path("song_(Live)_(Lead Vocals)_model.wav")

    assert compare_stem_outputs.extract_stem_name(path) == "Lead Vocals"
    assert compare_stem_outputs.extract_stem_name(Path("vocals.wav")) == "vocals"


def test_compare_collections_matches_stems_independent_of_order_and_case(tmp_path):
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    vocals = np.array([[0.25, -0.25], [0.5, -0.5], [0.0, 0.125]], dtype=np.float32)
    instrumental = np.array([[0.1, 0.2], [-0.1, -0.2], [0.3, -0.3]], dtype=np.float32)
    write_audio(reference_dir / "song_(Vocals)_torch.wav", vocals)
    write_audio(reference_dir / "song_(Instrumental)_torch.wav", instrumental)
    write_audio(candidate_dir / "song_(instrumental)_mlx.wav", instrumental)
    write_audio(candidate_dir / "song_(VOCALS)_mlx.wav", vocals)

    report = compare_stem_outputs.compare_collections([reference_dir], [candidate_dir])

    assert report["valid"] is True
    assert list(report["stems"]) == ["instrumental", "vocals"]
    assert report["summary"]["matched_stem_count"] == 2
    assert report["summary"]["comparable_stem_count"] == 2
    for stem in report["stems"].values():
        assert stem["checks"] == {
            "sample_rate_equal": True,
            "channels_equal": True,
            "shape_equal": True,
            "reference_finite": True,
            "candidate_finite": True,
        }
        assert stem["metrics"]["snr_db"] is None
        assert stem["metrics"]["snr_infinite"] is True
        assert stem["metrics"]["correlation"] == pytest.approx(1.0)
        assert stem["metrics"]["mae"] == 0.0
        assert stem["metrics"]["relative_l2"] == 0.0


def test_calculate_metrics_reports_expected_waveform_differences():
    reference = np.array([[1.0], [-1.0], [0.5], [-0.5]])
    candidate = np.array([[0.9], [-0.9], [0.4], [-0.4]])

    metrics = compare_stem_outputs.calculate_metrics(reference, candidate)

    assert metrics["snr_db"] == pytest.approx(17.9588001734)
    assert metrics["correlation"] == pytest.approx(0.9989685402)
    assert metrics["mae"] == pytest.approx(0.1)
    assert metrics["rmse"] == pytest.approx(0.1)
    assert metrics["max_abs"] == pytest.approx(0.1)
    assert metrics["relative_l2"] == pytest.approx(0.1264911064)
    assert metrics["reference_rms"] == pytest.approx(0.7905694150)
    assert metrics["candidate_rms"] == pytest.approx(0.6964194139)


def test_compare_collections_reports_shape_and_sample_rate_mismatch(tmp_path):
    reference = tmp_path / "reference_(Vocals).wav"
    candidate = tmp_path / "candidate_(Vocals).wav"
    write_audio(reference, np.zeros((8, 2), dtype=np.float32), sample_rate=44_100)
    write_audio(candidate, np.zeros((6, 1), dtype=np.float32), sample_rate=48_000)

    report = compare_stem_outputs.compare_collections([reference], [candidate])
    stem = report["stems"]["vocals"]

    assert report["valid"] is False
    assert stem["reference"]["sample_rate"] == 44_100
    assert stem["reference"]["channels"] == 2
    assert stem["reference"]["shape"] == [8, 2]
    assert stem["candidate"]["sample_rate"] == 48_000
    assert stem["candidate"]["channels"] == 1
    assert stem["candidate"]["shape"] == [6, 1]
    assert stem["checks"]["sample_rate_equal"] is False
    assert stem["checks"]["channels_equal"] is False
    assert stem["checks"]["shape_equal"] is False
    assert stem["metrics"] is None


def test_compare_collections_reports_unmatched_stems(tmp_path):
    reference = tmp_path / "reference_(Vocals).wav"
    candidate = tmp_path / "candidate_(Instrumental).wav"
    write_audio(reference, np.zeros((4, 1), dtype=np.float32))
    write_audio(candidate, np.zeros((4, 1), dtype=np.float32))

    report = compare_stem_outputs.compare_collections([reference], [candidate])

    assert report["valid"] is False
    assert report["reference_only_stems"] == ["vocals"]
    assert report["candidate_only_stems"] == ["instrumental"]
    assert report["stems"] == {}


def test_duplicate_stem_name_is_rejected(tmp_path):
    first = tmp_path / "one_(Vocals).wav"
    second = tmp_path / "two_(vocals).wav"
    write_audio(first, np.zeros((4, 1), dtype=np.float32))
    write_audio(second, np.zeros((4, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="stem名が重複"):
        compare_stem_outputs.map_files_by_stem([tmp_path], "reference")


def test_main_writes_only_final_json_to_stdout_and_progress_to_stderr(tmp_path, capsys):
    reference = tmp_path / "reference_(Vocals).wav"
    candidate = tmp_path / "candidate_(Vocals).wav"
    json_output = tmp_path / "result.json"
    audio = np.array([[0.25], [-0.25]], dtype=np.float32)
    write_audio(reference, audio)
    write_audio(candidate, audio)
    argv = [
        "compare_stem_outputs.py",
        "--reference",
        str(reference),
        "--candidate",
        str(candidate),
        "--json-output",
        str(json_output),
    ]

    with patch("sys.argv", argv):
        assert compare_stem_outputs.main() == 0

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["valid"] is True
    assert captured.err == "比較中: vocals\n"
    assert json.loads(json_output.read_text(encoding="utf-8")) == report
