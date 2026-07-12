import pytest

from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator


@pytest.mark.parametrize(
    ("audio_length", "chunk_size", "step", "expected"),
    [
        (3, 4, 2, [0]),
        (10, 4, 3, [0, 3, 6]),
        (11, 4, 3, [0, 3, 6, 7]),
        (20, 8, 5, [0, 5, 10, 12]),
    ],
)
def test_roformer_chunk_starts_cover_tail_once(audio_length, chunk_size, step, expected):
    assert MDXCSeparator._roformer_chunk_starts(audio_length, chunk_size, step) == expected
