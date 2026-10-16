"""Tests for selfcite_rl.trim."""

from selfcite_rl import trim


def test_no_trim_when_under_limit():
    kept, removed = trim.trim_sents_by_key_distance(["a", "b"], [1, 1], {0}, 5)
    assert kept == ["a", "b"]
    assert removed == set()


def test_removes_farthest_non_key_sentences_first():
    # Example from the original trim.py __main__ block.
    sents = [f"s{i}" for i in range(7)]
    lens = [100, 300, 100, 150, 200, 100, 250]
    kept, removed = trim.trim_sents_by_key_distance(sents, lens, {1, 4}, 800)
    # Distances: s6 -> 2, the rest -> 1; ties keep index order (0, 2, ...).
    assert removed == {6, 0, 2}
    assert kept == ["s1", "s3", "s4", "s5"]
    assert sum(lens[int(s[1:])] for s in kept) <= 800
