"""Tests for selfcite_rl.prepare_data."""

from selfcite_rl import chunking
from selfcite_rl import prepare_data


def _longcite_prompt(chunks, query):
    context = "".join(f"<C{i}>{c}" for i, c in enumerate(chunks))
    return (
        "System text.\n\n[Document Start]\n"
        f"{context}\n[Document End]\n\n{query}"
    )


def _word_count(text):
    return len(text.split())


def test_parse_longcite_prompt():
    chunks, query = prepare_data.parse_longcite_prompt(
        _longcite_prompt(["a b. ", "c d."], "Why?")
    )
    assert chunks == ["a b. ", "c d."]
    assert query == "Why?"


def test_reference_chunk_ids():
    response = "<statement>X<cite>[1-2][5-5]</cite></statement>"
    assert prepare_data.reference_chunk_ids(response) == {1, 2, 5}


def test_build_example_keeps_everything_under_budget():
    row = {
        "prompt": _longcite_prompt(["a b. ", "c d."], "Why?"),
        "response": "<statement>X<cite>[0-0]</cite></statement>",
    }
    example = prepare_data.build_example(
        row, _word_count, _word_count, max_prompt_tokens=10_000
    )
    assert example["metadata"]["chunks"] == ["a b. ", "c d."]
    assert example["metadata"]["query"] == "Why?"
    assert example["prompt"] == chunking.build_prompt(
        ["a b. ", "c d."], "Why?"
    )


def test_build_example_trims_far_chunks_and_keeps_cited():
    chunks = ["x " * 50, "cited here. ", "y " * 50, "z " * 50]
    row = {
        "prompt": _longcite_prompt(chunks, "Why?"),
        "response": "<statement>X<cite>[1-1]</cite></statement>",
    }
    template_len = _word_count(chunking.build_prompt([], "Why?"))
    example = prepare_data.build_example(
        row, _word_count, _word_count, max_prompt_tokens=template_len + 110
    )
    kept = example["metadata"]["chunks"]
    assert "cited here. " in kept
    assert chunks[3] not in kept  # Farthest from the cited chunk.
    assert _word_count(example["prompt"]) <= template_len + 110


def test_build_example_returns_none_when_cited_text_alone_is_too_long():
    row = {
        "prompt": _longcite_prompt(["w " * 100], "Why?"),
        "response": "<statement>X<cite>[0-0]</cite></statement>",
    }
    assert (
        prepare_data.build_example(
            row, _word_count, _word_count, max_prompt_tokens=20
        )
        is None
    )
