"""Tests for selfcite_rl.citations."""

from selfcite_rl import citations


def test_strip_think_returns_text_after_think_block():
    assert citations.strip_think("<think>hmm</think>\n\nAnswer.") == "Answer."
    assert citations.strip_think("No thinking here.") == "No thinking here."


def test_strip_think_returns_none_when_think_is_unterminated():
    assert citations.strip_think("<think>still thinking...") is None


def test_parse_cited_statement():
    answer = "<statement>Paris is big.<cite>[0-1]</cite></statement>"
    (stmt,) = citations.parse_statements(answer, num_chunks=5)
    assert stmt.text == "Paris is big."
    assert stmt.chunk_ids == frozenset({0, 1})
    assert stmt.has_cite_tag
    assert answer[stmt.start:stmt.end] == "Paris is big."


def test_parse_keeps_uncited_text_longer_than_five_chars():
    answer = "Intro text here. <statement>A.<cite></cite></statement> ok"
    stmts = citations.parse_statements(answer, num_chunks=5)
    assert [s.text for s in stmts] == ["Intro text here. ", "A."]
    assert not stmts[0].has_cite_tag
    assert stmts[1].chunk_ids == frozenset()


def test_parse_drops_out_of_range_and_clamps_end():
    answer = "<statement>X.<cite>[9-9][3-99][4-2]</cite></statement>"
    (stmt,) = citations.parse_statements(answer, num_chunks=5)
    # [9-9] starts past the end, [4-2] is reversed, [3-99] clamps to [3-4].
    assert stmt.chunk_ids == frozenset({3, 4})


def test_parse_merges_adjacent_spans_and_keeps_first_three():
    answer = (
        "<statement>X.<cite>[0-0][1-1][3-3][5-5][7-7]</cite></statement>"
    )
    (stmt,) = citations.parse_statements(answer, num_chunks=10)
    # [0-0][1-1] merge into [0-1]; then [3-3], [5-5] fill the cap of 3.
    assert stmt.chunk_ids == frozenset({0, 1, 3, 5})


def test_parse_truncates_long_spans():
    answer = "<statement>X.<cite>[0-50]</cite></statement>"
    (stmt,) = citations.parse_statements(
        answer, num_chunks=100, max_span_chunks=10
    )
    assert stmt.chunk_ids == frozenset(range(10))


def test_parse_accepts_chunk_tag_style_ids():
    # The prompt tags chunks as <C3>, so models often cite [C3] or [C3-C5].
    answer = "<statement>A.<cite>[C1][3][C6-C7]</cite></statement>"
    (stmt,) = citations.parse_statements(answer, num_chunks=10)
    assert stmt.chunk_ids == frozenset({1, 3, 6, 7})


def test_parse_ignores_contentless_statements():
    # Copying the prompt's "<statement>{S}<cite>...</cite></statement>"
    # template must not count as a cited statement.
    answer = "<statement>{S}<cite>[C1]</cite></statement>"
    assert citations.parse_statements(answer, num_chunks=5) == []


def test_strip_think_drops_end_of_sequence_text():
    # miles' response text ends with the literal EOS token; it must not be
    # parsed as an uncited statement.
    response = "<think>hmm</think>\n\n<statement>A.<cite>[0-0]</cite></statement><|im_end|>"
    answer = citations.strip_think(response)
    assert answer == "<statement>A.<cite>[0-0]</cite></statement>"
    assert len(citations.parse_statements(answer, num_chunks=1)) == 1
