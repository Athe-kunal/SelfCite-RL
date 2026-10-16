"""Tests for selfcite_rl.reward."""

import dataclasses

from selfcite_rl import citations
from selfcite_rl import reward


def _stmt(chunk_ids=(), has_cite_tag=True, text="text"):
    return citations.Statement(
        text, frozenset(chunk_ids), has_cite_tag, 0, len(text)
    )


def test_unfinished_thinking_gets_format_penalty():
    result = reward.combine(answer=None, statements=[], scores=[])
    assert result.value == reward.FORMAT_PENALTY
    assert not result.format_ok


def test_answer_without_statement_tags_gets_format_penalty():
    stmts = [_stmt(has_cite_tag=False)]
    result = reward.combine(answer="plain text", statements=stmts, scores=[None])
    assert result.value == reward.FORMAT_PENALTY


def test_mean_over_cited_and_penalized_uncited_statements():
    stmts = [_stmt({0}), _stmt({1}), _stmt()]
    result = reward.combine(
        answer="...",
        statements=stmts,
        scores=[2.0, 1.0, None],
        uncited_penalty=-0.5,
    )
    assert result.value == (2.0 + 1.0 - 0.5) / 3
    assert result.format_ok
    assert result.num_statements == 3
    assert result.num_cited == 2
    assert result.mean_cite_score == 1.5


def test_summarize_averages_breakdowns():
    results = [
        dataclasses.asdict(reward.RewardResult(-1.0, False)),
        dataclasses.asdict(reward.RewardResult(1.0, True, 4, 2, 2.0)),
        dataclasses.asdict(reward.RewardResult(0.0, True, 2, 2, 0.0)),
    ]
    metrics = reward.summarize(results)
    assert metrics["selfcite/reward"] == 0.0
    assert metrics["selfcite/format_ok_rate"] == 2 / 3
    assert metrics["selfcite/num_statements"] == 3.0
    assert metrics["selfcite/cited_fraction"] == 0.75
    assert metrics["selfcite/cite_score"] == 1.0


def test_all_uncited_statements_score_the_penalty():
    stmts = [_stmt(), _stmt()]
    result = reward.combine(
        answer="...", statements=stmts, scores=[None, None], uncited_penalty=-0.5
    )
    assert result.value == -0.5
    assert result.mean_cite_score is None


def test_statements_are_weighted_by_length():
    # A short cited statement cannot offset a long uncited answer.
    stmts = [_stmt({0}, text="a" * 10), _stmt(text="b" * 90)]
    result = reward.combine(
        answer="...", statements=stmts, scores=[2.0, None], uncited_penalty=-0.5
    )
    assert result.value == (10 * 2.0 + 90 * -0.5) / 100
