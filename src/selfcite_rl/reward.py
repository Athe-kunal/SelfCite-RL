"""Combines per-statement SelfCite scores into one GRPO reward.

Rules:
  * Format gate: if thinking never finished, or the answer has no
    `<statement>` tags, the reward is `FORMAT_PENALTY` (-1).
  * Otherwise, the reward is the mean over all statements, weighted by
    statement length in characters, of:
      - the statement's `hold - drop` score if it cites something, or
      - `uncited_penalty` if it cites nothing, so leaving statements
        uncited is not a way to dodge scoring.
    Length weighting stops one short cited statement from offsetting a long
    answer left uncited (all text outside `<statement>` tags is a single
    statement).

This module is pure; `miles_reward.py` wires it into miles.
"""

from collections.abc import Sequence
import dataclasses
from typing import Any

from selfcite_rl import citations

FORMAT_PENALTY = -1.0


@dataclasses.dataclass(frozen=True)
class RewardResult:
    """A reward and the numbers behind it, for logging.

    Attributes:
        value: The scalar reward used by GRPO.
        format_ok: Whether the response passed the format gate.
        num_statements: Statements in the answer.
        num_cited: Statements that cite at least one chunk.
        mean_cite_score: Mean `hold - drop` over cited statements, or `None`.
    """

    value: float
    format_ok: bool
    num_statements: int = 0
    num_cited: int = 0
    mean_cite_score: float | None = None


def combine(
    answer: str | None,
    statements: Sequence[citations.Statement],
    scores: Sequence[float | None],
    uncited_penalty: float = -0.5,
) -> RewardResult:
    """Computes the reward for one response.

    Args:
        answer: The answer after removing thinking, or `None` if thinking
            never finished.
        statements: Parsed statements of the answer.
        scores: `hold - drop` per statement, `None` where nothing is cited.
        uncited_penalty: Value of a statement that cites nothing.

    Returns:
        The reward with its breakdown.
    """
    if answer is None or not any(s.has_cite_tag for s in statements):
        return RewardResult(value=FORMAT_PENALTY, format_ok=False)

    cited = [score for score in scores if score is not None]
    values = [uncited_penalty if score is None else score for score in scores]
    weights = [max(len(s.text.strip()), 1) for s in statements]
    return RewardResult(
        value=sum(w * v for w, v in zip(weights, values)) / sum(weights),
        format_ok=True,
        num_statements=len(statements),
        num_cited=len(cited),
        mean_cite_score=sum(cited) / len(cited) if cited else None,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Averages reward breakdowns over a rollout for logging.

    Args:
        results: `dataclasses.asdict(RewardResult)` for each sample.

    Returns:
        Metrics keyed `selfcite/<name>`.
    """
    ok = [r for r in results if r["format_ok"]]
    scores = [r["mean_cite_score"] for r in ok if r["mean_cite_score"] is not None]
    return {
        "selfcite/reward": _mean([r["value"] for r in results]),
        "selfcite/format_ok_rate": _mean([float(r["format_ok"]) for r in results]),
        "selfcite/num_statements": _mean([r["num_statements"] for r in ok]),
        "selfcite/cited_fraction": _mean(
            [r["num_cited"] / r["num_statements"] for r in ok]
        ),
        "selfcite/cite_score": _mean(scores),
    }
