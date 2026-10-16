"""miles log hooks that add SelfCite reward metrics to TensorBoard.

Used as
  --custom-rollout-log-function-path selfcite_rl.miles_logging.log_rollout_data
  --custom-eval-rollout-log-function-path
      selfcite_rl.miles_logging.log_eval_rollout_data
They log `selfcite/*` and `eval/<name>/selfcite/*` metrics next to miles'
default ones and keep the default logging.
"""

import argparse
from typing import Any

from miles.utils import metric_utils
from miles.utils.tracking_utils import tracking
from miles.utils.types import Sample

from selfcite_rl import reward

_STEP_KEY = "rollout/step"
_EVAL_STEP_KEY = "eval/step"


def _breakdowns(samples: list[Sample]) -> list[dict[str, Any]]:
    return [s.metadata["selfcite"] for s in samples if "selfcite" in s.metadata]


def log_rollout_data(
    rollout_id: int,
    args: argparse.Namespace,
    samples: list[Sample],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    """Logs the reward breakdown stored by `miles_reward.reward`.

    Args:
        rollout_id: Index of the rollout.
        args: miles arguments.
        samples: The rollout's samples.
        rollout_extra_metrics: Unused; miles logs these itself.
        rollout_time: Unused; miles logs this itself.

    Returns:
        False, so miles still runs its default logging.
    """
    del rollout_extra_metrics, rollout_time
    results = _breakdowns(samples)
    if results:
        metrics = reward.summarize(results)
        metrics[_STEP_KEY] = metric_utils.compute_rollout_step(args, rollout_id)
        tracking.log(args, metrics, step_key=_STEP_KEY)
    return False


def log_eval_rollout_data(
    rollout_id: int,
    args: argparse.Namespace,
    data: dict[str, dict[str, Any]],
    extra_metrics: dict[str, Any] | None,
) -> bool:
    """Logs the reward breakdown of each eval dataset.

    Args:
        rollout_id: Index of the rollout the eval ran after.
        args: miles arguments.
        data: Eval results keyed by dataset name; each may hold `samples`.
        extra_metrics: Unused; miles logs these itself.

    Returns:
        False, so miles still runs its default logging.
    """
    del extra_metrics
    metrics = {}
    for name, result in data.items():
        results = _breakdowns(result.get("samples") or [])
        if results:
            for key, value in reward.summarize(results).items():
                metrics[f"eval/{name}/{key}"] = value
    if metrics:
        metrics[_EVAL_STEP_KEY] = metric_utils.compute_rollout_step(args, rollout_id)
        tracking.log(args, metrics, step_key=_EVAL_STEP_KEY)
    return False
