"""miles rollout-log hook that adds SelfCite reward metrics to TensorBoard.

Used as `--custom-rollout-log-function-path
selfcite_rl.miles_logging.log_rollout_data`. It logs `selfcite/*` metrics
next to miles' default `rollout/*` ones and keeps the default logging.
"""

import argparse
from typing import Any

from miles.utils import metric_utils
from miles.utils.tracking_utils import tracking
from miles.utils.types import Sample

from selfcite_rl import reward

_STEP_KEY = "rollout/step"


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
    results = [s.metadata["selfcite"] for s in samples if "selfcite" in s.metadata]
    if results:
        metrics = reward.summarize(results)
        metrics[_STEP_KEY] = metric_utils.compute_rollout_step(args, rollout_id)
        tracking.log(args, metrics, step_key=_STEP_KEY)
    return False
