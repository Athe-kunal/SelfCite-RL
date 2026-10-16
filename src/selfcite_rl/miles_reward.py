"""miles reward entry point: `--custom-rm-path selfcite_rl.miles_reward.reward`.

Per sample: strip thinking -> parse statements -> score each cited
statement with the live policy (through the miles SGLang router) -> combine.

Knobs, read from the environment so the launch script controls them:
  SELFCITE_UNCITED_PENALTY   Value of a statement citing nothing (-0.5).
  SELFCITE_MAX_SPANS         Merged citation ranges kept per statement (3).
  SELFCITE_MAX_SPAN_CHUNKS   Max chunks one range may cover (10).
  SELFCITE_MAX_CONCURRENCY   In-flight scoring requests per sample (64).

The reward breakdown is stored in `sample.metadata["selfcite"]` for logging.
"""

import argparse
import dataclasses
import functools
import os

from miles.utils import http_utils
from miles.utils.lora import utils as lora_utils
from miles.utils.types import Sample
import transformers

from selfcite_rl import citations
from selfcite_rl import reward as reward_lib
from selfcite_rl import scoring


@functools.cache
def _tokenizer(path: str) -> transformers.PreTrainedTokenizerBase:
    return transformers.AutoTokenizer.from_pretrained(
        path, trust_remote_code=True
    )


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


async def reward(args: argparse.Namespace, sample: Sample) -> float:
    """Computes the SelfCite GRPO reward for one rollout sample.

    Args:
        args: miles arguments (router address, checkpoint, LoRA flags).
        sample: A finished rollout; `metadata` holds `chunks` and `query`.

    Returns:
        The scalar reward.
    """
    chunks = sample.metadata["chunks"]
    query = sample.metadata["query"]
    answer = citations.strip_think(sample.response)

    statements, scores = [], []
    if answer is not None:
        statements = citations.parse_statements(
            answer,
            num_chunks=len(chunks),
            max_spans=_env_int("SELFCITE_MAX_SPANS", 3),
            max_span_chunks=_env_int("SELFCITE_MAX_SPAN_CHUNKS", 10),
        )
        lora_path = None
        if lora_utils.lora_rollout_enabled(args):
            lora_path = lora_utils.LORA_ADAPTER_NAME
        scores = await scoring.score_statements(
            statements,
            answer=answer,
            chunks=chunks,
            query=query,
            tokenizer=_tokenizer(args.hf_checkpoint),
            post=http_utils.post,
            url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
            "/generate",
            lora_path=lora_path,
            max_concurrency=_env_int("SELFCITE_MAX_CONCURRENCY", 64),
        )

    result = reward_lib.combine(
        answer,
        statements,
        scores,
        uncited_penalty=_env_float("SELFCITE_UNCITED_PENALTY", -0.5),
    )
    sample.metadata["selfcite"] = dataclasses.asdict(result)
    return result.value
