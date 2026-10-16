"""SelfCite context-ablation scores, computed with SGLang prefill logprobs.

Ported from `query_log_prob_drop_ablating` / `query_log_prob_hold_pruning`
in upstream SelfCite's `longcite_modeling_llama.py`. For a statement S that
cites chunk set C, the model re-reads S under two edited documents:

  * keep: only the chunks in C       -> hold = mean log p(S | keep)
  * drop: every chunk except C       -> drop = mean log p(S | drop)

The score is `hold - drop` (SelfCite's `log_prob_drop_and_hold` reranker):
high when the cited chunks alone support S and S becomes unlikely without
them. Log probs come from SGLang `/generate` with `max_new_tokens=0`, the
same request shape miles uses in `generate_utils/prefill_logprobs.py`.

The answer is re-scored without its reasoning: each variant prompt uses the
chat template with `enable_thinking=False` (an empty think block for Qwen3),
so reasoning that quotes the document cannot leak into the drop context.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol

from selfcite_rl import chunking
from selfcite_rl import citations

PostFn = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class Tokenizer(Protocol):
    """The subset of a Hugging Face tokenizer this module uses."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]:
        ...

    def apply_chat_template(self, messages: list[dict[str, str]],
                            **kwargs: Any) -> str:
        ...


def build_variant_prompt(
    tokenizer: Tokenizer,
    chunks: Sequence[str],
    query: str,
    keep: set[int] | None,
) -> str:
    """Builds the templated prompt for one ablated document.

    Args:
        tokenizer: The policy's tokenizer.
        chunks: All chunks of the document.
        query: The user's question.
        keep: Chunk ids to show. `None` shows every chunk.

    Returns:
        Prompt text ending where the answer starts (after an empty think).
    """
    user_text = chunking.build_prompt(chunks, query, keep)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


async def mean_target_logprob(
    prefix_ids: list[int],
    target_ids: list[int],
    *,
    post: PostFn,
    url: str,
    lora_path: str | None = None,
) -> float:
    """Returns the mean log prob of `target_ids` given `prefix_ids`.

    Args:
        prefix_ids: Context tokens (prompt plus earlier answer text).
        target_ids: Tokens to score.
        post: Async HTTP POST returning the decoded JSON response.
        url: SGLang (or miles router) `/generate` URL.
        lora_path: Adapter name to score under, or `None` for full weights.

    Returns:
        The average per-token log probability of the target.

    Raises:
        ValueError: If the returned logprobs do not line up with the target.
    """
    payload = {
        "input_ids": prefix_ids + target_ids,
        "sampling_params": {"max_new_tokens": 0, "temperature": 0},
        "return_logprob": True,
        # SGLang returns logprobs from this position on, the first being
        # None; start one early so every target token gets a value.
        "logprob_start_len": len(prefix_ids) - 1,
    }
    if lora_path is not None:
        payload["lora_path"] = lora_path
    output = await post(url, payload)
    items = output["meta_info"]["input_token_logprobs"][-len(target_ids):]
    if [item[1] for item in items] != target_ids:
        raise ValueError("SGLang logprobs do not align with target tokens.")
    logprobs = [item[0] for item in items]
    if any(lp is None for lp in logprobs):
        raise ValueError("SGLang returned None for a target-token logprob.")
    return sum(logprobs) / len(logprobs)


async def score_statements(
    statements: Sequence[citations.Statement],
    *,
    answer: str,
    chunks: Sequence[str],
    query: str,
    tokenizer: Tokenizer,
    post: PostFn,
    url: str,
    lora_path: str | None = None,
    max_concurrency: int = 64,
) -> list[float | None]:
    """Computes `hold - drop` for every cited statement of one answer.

    Args:
        statements: Parsed statements of `answer`.
        answer: The answer text (think block removed) the offsets refer to.
        chunks: All chunks of the document.
        query: The user's question.
        tokenizer: The policy's tokenizer.
        post: Async HTTP POST returning the decoded JSON response.
        url: SGLang (or miles router) `/generate` URL.
        lora_path: Adapter name to score under, or `None` for full weights.
        max_concurrency: Max in-flight scoring requests.

    Returns:
        One entry per statement: the score, or `None` if it cites nothing
        (or its text has no tokens).
    """
    semaphore = asyncio.Semaphore(max_concurrency)
    all_ids = set(range(len(chunks)))

    async def score_variant(keep: set[int], stmt: citations.Statement) -> float:
        prompt = build_variant_prompt(tokenizer, chunks, query, keep)
        # Encode prefix and target separately so the target's token ids are
        # exactly the ones whose logprobs we read back.
        prefix_ids = tokenizer.encode(
            prompt + answer[:stmt.start], add_special_tokens=False
        )
        target_ids = tokenizer.encode(stmt.text, add_special_tokens=False)
        async with semaphore:
            return await mean_target_logprob(
                prefix_ids, target_ids, post=post, url=url, lora_path=lora_path
            )

    async def score_one(stmt: citations.Statement) -> float | None:
        if not stmt.chunk_ids or not stmt.text.strip():
            return None
        hold, drop = await asyncio.gather(
            score_variant(set(stmt.chunk_ids), stmt),
            score_variant(all_ids - stmt.chunk_ids, stmt),
        )
        return hold - drop

    return list(await asyncio.gather(*(score_one(s) for s in statements)))
