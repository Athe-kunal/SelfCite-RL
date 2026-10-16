"""Converts THUDM/LongCite-45k into a miles prompt JSONL.

Ported from `truncate_irrelevant` in SelfCite's
`truncate_irrelevant_and_sampling.py`. For every example:

  1. Split the LongCite prompt into its `<C{i}>` chunks and the question.
  2. If the prompt is over budget, drop the chunks farthest from those the
     reference answer cites (see `trim.py`).
  3. Write `{"prompt": <user text>, "metadata": {"chunks", "query"}}`.

With `--eval_output`, a seeded random sample of `--num_eval` source rows
goes to a held-out eval file instead of the train file.

miles applies the chat template at load time (`--apply-chat-template`), and
the reward rebuilds drop/keep prompts from `metadata["chunks"]`.

Example:
    python -m selfcite_rl.prepare_data --output data/selfcite_train.jsonl \
        --eval_output data/selfcite_eval.jsonl --num_eval 100
"""

import argparse
from collections.abc import Callable
import contextlib
import json
import random
import re
from typing import Any

import datasets
import tqdm
import transformers

from selfcite_rl import chunking
from selfcite_rl import trim

_DOC_START = "\n\n[Document Start]\n"
_DOC_END = "\n[Document End]\n\n"
_CHUNK_TAG_RE = re.compile(r"<C\d+>")
_CITE_RE = re.compile(r"<cite>(.*?)</cite>", re.DOTALL)
_RANGE_RE = re.compile(r"\[(\d+)-(\d+)\]")

TokenCounter = Callable[[str], int]


def parse_longcite_prompt(prompt: str) -> tuple[list[str], str]:
    """Splits a LongCite-45k prompt into document chunks and the question.

    Args:
        prompt: The `prompt` field of a LongCite-45k row.

    Returns:
        A tuple of (chunks in order, question).
    """
    head, query = prompt.split(_DOC_END, 1)
    _, context = head.split(_DOC_START, 1)
    return _CHUNK_TAG_RE.split(context)[1:], query


def reference_chunk_ids(response: str) -> set[int]:
    """Returns every chunk id cited anywhere in a reference response."""
    ids = set()
    for cite_text in _CITE_RE.findall(response):
        for start, end in _RANGE_RE.findall(cite_text):
            ids.update(range(int(start), int(end) + 1))
    return ids


def build_example(
    row: dict[str, str],
    count_tokens: TokenCounter,
    count_prompt_tokens: TokenCounter,
    max_prompt_tokens: int,
) -> dict[str, Any] | None:
    """Builds one miles row, trimming the document to fit the budget.

    Args:
        row: A LongCite-45k row with `prompt` and `response`.
        count_tokens: Token count of a plain text fragment.
        count_prompt_tokens: Token count of a full user prompt after the
            chat template is applied.
        max_prompt_tokens: Budget for the templated prompt.

    Returns:
        The miles row, or `None` if the prompt cannot fit even after
        trimming.
    """
    chunks, query = parse_longcite_prompt(row["prompt"])
    doc_budget = max_prompt_tokens - count_prompt_tokens(
        chunking.build_prompt([], query)
    )
    # Count each chunk with its tag, since the tag is part of the prompt.
    chunk_lens = [count_tokens(f"<C{i}>{c}") for i, c in enumerate(chunks)]
    cited = {i for i in reference_chunk_ids(row["response"]) if i < len(chunks)}
    chunks, _ = trim.trim_sents_by_key_distance(
        chunks, chunk_lens, cited, doc_budget
    )

    prompt = chunking.build_prompt(chunks, query)
    # Per-chunk counts are approximate at token boundaries; check for real.
    if count_prompt_tokens(prompt) > max_prompt_tokens:
        return None
    return {"prompt": prompt, "metadata": {"chunks": chunks, "query": query}}


def main() -> None:
    """Writes the miles JSONL for LongCite-45k."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", maxsplit=1)[0]
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    # 32,768 total context minus 8,192 response tokens.
    parser.add_argument("--max_prompt_tokens", type=int, default=24576)
    parser.add_argument("--num_examples", type=int, default=None)
    parser.add_argument("--eval_output", default=None)
    parser.add_argument("--num_eval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_eval and args.eval_output is None:
        parser.error("--num_eval needs --eval_output")

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    def count_prompt_tokens(text: str) -> int:
        # Render to text, then encode: with tokenize=True newer transformers
        # return a BatchEncoding, whose len() is its number of keys.
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return count_tokens(templated)

    rows = datasets.load_dataset("THUDM/LongCite-45k", split="train")
    if args.num_examples is not None:
        rows = rows.select(range(min(args.num_examples, len(rows))))

    eval_ids = set(
        random.Random(args.seed).sample(
            range(len(rows)), min(args.num_eval, len(rows))
        )
    )
    counts = {"train": 0, "eval": 0}
    with contextlib.ExitStack() as stack:
        files = {
            "train": stack.enter_context(
                open(args.output, "w", encoding="utf-8")
            )
        }
        if args.eval_output is not None:
            files["eval"] = stack.enter_context(
                open(args.eval_output, "w", encoding="utf-8")
            )
        for idx, row in enumerate(tqdm.tqdm(rows, desc="Preparing")):
            example = build_example(
                row, count_tokens, count_prompt_tokens, args.max_prompt_tokens
            )
            if example is None:
                continue
            example["metadata"]["idx"] = idx
            split = "eval" if idx in eval_ids else "train"
            files[split].write(json.dumps(example, ensure_ascii=False) + "\n")
            counts[split] += 1
    print(
        f"Wrote {counts['train']} train / {counts['eval']} eval examples "
        f"from {len(rows)} rows"
    )


if __name__ == "__main__":
    main()
