"""Writes miles rollout dumps as a Markdown file for human review.

Reads `runs/<run>/rollouts/<id>.pt` (written with `--save-debug-rollout-data`)
and, for each prompt, lists its samples best reward first: the reward
breakdown, the thinking (truncated), the answer, and every cited statement
next to the text of the chunks it cites.

Example:
    python -m selfcite_rl.dump_rollouts --run_dir runs/review-en-base \
        --output runs/review-en-base/review.md
"""

import argparse
import collections
import glob
import os
import re
from typing import Any

import torch

from selfcite_rl import citations

_ROLLOUT_FILE_RE = re.compile(r"(\d+)\.pt$")


def _load_samples(run_dir: str) -> list[dict[str, Any]]:
    """Loads every training rollout (not eval) of a run, in rollout order."""
    paths = [
        p for p in glob.glob(os.path.join(run_dir, "rollouts", "*.pt"))
        if _ROLLOUT_FILE_RE.search(os.path.basename(p))
        and not os.path.basename(p).startswith("eval_")
    ]
    paths.sort(key=lambda p: int(_ROLLOUT_FILE_RE.search(p).group(1)))
    samples = []
    for path in paths:
        samples.extend(torch.load(path, weights_only=False)["samples"])
    return samples


def _format_sample(
    sample: dict[str, Any], rank: int, max_think_chars: int, max_chunk_chars: int
) -> list[str]:
    """Renders one sample as Markdown lines."""
    meta = sample["metadata"]
    breakdown = meta.get("selfcite", {})
    chunks = meta["chunks"]
    response = sample["response"]
    think, _, _ = response.partition("</think>")
    think = think.replace("<think>", "").strip()
    answer = citations.strip_think(response)

    lines = [
        f"### Sample {rank}: reward {sample['reward']:.3f}",
        "",
        f"- format ok: {breakdown.get('format_ok')}, statements: "
        f"{breakdown.get('num_statements')}, cited: {breakdown.get('num_cited')}, "
        f"mean cite score: {breakdown.get('mean_cite_score')}",
        f"- response: {sample['response_length']} tokens, "
        f"thinking: {len(think)} chars",
        "",
        "<details><summary>Thinking</summary>",
        "",
        "```text",
        think[:max_think_chars] + (" …[truncated]" if len(think) > max_think_chars else ""),
        "```",
        "",
        "</details>",
        "",
    ]
    if answer is None:
        lines += ["**Answer:** _thinking never finished (hit the length limit)_", ""]
        return lines
    lines += ["**Answer:**", "", "```text", answer, "```", ""]

    statements = citations.parse_statements(answer, num_chunks=len(chunks))
    cited = [s for s in statements if s.chunk_ids]
    if cited:
        lines += ["**Cited statements:**", ""]
    for statement in cited:
        lines.append(f"- _{statement.text.strip()}_")
        for chunk_id in sorted(statement.chunk_ids):
            text = chunks[chunk_id].strip().replace("\n", " ")
            if len(text) > max_chunk_chars:
                text = text[:max_chunk_chars] + " …"
            lines.append(f"  - `C{chunk_id}`: {text}")
    lines.append("")
    return lines


def render(
    samples: list[dict[str, Any]], max_think_chars: int, max_chunk_chars: int
) -> str:
    """Renders samples grouped by prompt, best reward first.

    Args:
        samples: Sample dicts from miles rollout dumps.
        max_think_chars: Thinking is cut to this many characters.
        max_chunk_chars: Each cited chunk is cut to this many characters.

    Returns:
        The Markdown document.
    """
    by_prompt = collections.defaultdict(list)
    for sample in samples:
        by_prompt[sample["metadata"]["idx"]].append(sample)

    rewards = [s["reward"] for s in samples]
    ok = [s for s in samples if s["metadata"].get("selfcite", {}).get("format_ok")]
    lines = [
        "# Rollout review",
        "",
        f"{len(samples)} samples over {len(by_prompt)} prompts; "
        f"format ok {len(ok)}/{len(samples)}; "
        f"mean reward {sum(rewards) / len(rewards):.3f}.",
        "",
    ]
    for idx, group in by_prompt.items():
        group.sort(key=lambda s: s["reward"], reverse=True)
        meta = group[0]["metadata"]
        lines += [
            "---",
            "",
            f"## Prompt {idx} ({len(meta['chunks'])} chunks)",
            "",
            f"**Question:** {meta['query'].strip()}",
            "",
            "Rewards: " + ", ".join(f"{s['reward']:.2f}" for s in group),
            "",
        ]
        for rank, sample in enumerate(group, start=1):
            lines += _format_sample(sample, rank, max_think_chars, max_chunk_chars)
    return "\n".join(lines)


def main() -> None:
    """Writes the review Markdown for one run."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_think_chars", type=int, default=3000)
    parser.add_argument("--max_chunk_chars", type=int, default=400)
    args = parser.parse_args()

    samples = _load_samples(args.run_dir)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(render(samples, args.max_think_chars, args.max_chunk_chars))
    print(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
