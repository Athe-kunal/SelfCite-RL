"""Trims a document to a length budget, keeping sentences near key ones.

Ported from SelfCite's `training/trim.py`. Used during data preparation:
sentences cited by the LongCite-45k reference answer are "key" and always
kept; other sentences are removed farthest-first until the budget fits.
"""

from collections.abc import Collection, Sequence


def trim_sents_by_key_distance(
    sents: Sequence[str],
    sent_lens: Sequence[int],
    key_sents_ids: Collection[int],
    max_len: int,
) -> tuple[list[str], set[int]]:
    """Removes non-key sentences farthest from any key sentence.

    Args:
        sents: All sentences, in order.
        sent_lens: Length (e.g. token count) of each sentence.
        key_sents_ids: Indices of sentences that must be kept.
        max_len: Total length budget.

    Returns:
        A tuple of (kept sentences in original order, removed indices).

    Raises:
        ValueError: If `sents` and `sent_lens` differ in length.
    """
    n = len(sents)
    if len(sent_lens) != n:
        raise ValueError("sents and sent_lens must have the same length.")

    total_len = sum(sent_lens)
    if total_len <= max_len:
        return list(sents), set()

    # Index distance from each sentence to its nearest key sentence.
    dist = [0 if i in key_sents_ids else float("inf") for i in range(n)]
    for order in (range(n), reversed(range(n))):
        last_key = None
        for i in order:
            if dist[i] == 0:
                last_key = i
            elif last_key is not None:
                dist[i] = min(dist[i], abs(i - last_key))

    non_key = [i for i in range(n) if i not in key_sents_ids]
    non_key.sort(key=lambda i: dist[i], reverse=True)

    removed = set()
    for idx in non_key:
        if total_len <= max_len:
            break
        total_len -= sent_lens[idx]
        removed.add(idx)

    kept = [sents[i] for i in range(n) if i not in removed]
    return kept, removed
