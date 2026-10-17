"""Parses `<statement>...<cite>[i-j]</cite></statement>` responses.

Ported from `postprocess` / `get_citations` in SelfCite's
`make_data_get_prompt.py`. The model only writes chunk-id ranges such as
`[3-5]` (also accepted: `[3]`, `[C3]`, `[C3-C5]`); this module turns each statement into its text plus the set of
chunk ids it cites, which is what the drop/hold reward needs.

Rules kept from SelfCite:
  * Text outside `<statement>` tags longer than 5 characters counts as an
    uncited statement.
  * Added here: a `<statement>` whose text is only a template placeholder
    such as `{S}` is ignored, so copying the prompt's
    `<statement>{S}<cite>...` template does not count as a cited statement.
  * Ranges that start past the last chunk or end before they start are
    dropped; ends are clamped to the last chunk.
  * Adjacent ranges are merged, then only the first 3 merged ranges are kept.
  * Added here: a merged range longer than `max_span_chunks` is truncated
    (SelfCite's length balancing used 10). The cap applies after merging, so
    chaining adjacent ranges cannot get around it.
"""

import dataclasses
import re

_THINK_END = "</think>"
# Qwen end-of-turn / end-of-text tokens; miles keeps them in the response text.
_END_TOKENS = ("<|im_end|>", "<|endoftext|>")
_STATEMENT_OPEN = "<statement>"
_STATEMENT_CLOSE = "</statement>"
_CITE_RE = re.compile(r"<cite>(.*?)</cite>", re.DOTALL)
# `[3-5]` is SelfCite's form; `[3]`, `[C3]` and `[C3-C5]` (after the prompt's
# `<C3>` tags) name the same chunks, so they count too.
_RANGE_RE = re.compile(r"\[C?(\d+)(?:-C?(\d+))?\]")
_MIN_UNCITED_CHARS = 5
_PLACEHOLDER_RE = re.compile(r"\{\w*\}")


@dataclasses.dataclass(frozen=True)
class Statement:
    """One statement of an answer.

    Attributes:
        text: The statement text with the `<cite>` block removed.
        chunk_ids: Ids of the chunks this statement cites (may be empty).
        has_cite_tag: True if the text was inside `<statement>` tags.
        start: Offset of the statement text in the answer string.
        end: End offset (exclusive) of the statement text in the answer.
    """

    text: str
    chunk_ids: frozenset[int]
    has_cite_tag: bool
    start: int
    end: int


def _strip_end_tokens(text: str) -> str:
    text = text.strip()
    while text.endswith(_END_TOKENS):
        for token in _END_TOKENS:
            text = text.removesuffix(token).strip()
    return text


def strip_think(response: str) -> str | None:
    """Removes a leading `<think>...</think>` block and trailing EOS text.

    Args:
        response: The raw model response.

    Returns:
        The answer after the think block, stripped and without trailing
        end-of-sequence tokens. The response itself (cleaned the same way) if
        it has no think block. `None` if thinking never finished, e.g. the
        response hit the length limit.
    """
    if _THINK_END in response:
        return _strip_end_tokens(response.rsplit(_THINK_END, 1)[1])
    if "<think>" in response:
        return None
    return _strip_end_tokens(response)


def _parse_chunk_ids(
    cite_text: str, num_chunks: int, max_spans: int, max_span_chunks: int
) -> frozenset[int]:
    """Turns `[s-e][s-e]...` into chunk ids using SelfCite's rules."""
    merged: list[list[int]] = []
    for match in _RANGE_RE.finditer(cite_text):
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start > num_chunks - 1 or end < start:
            continue
        end = min(end, num_chunks - 1)
        if merged and start == merged[-1][1] + 1:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    ids = set()
    for start, end in merged[:max_spans]:
        # Cap after merging, so chaining [1-1][2-2][3-3]... cannot dodge it.
        ids.update(range(start, min(end, start + max_span_chunks - 1) + 1))
    return frozenset(ids)


def parse_statements(
    answer: str,
    num_chunks: int,
    max_spans: int = 3,
    max_span_chunks: int = 10,
) -> list[Statement]:
    """Splits an answer into statements and their cited chunk ids.

    Args:
        answer: The answer text, with any think block already removed.
        num_chunks: Number of chunks in the document.
        max_spans: Max merged citation ranges kept per statement.
        max_span_chunks: Max chunks a single range may cover.

    Returns:
        Statements in answer order.
    """
    statements = []
    pos = 0
    while True:
        open_at = answer.find(_STATEMENT_OPEN, pos)
        if open_at == -1:
            open_at = len(answer)
        close_at = answer.find(_STATEMENT_CLOSE, open_at)

        outside = answer[pos:open_at]
        if len(outside.strip()) > _MIN_UNCITED_CHARS:
            statements.append(
                Statement(outside, frozenset(), False, pos, open_at)
            )
        if close_at == -1:
            break

        body_start = open_at + len(_STATEMENT_OPEN)
        body = answer[body_start:close_at]
        cite = _CITE_RE.search(body)
        text_end = body_start + (cite.start() if cite else len(body))
        text = answer[body_start:text_end]
        if text.strip() and not _PLACEHOLDER_RE.fullmatch(text.strip()):
            chunk_ids = frozenset()
            if cite:
                chunk_ids = _parse_chunk_ids(
                    cite.group(1), num_chunks, max_spans, max_span_chunks
                )
            statements.append(
                Statement(text, chunk_ids, True, body_start, text_end)
            )
        pos = close_at + len(_STATEMENT_CLOSE)
    return statements
