"""Splits documents into numbered chunks and builds SelfCite prompts.

Ported from `get_prompt` / `text_split_by_punctuation` in SelfCite's
`make_data_get_prompt.py`. A document is split into sentence-level chunks,
and each chunk is prefixed with `<C{i}>` so the model can cite it as `[i-j]`.

The `keep` argument lets the reward rebuild the same prompt with only some
chunks present, which is how SelfCite's context ablation works: chunk ids
stay the original ones even when other chunks are removed.

Added here: the prompt ends its instructions with a short example of the
format. SelfCite started from LongCite-8B, already fine-tuned on this format;
a base model has only the instruction's `{S}` placeholder to go on.
"""

from collections.abc import Container, Sequence
import re

from nltk.tokenize import punkt

_CJK_PUNCTUATION = r"([。；！？])"

PROMPT_TEMPLATE = (
    "Please answer the user's question based on the following document. "
    "When a sentence S in your response uses information from some chunks "
    "in the document (i.e., <C{s1}>-<C_{e1}>, <C{s2}>-<C{e2}>, ...), please "
    "append these chunk numbers to S in the format "
    '"<statement>{S}<cite>[{s1}-{e1}][{s2}-{e2}]...</cite></statement>". '
    "You must answer in the same language as the user's question. "
    "For example: <statement>The bridge opened in 1932.<cite>[3-3]</cite>"
    "</statement><statement>It carries six lanes of traffic.<cite>[7-8]"
    "[12-12]</cite></statement>"
    "\n\n[Document Start]\n%s\n[Document End]\n\n%s"
)


def _split_sentences(text: str) -> list[str]:
    """Splits text into stripped sentences (Punkt plus CJK punctuation).

    Args:
        text: The raw document.

    Returns:
        Non-empty, stripped sentences in document order.
    """
    pieces = punkt.PunktSentenceTokenizer().tokenize(text)
    pieces = sum([re.split(_CJK_PUNCTUATION, p) for p in pieces], [])
    # re.split keeps the punctuation as separate items; glue it back on.
    for i in range(1, len(pieces)):
        if re.fullmatch(_CJK_PUNCTUATION, pieces[i]):
            pieces[i - 1] += pieces[i]
            pieces[i] = ""
    pieces = [p for p in pieces if p]
    if len(pieces) == 1:
        pieces = text.split("\n\n")
    return [p.strip() for p in pieces if p.strip()]


def split_into_chunks(text: str) -> list[str]:
    """Splits a document into chunks that each run up to the next sentence.

    Each chunk starts where its sentence starts and ends where the next
    sentence starts, so trailing whitespace is kept and joining all chunks
    reproduces the document from its first sentence onward.

    Args:
        text: The raw document.

    Returns:
        The chunks, in order. Chunk `i` is cited as `[i-i]`.
    """
    starts = []
    pos = 0
    for sentence in _split_sentences(text):
        start = text.find(sentence, pos)
        assert start != -1, sentence
        starts.append(start)
        pos = start + len(sentence)
    ends = starts[1:] + [len(text)]
    return [text[st:ed] for st, ed in zip(starts, ends)]


def render_context(
    chunks: Sequence[str], keep: Container[int] | None = None
) -> str:
    """Renders chunks as `<C{i}>chunk` text.

    Args:
        chunks: All chunks of the document.
        keep: Chunk ids to include. `None` includes every chunk.

    Returns:
        The concatenated, numbered context.
    """
    return "".join(
        f"<C{i}>{chunk}"
        for i, chunk in enumerate(chunks)
        if keep is None or i in keep
    )


def build_prompt(
    chunks: Sequence[str],
    question: str,
    keep: Container[int] | None = None,
) -> str:
    """Builds the SelfCite user prompt for a (possibly ablated) document.

    Args:
        chunks: All chunks of the document.
        question: The user's question.
        keep: Chunk ids to include. `None` includes every chunk.

    Returns:
        The user-turn text, before any chat template is applied.
    """
    return PROMPT_TEMPLATE % (render_context(chunks, keep), question)
