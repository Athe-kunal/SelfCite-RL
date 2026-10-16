"""Tests for selfcite_rl.scoring, using a fake tokenizer and SGLang server."""

import asyncio

from selfcite_rl import citations
from selfcite_rl import scoring


class _CharTokenizer:
    """One token per character; a minimal chat template."""

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                            enable_thinking):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return f"<u>{messages[0]['content']}</u><a>"


class _FakeServer:
    """Scores a token -1 if its character occurs in the prompt, else -5.

    So a statement is likely when its letters are in the visible document,
    like a model that grounds its answer in the context.
    """

    def __init__(self):
        self.payloads = []

    async def post(self, url, payload):
        self.payloads.append((url, payload))
        ids = payload["input_ids"]
        start = payload["logprob_start_len"]
        doc = set(ids[:start])
        items = [[None, ids[start], None]]
        items += [[-1.0 if t in doc else -5.0, t, None] for t in ids[start + 1:]]
        return {"meta_info": {"input_token_logprobs": items}}


def test_variant_prompt_uses_empty_think_template():
    text = scoring.build_variant_prompt(_CharTokenizer(), ["a. "], "q", None)
    assert text.startswith("<u>Please answer")
    assert text.endswith("q</u><a>")


def _score_single(answer, chunks):
    stmts = citations.parse_statements(answer, num_chunks=len(chunks))
    (score,) = asyncio.run(
        scoring.score_statements(
            stmts,
            answer=answer,
            chunks=chunks,
            query="?",
            tokenizer=_CharTokenizer(),
            post=_FakeServer().post,
            url="http://router/generate",
        )
    )
    return score


def test_correct_citation_scores_higher_than_wrong_one():
    # CJK characters occur nowhere in the template, only in the document.
    chunks = ["甲乙。", "丙丁。"]
    right = _score_single("<statement>甲乙<cite>[0-0]</cite></statement>", chunks)
    wrong = _score_single("<statement>甲乙<cite>[1-1]</cite></statement>", chunks)
    # Right: keep shows 甲乙 (-1 each), drop hides it (-5 each) -> +4.
    assert right == 4.0
    # Wrong: keep hides the evidence, drop still shows it -> -4.
    assert wrong == -4.0


def test_uncited_statement_is_not_scored():
    answer = "Some uncited text. <statement>xyz<cite></cite></statement>"
    stmts = citations.parse_statements(answer, num_chunks=1)
    server = _FakeServer()
    scores = asyncio.run(
        scoring.score_statements(
            stmts, answer=answer, chunks=["xyz. "], query="?",
            tokenizer=_CharTokenizer(), post=server.post, url="u",
        )
    )
    assert scores == [None, None]
    assert not server.payloads


def test_request_payload_sets_lora_and_zero_new_tokens():
    answer = "<statement>xyz<cite>[0-0]</cite></statement>"
    stmts = citations.parse_statements(answer, num_chunks=1)
    server = _FakeServer()
    asyncio.run(
        scoring.score_statements(
            stmts, answer=answer, chunks=["xyz. "], query="?",
            tokenizer=_CharTokenizer(), post=server.post, url="u",
            lora_path="miles_lora",
        )
    )
    assert len(server.payloads) == 2  # One keep and one drop request.
    for _, payload in server.payloads:
        assert payload["sampling_params"]["max_new_tokens"] == 0
        assert payload["lora_path"] == "miles_lora"
        assert payload["return_logprob"]
