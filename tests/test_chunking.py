"""Tests for selfcite_rl.chunking."""

from selfcite_rl import chunking


def test_split_into_chunks_covers_text_after_first_sentence():
    text = "  First sentence. Second one! Third?"
    chunks = chunking.split_into_chunks(text)
    assert chunks == ["First sentence. ", "Second one! ", "Third?"]
    assert "".join(chunks) == text.lstrip()


def test_split_into_chunks_handles_cjk_punctuation():
    chunks = chunking.split_into_chunks("今天下雨。明天晴天！")
    assert chunks == ["今天下雨。", "明天晴天！"]


def test_split_into_chunks_falls_back_to_paragraphs():
    # Punkt finds one sentence here, so the original code splits on blank
    # lines instead.
    chunks = chunking.split_into_chunks("alpha beta\n\ngamma delta")
    assert chunks == ["alpha beta\n\n", "gamma delta"]


def test_render_context_numbers_all_chunks():
    assert chunking.render_context(["a. ", "b."]) == "<C0>a. <C1>b."


def test_render_context_keep_preserves_original_ids():
    chunks = ["a. ", "b. ", "c."]
    assert chunking.render_context(chunks, keep={0, 2}) == "<C0>a. <C2>c."
    assert chunking.render_context(chunks, keep=set()) == ""


def test_build_prompt_matches_selfcite_template():
    prompt = chunking.build_prompt(["a."], "What is a?")
    assert prompt.startswith("Please answer the user's question")
    assert prompt.endswith(
        "[Document Start]\n<C0>a.\n[Document End]\n\nWhat is a?"
    )
