from app import llm
from app.rag import answer as rag
from app.rag.search import intent, search

REVIEWS = [
    {"id": 1, "text": "The battery barely lasts a day, charging is slow.", "positive_prob": 0.1, "fake_prob": 0.1},
    {"id": 2, "text": "Love the color and the design.", "positive_prob": 0.9, "fake_prob": 0.1},
    {"id": 3, "text": "Battery life is amazing, two days easily.", "positive_prob": 0.95, "fake_prob": 0.1},
    {"id": 4, "text": "Great battery great product love it.", "positive_prob": 0.99, "fake_prob": 0.95},
]


def test_intent():
    assert intent("What do people hate about it?") == "negative"
    assert intent("What do people love?") == "positive"
    assert intent("How big is it?") is None


def test_search_aspect_expansion_and_intent_rerank():
    ids = [r["id"] for r in search("what do people hate about the battery", REVIEWS[:3])]
    assert ids[0] == 1 and 2 not in ids
    # "charge" matches through the battery aspect lexicon
    assert [r["id"] for r in search("how is the charge", REVIEWS[:3])][0] == 1


def test_offline_answer_excludes_fake_and_cites():
    r = rag.answer("how is the battery", REVIEWS)
    assert r["mode"] == "offline" and r["excluded_fake"] == 1
    assert 4 not in r["retrieved"]
    assert r["citations"] and all(c["id"] in r["retrieved"] for c in r["citations"])


def test_include_fake_when_asked():
    r = rag.answer("how is the battery", REVIEWS, exclude_fake=False)
    assert 4 in r["retrieved"]


def test_no_match():
    r = rag.answer("does it come with a warranty card", REVIEWS[:3])
    assert r["mode"] == "none" and r["citations"] == []


def test_llm_hallucinated_citations_are_removed(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setattr(llm, "chat", lambda messages, tools=None, timeout=30.0: {
        "content": "Battery is weak [#1] but some love it [#3]. Also [#99] says it explodes."})
    r = rag.answer("how is the battery", REVIEWS)
    assert r["mode"] == "llm"
    assert "[#99]" not in r["answer"] and "[#1]" in r["answer"]
    assert [c["id"] for c in r["citations"]] == [1, 3]


def test_llm_failure_falls_back(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test")

    def boom(*a, **k):
        raise llm.LLMError("HTTP 429")

    monkeypatch.setattr(llm, "chat", boom)
    r = rag.answer("how is the battery", REVIEWS)
    assert r["mode"] == "offline" and "notice" in r


def test_loose_citation_formats_are_recognised(monkeypatch):
    """Seen live: the LLM wrote 'Review #1' instead of '[#1]', and no sources showed."""
    monkeypatch.setenv("LLM_API_KEY", "test")
    monkeypatch.setattr(llm, "chat", lambda messages, tools=None, timeout=30.0: {
        "content": "Review #1 says it dies fast. (#3) disagrees, and review 99 is made up. [#3] again."})
    r = rag.answer("how is the battery", REVIEWS)
    assert [c["id"] for c in r["citations"]] == [1, 3]
    assert "[#1]" in r["answer"] and "99" not in r["answer"]
    assert "[[" not in r["answer"] and "]]" not in r["answer"]


def test_citation_normalisation_formats():
    from app.rag.answer import _normalize_citations as norm
    assert norm("Review #156 says x") == "[#156] says x"
    assert norm("see (#7)") == "see [#7]"
    assert norm("grouped [#126, #209]") == "grouped [#126], [#209]"
    assert norm("grouped [126, 209 and 3]") == "grouped [#126], [#209], [#3]"
    assert norm("already ok [#12], [#7]") == "already ok [#12], [#7]"
    assert norm("(#126, #209)") == "[#126], [#209]"
