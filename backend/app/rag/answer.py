"""Answer a question about the reviews with citations (RAG).

1. Retrieve the most relevant reviews (app/rag/search.py), by default only the
   ones the detector thinks are genuine.
2. Ask the LLM to answer ONLY from those reviews and cite them as [#id].
3. Check the citations: any [#id] that was not in the retrieved set is removed,
   so the answer can't point at a review that doesn't exist.
No API key, or the LLM fails: an extractive answer quoting the top reviews.
"""
from __future__ import annotations

import re

from app import llm
from app.ml.aspects import FAKE_THRESHOLD
from app.rag.search import intent, search

SYSTEM = """You analyse customer reviews of one product for a shopper.
Answer ONLY from the numbered reviews provided. Cite every claim with the review number in exactly
this format: [#12]. Do not write "Review #12".
If the reviews do not cover the question, say so plainly. Never invent reviews or numbers.
Keep it short: 2-5 sentences or a few bullets. Mention how many of the shown reviews agree when useful."""
_CITE = re.compile(r"\[#(\d+)\]")
# LLMs don't always follow the format: "Review #12", "review 12", "(#12)", "#12".
_LOOSE = re.compile(r"(?<!\[)\(?(?:\breviews?\s*)?#\s?(\d+)\)?(?!\])|\breview\s+(\d+)\b", re.I)


def _normalize_citations(text: str) -> str:
    return _LOOSE.sub(lambda m: f"[#{m.group(1) or m.group(2)}]", text)


def _context(found: list[dict]) -> str:
    lines = []
    for r in found:
        stars = f"{r['rating']}★ " if r.get("rating") else ""
        lines.append(f"[#{r['id']}] {stars}{r['text'][:600]}")
    return "\n".join(lines)


def _clean_citations(text: str, allowed: set[int]) -> tuple[str, list[int]]:
    used: list[int] = []

    def keep(m: re.Match) -> str:
        n = int(m.group(1))
        if n in allowed:
            if n not in used:
                used.append(n)
            return m.group(0)
        return ""

    return re.sub(r"\s{2,}", " ", _CITE.sub(keep, text)).strip(), used


def _offline(question: str, found: list[dict]) -> str:
    want = intent(question)
    neg = sum(1 for r in found if r.get("positive_prob", 0.5) < 0.5)
    head = f"I found {len(found)} relevant reviews ({len(found) - neg} positive, {neg} negative)."
    if want == "negative":
        head = f"I found {neg} critical reviews among the {len(found)} most relevant."
    quotes = [f"- \"{r['text'][:160].rstrip()}{'…' if len(r['text']) > 160 else ''}\" [#{r['id']}]"
              for r in found[:3]]
    return head + " The most relevant ones say:\n" + "\n".join(quotes)


def answer(question: str, reviews: list[dict], exclude_fake: bool = True) -> dict:
    pool = [r for r in reviews if not exclude_fake or r.get("fake_prob", 0.0) < FAKE_THRESHOLD]
    found = search(question, pool)
    base = {"retrieved": [r["id"] for r in found], "excluded_fake": len(reviews) - len(pool)}
    if not found:
        return {**base, "answer": "None of the reviews talk about that. Try different words.",
                "citations": [], "mode": "none"}

    allowed = {r["id"] for r in found}
    by_id = {r["id"]: r for r in found}
    mode, notice = "offline", None
    if llm.is_configured():
        try:
            msg = llm.chat([{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": f"Reviews:\n{_context(found)}\n\nQuestion: {question}"}])
            text, mode = msg.get("content") or "", "llm"
        except llm.LLMError as e:
            text, notice = _offline(question, found), f"LLM unavailable, used offline mode ({e})"
    else:
        text = _offline(question, found)
    text, used = _clean_citations(_normalize_citations(text), allowed)
    citations = [{"id": i, "text": by_id[i]["text"], "rating": by_id[i].get("rating"),
                  "fake_prob": by_id[i].get("fake_prob")} for i in used]
    out = {**base, "answer": text, "citations": citations, "mode": mode}
    if notice:
        out["notice"] = notice
    return out
