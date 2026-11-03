"""Retrieval for RAG: BM25 keyword search over reviews, with two small twists.

- Aspect expansion: a question about "battery" also matches "charge", "charging" ...
- Intent rerank: "what do people hate ..." prefers negative reviews, "love" positive.

BM25 needs no extra model or vector database, so it fits a free 512 MB server and
is easy to explain. Implemented here in ~40 lines instead of pulling a library.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from app.ml.aspects import ASPECTS, aspect_terms

_TOKEN = re.compile(r"[a-z0-9']+")
STOP = set("""a an the and or but if of to in on for with at by from is are was were be been it its this that
these those i me my we our you your he she they them their what which who how do does did have has had not no
so very just can could would should will about any all some more most than then there here as up out""".split())
NEGATIVE_INTENT = ("hate", "complain", "complaint", "bad", "worst", "problem", "issue", "negative",
                   "dislike", "wrong", "disappoint", "fail", "cons", "broke")
POSITIVE_INTENT = ("love", "like", "best", "good", "great", "positive", "praise", "pros", "recommend")


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP and len(t) > 1]


def intent(question: str) -> str | None:
    q = question.lower()
    if any(w in q for w in NEGATIVE_INTENT):
        return "negative"
    if any(w in q for w in POSITIVE_INTENT):
        return "positive"
    return None


def search(question: str, reviews: list[dict], k: int = 8, k1: float = 1.5, b: float = 0.75) -> list[dict]:
    """Return up to k reviews ranked for the question. Each review needs id and text;
    positive_prob (from analysis) enables the intent rerank."""
    if not reviews:
        return []
    q_terms = tokens(question)
    for a in aspect_terms(question):
        q_terms += [t for kw in ASPECTS[a] for t in tokens(kw)]
    q_terms = list(dict.fromkeys(q_terms))
    docs = [tokens(r["text"]) for r in reviews]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n or 1.0
    df = Counter(t for d in docs for t in set(d))
    want = intent(question)

    scored = []
    for r, d in zip(reviews, docs):
        tf = Counter(d)
        s = 0.0
        for t in q_terms:
            if tf[t]:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        if s <= 0:
            continue
        pos = r.get("positive_prob")
        if want and pos is not None:
            s *= 1 + (1 - pos if want == "negative" else pos)
        scored.append((s, r))
    if not scored and want:
        # "What do people hate?" with no keyword hits: fall back to most negative/positive.
        key = (lambda r: r.get("positive_prob", 0.5)) if want == "negative" else (lambda r: -r.get("positive_prob", 0.5))
        return sorted(reviews, key=key)[:k]
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:k]]
