"""Aspect-based sentiment: which product aspects people praise or complain about.

1. Split each review into sentences.
2. Tag sentences with aspects using a keyword lexicon (battery, price, fit ...).
3. Score each sentence with the trained sentiment model.
4. Aggregate per aspect, by default over likely-genuine reviews only.

Sentence-level scoring means one review can like the price and hate the battery.
"""
from __future__ import annotations

import re

from app.ml import models

ASPECTS: dict[str, tuple[str, ...]] = {
    "Quality": ("quality", "well made", "cheaply made", "flimsy", "sturdy", "material", "build", "solid"),
    "Price & value": ("price", "value", "worth", "money", "expensive", "cheap", "overpriced", "bargain"),
    "Size & fit": ("size", "fit", "fits", "fitting", "tight", "loose", "too small", "too big", "runs small", "runs large"),
    "Comfort": ("comfortable", "comfort", "uncomfortable", "soft", "itchy", "cushion"),
    "Durability": ("broke", "broken", "lasted", "durable", "fell apart", "stopped working", "wear", "torn", "ripped"),
    "Ease of use": ("easy to", "easy", "simple", "instructions", "setup", "set up", "install", "difficult", "hard to", "assemble"),
    "Looks & design": ("color", "colour", "looks", "look", "design", "pretty", "beautiful", "style", "cute"),
    "Shipping & packaging": ("shipping", "arrived", "delivery", "delivered", "package", "packaging", "box"),
    "Battery & power": ("battery", "batteries", "charge", "charging", "charger", "power"),
    "Sound & screen": ("sound", "audio", "volume", "speaker", "screen", "picture", "display", "bass"),
    "Seller & support": ("seller", "customer service", "return", "returned", "refund", "warranty", "replacement"),
}
_PATTERNS = {a: re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b", re.I)
             for a, kws in ASPECTS.items()}
_SPLIT = re.compile(r"(?<=[.!?])\s+|\s*;\s*")
FAKE_THRESHOLD = 0.5


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(text) if len(s.strip()) >= 8]


def tag(sentence: str) -> list[str]:
    return [a for a, p in _PATTERNS.items() if p.search(sentence)]


def aspect_terms(question: str) -> list[str]:
    """Aspects a question is about, e.g. 'how is the battery?' -> ['Battery & power']."""
    q = question.lower()
    hits = [a for a, p in _PATTERNS.items() if p.search(q) or a.split(" ")[0].lower() in q]
    return hits


def analyze(reviews: list[dict], fake_probs: list[float]) -> dict:
    """Per-review fake/sentiment scores plus the per-aspect summary."""
    sent_rows: list[tuple[int, str, list[str]]] = []  # (review index, sentence, aspects)
    for i, r in enumerate(reviews):
        for s in sentences(r["text"]):
            aspects = tag(s)
            if aspects:
                sent_rows.append((i, s, aspects))
    sent_pos = models.positive_probability([s for _, s, _ in sent_rows])
    review_pos = models.positive_probability([r["text"] for r in reviews])

    per_review = [{"id": r["id"], "fake_prob": round(fp, 4), "positive_prob": round(rp, 4), "aspects": []}
                  for r, fp, rp in zip(reviews, fake_probs, review_pos)]
    agg: dict[str, dict] = {}
    for (i, s, aspects), p in zip(sent_rows, sent_pos):
        genuine = fake_probs[i] < FAKE_THRESHOLD
        for a in aspects:
            if a not in per_review[i]["aspects"]:
                per_review[i]["aspects"].append(a)
            if not genuine:
                continue
            d = agg.setdefault(a, {"aspect": a, "mentions": 0, "positive": 0, "negative": 0,
                                   "best": None, "worst": None})
            d["mentions"] += 1
            d["positive" if p >= 0.5 else "negative"] += 1
            quote = {"review_id": reviews[i]["id"], "sentence": s[:240], "positive_prob": round(p, 3)}
            if d["best"] is None or p > d["best"]["positive_prob"]:
                d["best"] = quote
            if d["worst"] is None or p < d["worst"]["positive_prob"]:
                d["worst"] = quote

    aspects_out = sorted(agg.values(), key=lambda d: -d["mentions"])
    for d in aspects_out:
        d["score"] = round((d["positive"] - d["negative"]) / d["mentions"], 3)
    n_fake = sum(p >= FAKE_THRESHOLD for p in fake_probs)
    genuine_ratings = [r.get("rating") for r, p in zip(reviews, fake_probs)
                       if p < FAKE_THRESHOLD and r.get("rating")]
    all_ratings = [r["rating"] for r in reviews if r.get("rating")]
    return {
        "reviews": per_review,
        "aspects": aspects_out,
        "summary": {
            "total": len(reviews),
            "suspected_fake": int(n_fake),
            "fake_share": round(n_fake / len(reviews), 3) if reviews else 0.0,
            "avg_rating_all": round(sum(all_ratings) / len(all_ratings), 2) if all_ratings else None,
            "avg_rating_genuine": round(sum(genuine_ratings) / len(genuine_ratings), 2) if genuine_ratings else None,
            "detector": models.detector_name(),
        },
    }
