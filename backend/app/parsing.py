"""Parse an uploaded reviews CSV.

Needs a text column (review / text / review_text / body / content / comment).
Optional: a rating column (rating / stars / score), values 1-5.
"""
from __future__ import annotations

import csv
import io

MAX_REVIEWS = 300
MAX_TEXT = 2000
TEXT_KEYS = ("review", "text", "review_text", "reviewtext", "body", "content", "comment", "review body")
RATING_KEYS = ("rating", "stars", "star_rating", "score", "overall")


class ParseError(ValueError):
    pass


def _pick(header: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in header:
            return header[k]
    return None


def _rating(v: str | None) -> int | None:
    try:
        r = round(float((v or "").strip()))
    except ValueError:
        return None
    return r if 1 <= r <= 5 else None


def parse_reviews_csv(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ParseError("empty file")
    header = {h.strip().lower(): h for h in reader.fieldnames if h}
    text_col = _pick(header, TEXT_KEYS)
    if not text_col:
        raise ParseError("CSV needs a review text column (e.g. 'review' or 'text')")
    rating_col = _pick(header, RATING_KEYS)
    out: list[dict] = []
    for row in reader:
        body = " ".join((row.get(text_col) or "").split())
        if len(body) < 10:
            continue
        if len(out) >= MAX_REVIEWS:
            raise ParseError(f"too many reviews (max {MAX_REVIEWS})")
        out.append({"id": len(out) + 1, "text": body[:MAX_TEXT],
                    "rating": _rating(row.get(rating_col)) if rating_col else None})
    if not out:
        raise ParseError("no reviews found (each needs at least 10 characters)")
    return out
