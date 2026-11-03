"""Demo data: 1,000 reviews of 4 product categories, taken only from the held-out
TEST split of the Salminen et al. fake-review dataset, so the app never shows
the models reviews they were trained on.

The analysis of the sample is precomputed at build time (python -m app.sample),
because the free server has a fraction of one CPU and the transformer would make
the first visitor wait.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.ml import aspects, models

DATA = Path(__file__).resolve().parent / "data"
SAMPLE = DATA / "sample_reviews.json"
PRECOMPUTED = models.MODEL_DIR / "sample_analysis.json"


@lru_cache(maxsize=1)
def reviews() -> list[dict]:
    return json.loads(SAMPLE.read_text())


def products() -> list[str]:
    return sorted({r["product"] for r in reviews()})


def for_product(product: str) -> list[dict]:
    return [r for r in reviews() if r["product"] == product]


def _analyze_all() -> dict:
    out = {}
    for p in products():
        rs = for_product(p)
        res = aspects.analyze(rs, models.fake_probability([r["text"] for r in rs]))
        truth = {r["id"]: r["true_label"] for r in rs}
        hits = sum((x["fake_prob"] >= aspects.FAKE_THRESHOLD) == (truth[x["id"]] == "fake") for x in res["reviews"])
        res["summary"]["detector_accuracy_on_sample"] = round(hits / len(rs), 3)
        out[p] = res
    return out


@lru_cache(maxsize=1)
def analysis() -> dict:
    if PRECOMPUTED.exists():
        data = json.loads(PRECOMPUTED.read_text())
        if data.get("detector") == models.detector_name():
            return data["products"]
    return _analyze_all()


if __name__ == "__main__":
    PRECOMPUTED.parent.mkdir(parents=True, exist_ok=True)
    PRECOMPUTED.write_text(json.dumps({"detector": models.detector_name(), "products": _analyze_all()}))
    print(f"wrote {PRECOMPUTED}")
