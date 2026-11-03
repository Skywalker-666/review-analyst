"""Load the trained models once and run them.

Artifacts live in MODEL_DIR (default backend/models/). They are produced by
training/ and published as a GitHub Release, which the deploy downloads
(see app/fetch_models.py):
  fake_detector.onnx + tokenizer.json   fine-tuned ELECTRA-small, int8 ONNX
  fake_baseline.joblib                  TF-IDF + logistic regression (fallback)
  sentiment.joblib                      TF-IDF + logistic regression
  *_metrics.json                        held-out test scores shown in the UI
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[2] / "models"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # one thread on the server anyway
MAX_LEN = 128
BATCH = 32


class ModelsMissing(RuntimeError):
    pass


def _need(name: str) -> Path:
    p = MODEL_DIR / name
    if not p.exists():
        raise ModelsMissing(f"missing {p}. Run: python -m app.fetch_models")
    return p


@lru_cache(maxsize=1)
def _sentiment():
    return joblib.load(_need("sentiment.joblib"))


@lru_cache(maxsize=1)
def _transformer():
    """(session, tokenizer) or None when the ONNX model is not available."""
    if not (MODEL_DIR / "fake_detector.onnx").exists():
        return None
    import onnxruntime as ort
    from tokenizers import Tokenizer

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "1"))
    sess = ort.InferenceSession(str(MODEL_DIR / "fake_detector.onnx"), opts,
                                providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(_need("tokenizer.json")))
    tok.enable_truncation(MAX_LEN)
    tok.enable_padding()
    return sess, tok


@lru_cache(maxsize=1)
def _baseline():
    return joblib.load(_need("fake_baseline.joblib"))


def detector_name() -> str:
    try:
        return "transformer" if _transformer() else "baseline"
    except ModelsMissing:
        return "missing"


def fake_probability(texts: list[str]) -> list[float]:
    """Probability that each review is computer-generated (0..1)."""
    if not texts:
        return []
    t = _transformer()
    if t is None:
        return [float(p) for p in _baseline().predict_proba(texts)[:, 1]]
    sess, tok = t
    out: list[float] = []
    for i in range(0, len(texts), BATCH):
        enc = tok.encode_batch(texts[i:i + BATCH])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        logits = sess.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]
        e = np.exp(logits - logits.max(-1, keepdims=True))
        out.extend(float(p) for p in (e / e.sum(-1, keepdims=True))[:, 1])
    return out


def positive_probability(texts: list[str]) -> list[float]:
    """Probability that each text (a review or one sentence) is positive."""
    if not texts:
        return []
    return [float(p) for p in _sentiment().predict_proba(texts)[:, 1]]


def metrics() -> dict:
    out: dict = {"detector": detector_name()}
    for name in ("transformer_metrics.json", "classic_metrics.json"):
        p = MODEL_DIR / name
        if p.exists():
            out.update({name.replace("_metrics.json", ""): json.loads(p.read_text())})
    return out


def warm_up() -> None:
    """Load everything at startup so the first visitor doesn't wait."""
    positive_probability(["warm up"])
    fake_probability(["warm up"])
