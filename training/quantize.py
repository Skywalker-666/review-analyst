"""Quantize the exported ONNX model to int8, check it on the test split, write metrics.

finetune.py calls this at the end. It can also be run on its own when an
fp32 export already exists (training/artifacts/fake_detector_fp32.onnx):

  training/.venv/bin/python training/quantize.py
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer

BASE = "google/electra-small-discriminator"
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
MAX_LEN = 128


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    return {"accuracy": round(float(accuracy_score(y, pred)), 4), "f1": round(float(f1_score(y, pred)), 4)}


def predict_onnx(path: Path, tok, texts: list[str]) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    probs = []
    for i in range(0, len(texts), 128):
        enc = tok(texts[i:i + 128], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="np")
        logits = sess.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                       "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
        e = np.exp(logits - logits.max(-1, keepdims=True))
        probs.append((e / e.sum(-1, keepdims=True))[:, 1])
    return np.concatenate(probs)


def finish(fp32: Path = OUT / "fake_detector_fp32.onnx") -> dict:
    # onnxruntime writes temp files; use a directory we own, not a possibly-stale $TMPDIR.
    work = OUT / "tmp"
    work.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(work)
    tempfile.tempdir = str(work)

    tok = AutoTokenizer.from_pretrained(BASE)
    int8 = OUT / "fake_detector.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    test = pd.read_csv(ROOT / "data" / "test.csv")
    onnx_scores = scores(test["is_fake"].values, predict_onnx(int8, tok, test["text"].tolist()))
    print("onnx int8 test:", onnx_scores, flush=True)

    tok.backend_tokenizer.save(str(OUT / "tokenizer.json"))
    info_path = OUT / "train_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    metrics = {"model": BASE, **info, "test_size": len(test), "max_len": MAX_LEN,
               "test_onnx_int8": onnx_scores, "onnx_int8_mb": round(int8.stat().st_size / 1e6, 1)}
    (OUT / "transformer_metrics.json").write_text(json.dumps(metrics, indent=2))
    fp32.unlink()
    for f in work.iterdir():
        f.unlink()
    work.rmdir()
    return metrics


if __name__ == "__main__":
    print(json.dumps(finish(), indent=2))
