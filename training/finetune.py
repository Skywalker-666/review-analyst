"""Fine-tune a small transformer (ELECTRA-small, 13.5M params) to detect computer-generated
reviews, then export it to ONNX and quantize to int8 so it runs on a 512 MB free server.

Run:  training/.venv/bin/python training/finetune.py
Out:  training/artifacts/{fake_detector.onnx, tokenizer.json, transformer_metrics.json}
CPU only. About 20 minutes on 16 cores. If the ONNX step fails, rerun just
training/quantize.py: it picks up the fp32 export and train_info.json.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import quantize
from quantize import BASE, MAX_LEN, OUT, ROOT, scores
BATCH = 32
EPOCHS = 2
LR = 1e-4
SEED = 42


def batches(n: int, size: int, shuffle: bool, rng: np.random.Generator):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, size):
        yield idx[i:i + size]


def predict_torch(model, tok, texts: list[str]) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), 128):
            enc = tok(texts[i:i + 128], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            probs.append(torch.softmax(model(**enc).logits, -1)[:, 1].numpy())
    return np.concatenate(probs)


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    OUT.mkdir(exist_ok=True)
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2)

    texts, labels = train["text"].tolist(), torch.tensor(train["is_fake"].values)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(texts) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / (0.06 * steps)) * max(0.0, (steps - s) / (steps * 0.94)))
    t0, step = time.time(), 0
    for epoch in range(EPOCHS):
        model.train()
        for b in batches(len(texts), BATCH, True, rng):
            enc = tok([texts[i] for i in b], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            loss = model(**enc, labels=labels[b]).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch + 1} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)

    y = test["is_fake"].values
    torch_scores = scores(y, predict_torch(model, tok, test["text"].tolist()))
    print("torch fp32 test:", torch_scores, flush=True)

    fp32 = OUT / "fake_detector_fp32.onnx"
    model.eval()
    dummy = tok(["a sample review"], return_tensors="pt")
    torch.onnx.export(model, (dummy["input_ids"], dummy["attention_mask"]), fp32,
                      input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                                    "logits": {0: "b"}}, opset_version=17)
    (OUT / "train_info.json").write_text(json.dumps({
        "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "train_size": len(train), "epochs": EPOCHS, "test_torch_fp32": torch_scores,
        "train_seconds": round(time.time() - t0)}, indent=2))
    quantize.finish(fp32)  # int8 quantization + test check + metrics
    print("done", flush=True)


if __name__ == "__main__":
    main()
