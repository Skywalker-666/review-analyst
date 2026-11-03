"""Train the two classic models (scikit-learn, TF-IDF + logistic regression).

1. Sentiment: positive (4-5 stars) vs negative (1-2 stars). 3-star reviews are dropped.
   Applied per SENTENCE at runtime, so one review can praise the price and hate the battery.
2. Fake-review baseline: the same features predicting computer-generated vs original.
   This is the yardstick the fine-tuned transformer has to beat, and the fallback
   if the ONNX model is missing.

Run:  training/.venv/bin/python training/train_classic.py
Out:  training/artifacts/{sentiment.joblib, fake_baseline.joblib, classic_metrics.json}
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, make_pipeline

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"


def word_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")

    def polar(df: pd.DataFrame) -> pd.DataFrame:
        d = df[df["rating"] != 3].copy()
        d["pos"] = (d["rating"] >= 4).astype(int)
        return d

    # Human-written reviews only: computer-generated ones often have a star rating that
    # does not match their text, which is label noise for sentiment.
    human_train, human_test = train[train["is_fake"] == 0], test[test["is_fake"] == 0]
    s_train, s_test = polar(human_train), polar(human_test)
    sentiment = make_pipeline(word_tfidf(), LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
    sentiment.fit(s_train["text"], s_train["pos"])
    s_pred = sentiment.predict(s_test["text"])

    fake = make_pipeline(
        FeatureUnion([("w", word_tfidf()),
                      ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                            max_features=80000, sublinear_tf=True))]),
        LogisticRegression(max_iter=2000, C=4.0))
    fake.fit(train["text"], train["is_fake"])
    f_pred = fake.predict(test["text"])

    metrics = {
        "sentiment": {"model": "TF-IDF (1-2 grams) + logistic regression, trained on human reviews",
                      "test_size": len(s_test),
                      "accuracy": round(float(accuracy_score(s_test["pos"], s_pred)), 4),
                      "f1_negative": round(float(f1_score(s_test["pos"], s_pred, pos_label=0)), 4)},
        "fake_baseline": {"model": "TF-IDF (words + char 3-5) + logistic regression",
                          "test_size": len(test),
                          "accuracy": round(float(accuracy_score(test["is_fake"], f_pred)), 4),
                          "f1": round(float(f1_score(test["is_fake"], f_pred)), 4)},
    }
    joblib.dump(sentiment, OUT / "sentiment.joblib", compress=3)
    joblib.dump(fake, OUT / "fake_baseline.joblib", compress=3)
    (OUT / "classic_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
