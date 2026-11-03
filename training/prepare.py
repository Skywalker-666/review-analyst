"""Download the labeled fake-review dataset and make train/test splits + the demo sample.

Dataset: Salminen et al. (2022), "Creating and detecting fake reviews of online
products", ~40k Amazon reviews. label OR = original (human), CG = computer-generated.
Source: https://osf.io/tyue9/ (file 3vds7), licensed CC BY 4.0.

Outputs (training/data/ is git-ignored, the sample is committed):
  training/data/train.csv, training/data/test.csv   stratified 80/20 split, seed 42
  backend/app/data/sample_reviews.json               demo reviews, drawn ONLY from test
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "fake_reviews.csv"
URL = "https://osf.io/download/3vds7/"
SAMPLE_OUT = ROOT.parent / "backend" / "app" / "data" / "sample_reviews.json"
DEMO_CATEGORIES = {
    "Electronics_5": "Electronics",
    "Clothing_Shoes_and_Jewelry_5": "Clothing & Shoes",
    "Home_and_Kitchen_5": "Home & Kitchen",
    "Pet_Supplies_5": "Pet Supplies",
}
PER_CATEGORY = 250


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def main() -> None:
    DATA.mkdir(exist_ok=True)
    if not RAW.exists():
        print("downloading dataset ...")
        urllib.request.urlretrieve(URL, RAW)
    df = pd.read_csv(RAW).rename(columns={"text_": "text"})
    df["text"] = df["text"].map(clean)
    df = df[df["text"].str.len() >= 10].drop_duplicates("text").reset_index(drop=True)
    df["is_fake"] = (df["label"] == "CG").astype(int)
    print(f"{len(df)} reviews, {df['is_fake'].mean():.1%} fake, {df['category'].nunique()} categories")

    strat = df["category"] + "_" + df["label"]
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=strat)
    train.to_csv(DATA / "train.csv", index=False)
    test.to_csv(DATA / "test.csv", index=False)
    print(f"train {len(train)}  test {len(test)}")

    rows = []
    for cat, nice in DEMO_CATEGORIES.items():
        part = test[test["category"] == cat].sample(PER_CATEGORY, random_state=7)
        for _, r in part.iterrows():
            rows.append({"product": nice, "rating": int(r["rating"]), "text": r["text"],
                         "true_label": "fake" if r["is_fake"] else "real"})
    for i, r in enumerate(rows):
        r["id"] = i + 1
    SAMPLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_OUT.write_text(json.dumps(rows, ensure_ascii=False))
    print(f"wrote {len(rows)} demo reviews -> {SAMPLE_OUT}")


if __name__ == "__main__":
    main()
