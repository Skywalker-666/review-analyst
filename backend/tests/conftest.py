"""Test setup: build tiny stand-in models in a temp dir, so tests need neither the
dataset nor the GitHub Release. The real models are evaluated in training/."""
import os
import tempfile
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

_DIR = Path(tempfile.mkdtemp(prefix="ra-models-"))
os.environ["MODEL_DIR"] = str(_DIR)
os.environ.pop("LLM_API_KEY", None)

POS = ["great battery life", "love the price", "very comfortable and soft", "works perfectly",
       "excellent quality", "fits well and looks great", "easy to set up", "good value for money"]
NEG = ["battery died fast", "terrible price for this", "uncomfortable and itchy", "stopped working",
       "cheaply made and flimsy", "runs small and tight", "hard to set up", "waste of money broke"]
FAKE = ["love this product great product would recommend", "great product love it",
        "nice product very good product", "this is a great product for the price love it"]
REAL = ["the strap snapped after two weeks of daily use at the gym",
        "my daughter wears these to school, the left shoe squeaks on tile",
        "replaced the hinge screw myself, now the lid closes fine",
        "arrived a day late but the box was intact and the charger works with my old phone"]

joblib.dump(make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(POS + NEG, [1] * 8 + [0] * 8),
            _DIR / "sentiment.joblib")
joblib.dump(make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(FAKE + REAL, [1] * 4 + [0] * 4),
            _DIR / "fake_baseline.joblib")
(_DIR / "classic_metrics.json").write_text('{"sentiment": {"accuracy": 0.9}, "fake_baseline": {"accuracy": 0.9}}')
