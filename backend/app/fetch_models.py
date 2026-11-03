"""Download the trained model files from the GitHub Release into MODEL_DIR.

Used by the Render build. Locally you can instead copy training/artifacts/ to backend/models/.
Each file is checked against the SHA-256 listed in the release's SHA256SUMS file.

  python -m app.fetch_models
"""
from __future__ import annotations

import hashlib
import os
import sys

import httpx

from app.ml.models import MODEL_DIR

RELEASE = os.getenv("MODELS_RELEASE_URL",
                    "https://github.com/Skywalker-666/review-analyst/releases/download/models-v1")
FILES = ("fake_detector.onnx", "tokenizer.json", "fake_baseline.joblib", "sentiment.joblib",
         "transformer_metrics.json", "classic_metrics.json")


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=120) as c:
        sums = {}
        for line in c.get(f"{RELEASE}/SHA256SUMS").raise_for_status().text.splitlines():
            if line.strip():
                digest, name = line.split()
                sums[name] = digest
        for name in FILES:
            data = c.get(f"{RELEASE}/{name}").raise_for_status().content
            if hashlib.sha256(data).hexdigest() != sums.get(name):
                print(f"checksum mismatch for {name}", file=sys.stderr)
                return 1
            (MODEL_DIR / name).write_bytes(data)
            print(f"ok {name} ({len(data) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
