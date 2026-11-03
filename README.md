# Product Review Analyst

Paste or upload product reviews and get three things back:

1. **Which reviews look fake.** A fine-tuned transformer (ELECTRA-small) flags computer-generated reviews.
2. **What people actually praise and complain about**, broken down by aspect (battery, fit, price …) and scored
   sentence by sentence, with the suspected fakes left out.
3. **Answers to questions** like *"What do people hate about the sound?"*, grounded in the reviews and
   **citing them** as [#id] (RAG). Citations are checked, so an answer can't point at a review that doesn't exist.

**🚀 Live demo: [skywalker-666.github.io/review-analyst](https://skywalker-666.github.io/review-analyst/)** · API health: [review-analyst-api.onrender.com/api/health](https://review-analyst-api.onrender.com/api/health)

> The backend runs on Render's free tier and sleeps when idle, so the first request can take about 50 seconds.
> Pick a sample product, then ask a question.

## What it shows

| Area | What's inside |
|---|---|
| Deep learning | ELECTRA-small (13.5M params) fine-tuned on 32k labelled reviews, exported to ONNX and quantized to int8, so it runs on a free 512 MB server with no PyTorch |
| Classic ML | TF-IDF + logistic regression for sentiment, and a fake-review baseline the transformer has to beat |
| NLP | Sentence splitting, an aspect lexicon, sentence-level aspect sentiment |
| RAG | BM25 retrieval written from scratch, aspect-term query expansion, intent-aware reranking ("hate" → negative reviews first), citation validation |
| LLM | Any OpenAI-compatible API (Groq free tier by default), with an offline extractive fallback |
| Backend | FastAPI, Pydantic validation, rate limiting, CORS allowlist, upload size limits, stateless (stores nothing) |
| Frontend | React + TypeScript + Recharts: diverging aspect chart, filterable review table, Q&A with sources, keyboard and screen-reader friendly |
| MLOps | Reproducible training scripts, held-out test metrics shown in the app, model files shipped as a checksum-verified GitHub Release |
| CI/CD | GitHub Actions: tests and build on every push, frontend deployed to GitHub Pages, backend Blueprint for Render |

## Results (held-out test split, 8,081 reviews)

| Task | Model | Accuracy | F1 |
|---|---|---|---|
| Fake review | **Fine-tuned ELECTRA-small, int8 ONNX (14 MB)** | **97.6%** | **97.6%** |
| Fake review | Baseline: TF-IDF (words + char 3-5) + logistic regression | 94.8% | 94.8% |
| Sentiment (human reviews, 1-2★ vs 4-5★) | TF-IDF + logistic regression | 92.8% | 71.4% (negative class) |

Fine-tuning cuts the baseline's error rate by more than half (5.2% → 2.4%). Int8 quantization shrinks the model
from 54 MB to 14 MB with no accuracy loss (fp32 scored 97.2%). On the four demo products the detector scores 97-98%.

The sentiment model is trained on human-written reviews only. Computer-generated reviews often have a star rating
that doesn't match their text, and dropping them raised negative-class F1 from 0.51 to 0.71.

## Architecture

```
React (GitHub Pages)  ──HTTPS──▶  FastAPI (Render)
  pick sample / upload CSV           /api/sample/{product}   precomputed analysis of the demo data
  aspect chart, review table         /api/upload, /api/analyze
  ask a question  ─────────────────▶ /api/ask
                                        BM25 search (+ aspect expansion, intent rerank)
                                        → LLM answers from top reviews, cites [#id]
                                        → citations validated against what was retrieved
                                     models: ONNX transformer + 2 sklearn pipelines
```

The server keeps no state. The browser holds the reviews and their scores and sends them with each question.

## Run locally

```bash
# 1. Train (one time, CPU, about 25 minutes)
cd training
python3 -m venv .venv && .venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install transformers==4.46.3 onnx==1.17.0 onnxruntime==1.19.2 scikit-learn==1.6.1 pandas==2.2.3 numpy==2.0.2 tokenizers==0.20.3
.venv/bin/python prepare.py && .venv/bin/python train_classic.py && .venv/bin/python finetune.py

# 2. Backend (http://127.0.0.1:8000)
cd ../backend
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
mkdir -p models && cp ../training/artifacts/* models/
.venv/bin/python -m app.sample      # precompute the demo analysis
.venv/bin/python -m pytest -q
.venv/bin/python -m app.main

# 3. Frontend (http://127.0.0.1:5173)
cd ../frontend && npm ci && npm run dev
```

Set `LLM_API_KEY` (see `backend/.env.example`) for LLM answers. Without it, answers are extractive quotes.

## Deploy

1. GitHub: Settings → Pages → Source: GitHub Actions. Add the repo variable `API_URL` (the Render URL).
2. Render: New → Blueprint → this repo. Paste `LLM_API_KEY` when asked.

## Honest limits

- "Fake" here means **computer-generated** (the dataset's label), not paid human reviewers.
- The aspect lexicon is hand-written and English-only.
- BM25 matches words, not meaning. Embeddings would catch synonyms, but they don't fit in 512 MB alongside the transformer.

## Data

Salminen, J., Kandpal, C., Kamel, A. M., Jung, S., & Jansen, B. J. (2022). *Creating and detecting fake reviews of
online products.* Journal of Retailing and Consumer Services, 64. Dataset: [osf.io/tyue9](https://osf.io/tyue9/).
The demo uses 1,000 reviews from the held-out test split only.

## License

MIT for the code. The dataset and the sample reviews in `backend/app/data/` are © the dataset authors,
used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) with the attribution above.
