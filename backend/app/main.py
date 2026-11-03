"""FastAPI app for the Product Review Analyst.

Stateless by design: the browser holds the reviews and their scores and sends
them with each question. Nothing is stored on the server.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import llm, sample
from app.ml import aspects, models
from app.parsing import MAX_REVIEWS, MAX_TEXT, ParseError, parse_reviews_csv
from app.rag.answer import answer

MAX_UPLOAD_BYTES = 1_000_000
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "20"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        models.warm_up()
        sample.analysis()
    except models.ModelsMissing as e:
        print(f"WARNING: {e}")
    yield


app = FastAPI(title="Product Review Analyst", version="1.0.0", lifespan=lifespan)

_origins = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

# Tiny in-memory rate limiter, so a public demo can't burn the LLM quota or the CPU.
_hits: dict = defaultdict(deque)


def _rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "Too many requests, wait a minute.")
    q.append(now)


class Review(BaseModel):
    id: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=MAX_TEXT)
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    fake_prob: Optional[float] = Field(default=None, ge=0, le=1)
    positive_prob: Optional[float] = Field(default=None, ge=0, le=1)


class AnalyzeRequest(BaseModel):
    reviews: List[Review] = Field(min_length=1, max_length=MAX_REVIEWS)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=300)
    reviews: List[Review] = Field(min_length=1, max_length=1000)
    exclude_fake: bool = True


def _models_ready() -> None:
    if models.detector_name() == "missing":
        raise HTTPException(503, "Models are not installed on the server.")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "llm": "configured" if llm.is_configured() else "offline",
            "detector": models.detector_name()}


@app.get("/api/model")
def model_info() -> dict:
    return models.metrics()


@app.get("/api/sample")
def sample_products() -> dict:
    return {"products": sample.products()}


@app.get("/api/sample/{product}")
def sample_product(product: str) -> dict:
    if product not in sample.products():
        raise HTTPException(404, "Unknown sample product.")
    _models_ready()
    rs = [{k: r[k] for k in ("id", "text", "rating", "true_label")} for r in sample.for_product(product)]
    return {"product": product, "reviews": rs, "analysis": sample.analysis()[product]}


@app.post("/api/upload")
async def upload(request: Request, file: UploadFile = File(...)) -> dict:
    _rate_limit(request)
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 1 MB).")
    try:
        reviews = parse_reviews_csv(raw.decode("utf-8-sig"))
    except UnicodeDecodeError:
        raise HTTPException(400, "File must be a UTF-8 CSV.")
    except ParseError as e:
        raise HTTPException(400, str(e))
    return {"reviews": reviews}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, request: Request) -> dict:
    _rate_limit(request)
    _models_ready()
    rs = [r.model_dump() for r in req.reviews]
    if len({r["id"] for r in rs}) != len(rs):
        raise HTTPException(400, "Review ids must be unique.")
    return aspects.analyze(rs, models.fake_probability([r["text"] for r in rs]))


@app.post("/api/ask")
def ask(req: AskRequest, request: Request) -> dict:
    _rate_limit(request)
    return answer(req.question, [r.model_dump() for r in req.reviews], req.exclude_fake)


if __name__ == "__main__":
    import uvicorn

    # Local default is loopback only. Render sets HOST=0.0.0.0 inside its container.
    uvicorn.run("app.main:app", host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
