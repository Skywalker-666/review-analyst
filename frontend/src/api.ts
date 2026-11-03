import type { Analysis, AskResult, ModelMetrics, Review } from "./types";

// Backend URL. Set VITE_API_URL at build time for the deployed site.
const API = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") || "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API}${path}`, init);
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const body = await r.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return r.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export interface ScoredReview extends Review {
  fake_prob?: number;
  positive_prob?: number;
}

export const api = {
  health: () => request<{ status: string; llm: string; detector: string }>("/api/health"),
  model: () => request<ModelMetrics>("/api/model"),
  products: () => request<{ products: string[] }>("/api/sample"),
  sample: (product: string) =>
    request<{ product: string; reviews: Review[]; analysis: Analysis }>(`/api/sample/${encodeURIComponent(product)}`),
  upload: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<{ reviews: Review[] }>("/api/upload", { method: "POST", body: fd });
  },
  analyze: (reviews: Review[]) =>
    request<Analysis>("/api/analyze", json({ reviews: reviews.map(({ id, text, rating }) => ({ id, text, rating })) })),
  ask: (question: string, reviews: ScoredReview[], excludeFake: boolean) =>
    request<AskResult>("/api/ask", json({
      question,
      exclude_fake: excludeFake,
      reviews: reviews.map(({ id, text, rating, fake_prob, positive_prob }) => ({ id, text, rating, fake_prob, positive_prob })),
    })),
};
