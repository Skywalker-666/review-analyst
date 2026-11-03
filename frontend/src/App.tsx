import { useEffect, useMemo, useState } from "react";
import { api, type ScoredReview } from "./api";
import AskPanel from "./components/AskPanel";
import AspectChart from "./components/AspectChart";
import ModelCard from "./components/ModelCard";
import ReviewsTable from "./components/ReviewsTable";
import SourcePanel from "./components/SourcePanel";
import Summary from "./components/Summary";
import type { Analysis, ModelMetrics, Review } from "./types";

export default function App() {
  const [name, setName] = useState("");
  const [reviews, setReviews] = useState<Review[]>([]);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [busy, setBusy] = useState(false);
  const [cited, setCited] = useState<number[]>([]);
  const [model, setModel] = useState<ModelMetrics | null>(null);
  const [status, setStatus] = useState<{ llm: string; detector: string } | null>(null);

  useEffect(() => {
    api.health().then(setStatus).catch(() => setStatus({ llm: "unreachable", detector: "unreachable" }));
    api.model().then(setModel).catch(() => undefined);
  }, []);

  // Merge each review with its scores, so questions can send them back.
  const scored = useMemo<(ScoredReview & { aspects?: string[] })[]>(() => {
    if (!analysis) return [];
    const byId = new Map(analysis.reviews.map((s) => [s.id, s]));
    return reviews.map((r) => ({ ...r, ...byId.get(r.id) }));
  }, [reviews, analysis]);

  return (
    <>
      <header className="header">
        <div>
          <h1>Product Review Analyst</h1>
          <p className="muted">Fake-review detection · aspect sentiment · answers with citations</p>
        </div>
        <div className="pills" aria-label="System status">
          <span className={`badge ${status?.detector === "transformer" ? "ok" : "warn"}`}>
            Detector: {status ? status.detector : "checking"}
          </span>
          <span className={`badge ${status?.llm === "configured" ? "ok" : "warn"}`}>
            LLM: {status ? (status.llm === "configured" ? "online" : status.llm) : "checking"}
          </span>
        </div>
      </header>
      <main className="layout">
        <div className="left">
          <SourcePanel busy={busy} setBusy={setBusy} onLoaded={(n, rs, a) => {
            setName(n); setReviews(rs); setAnalysis(a); setCited([]);
          }} />
          {analysis && <Summary name={name} analysis={analysis} />}
          {analysis && <AspectChart aspects={analysis.aspects} />}
          {analysis && <ReviewsTable reviews={scored} highlight={cited} />}
          <ModelCard m={model} />
        </div>
        <div className="right">
          <AskPanel reviews={scored} onCite={setCited} />
        </div>
      </main>
      <footer className="muted small footer">
        Demo project. Sample reviews: Salminen et al. (2022) fake-review dataset, test split only. Model output is a signal, not proof.
      </footer>
    </>
  );
}
