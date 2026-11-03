import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { Analysis, Review } from "../types";

interface Props {
  busy: boolean;
  setBusy: (b: boolean) => void;
  onLoaded: (name: string, reviews: Review[], analysis: Analysis, isSample: boolean) => void;
}

export default function SourcePanel({ busy, setBusy, onLoaded }: Props) {
  const [products, setProducts] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.products().then((r) => setProducts(r.products)).catch(() => setError("Backend is waking up, try again in a moment."));
  }, []);

  async function run(task: () => Promise<void>) {
    setError(null);
    setBusy(true);
    try {
      await task();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  const loadSample = (p: string) =>
    run(async () => {
      const r = await api.sample(p);
      onLoaded(`${p} (sample)`, r.reviews, r.analysis, true);
    });

  const upload = (f: File) =>
    run(async () => {
      const { reviews } = await api.upload(f);
      onLoaded(f.name, reviews, await api.analyze(reviews), false);
    });

  return (
    <section className="card" aria-labelledby="src-title">
      <h2 id="src-title">1. Pick reviews</h2>
      <p className="muted small">
        Sample products: 250 real Amazon-style reviews each, about half written by a computer. Or upload your own CSV
        with a <code>review</code> column and an optional <code>rating</code> column (max 150 reviews).
      </p>
      <div className="row">
        {products.map((p) => (
          <button key={p} className="secondary" disabled={busy} onClick={() => loadSample(p)}>{p}</button>
        ))}
        {!products.length && !error && <span className="muted small">Loading sample products…</span>}
      </div>
      <div className="row" style={{ marginTop: "0.75rem" }}>
        <label className="button" htmlFor="csv">Upload CSV</label>
        <input id="csv" ref={fileRef} type="file" accept=".csv,text/csv" className="visually-hidden" disabled={busy}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) upload(f);
            e.target.value = "";
          }} />
        {busy && <span className="muted small" role="status">Analysing… (first request can take ~50 s while the free server wakes up)</span>}
      </div>
      {error && <p className="error small" role="alert">{error}</p>}
    </section>
  );
}
