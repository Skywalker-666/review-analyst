import { useEffect, useRef, useState } from "react";
import { api, type ScoredReview } from "../api";
import type { AskResult } from "../types";
import Markdown from "./Markdown";

const SUGGESTIONS = [
  "What do people hate about it?",
  "What do people love most?",
  "Is the quality worth the price?",
  "Any problems with size or fit?",
];

interface Turn { q: string; r?: AskResult; error?: string }

export default function AskPanel({ reviews, onCite }: { reviews: ScoredReview[]; onCite: (ids: number[]) => void }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [excludeFake, setExcludeFake] = useState(true);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => setTurns([]), [reviews]);
  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [turns]);

  async function ask(text: string) {
    const q = text.trim();
    if (q.length < 3 || busy || !reviews.length) return;
    setInput("");
    setBusy(true);
    setTurns((t) => [...t, { q }]);
    try {
      const r = await api.ask(q, reviews, excludeFake);
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, r } : x)));
      onCite(r.citations.map((c) => c.id));
    } catch (e) {
      const error = e instanceof Error ? e.message : "unknown error";
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, error } : x)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card chat" aria-labelledby="ask-title">
      <h2 id="ask-title">5. Ask the reviews</h2>
      <label className="row small">
        <input type="checkbox" checked={excludeFake} onChange={(e) => setExcludeFake(e.target.checked)} />
        Ignore suspected fake reviews
      </label>
      <div className="messages" aria-live="polite">
        {!turns.length && (
          <div className="suggestions">
            {SUGGESTIONS.map((s) => (
              <button key={s} className="chip" disabled={!reviews.length || busy} onClick={() => ask(s)}>{s}</button>
            ))}
            {!reviews.length && <p className="muted small">Pick a sample product or upload reviews first.</p>}
          </div>
        )}
        {turns.map((t, i) => (
          <div key={i} className="turn">
            <div className="msg user"><p>{t.q}</p></div>
            {t.error && <div className="msg assistant error">Sorry, that failed: {t.error}</div>}
            {t.r && (
              <div className="msg assistant">
                <Markdown text={t.r.answer} />
                <p className="muted small">
                  <span className="badge">{t.r.mode === "llm" ? "LLM + retrieval" : t.r.mode === "offline" ? "offline" : "no match"}</span>
                  {" "}searched {t.r.retrieved.length} reviews
                  {t.r.excluded_fake ? `, ignored ${t.r.excluded_fake} suspected fake` : ""}
                </p>
                {t.r.notice && <p className="muted small">{t.r.notice}</p>}
                {t.r.citations.length > 0 && (
                  <details className="steps" open={i === turns.length - 1}>
                    <summary>Sources ({t.r.citations.length})</summary>
                    {t.r.citations.map((c) => (
                      <blockquote key={c.id} className="cite">
                        <strong>#{c.id}</strong>{c.rating ? ` · ${c.rating}★` : ""} {c.text.length > 300 ? `${c.text.slice(0, 300)}…` : c.text}
                      </blockquote>
                    ))}
                  </details>
                )}
              </div>
            )}
          </div>
        ))}
        {busy && <div className="msg assistant muted" role="status">Searching reviews…</div>}
        <div ref={end} />
      </div>
      <form className="row" onSubmit={(e) => { e.preventDefault(); ask(input); }}>
        <label htmlFor="ask-input" className="visually-hidden">Your question</label>
        <input id="ask-input" value={input} maxLength={300} disabled={!reviews.length || busy}
          placeholder={reviews.length ? "Ask about these reviews…" : "Load reviews first"}
          onChange={(e) => setInput(e.target.value)} />
        <button type="submit" disabled={!reviews.length || busy || input.trim().length < 3}>Ask</button>
      </form>
    </section>
  );
}
