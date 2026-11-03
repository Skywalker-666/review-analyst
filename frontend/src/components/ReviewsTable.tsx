import { useMemo, useState } from "react";
import type { ScoredReview } from "../api";
import { FAKE_THRESHOLD } from "../types";

type Filter = "all" | "fake" | "genuine";

export default function ReviewsTable({ reviews, highlight }: { reviews: (ScoredReview & { aspects?: string[] })[]; highlight: number[] }) {
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");
  const hasTruth = reviews.some((r) => r.true_label);
  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return reviews
      .filter((r) => filter === "all" || (filter === "fake") === ((r.fake_prob ?? 0) >= FAKE_THRESHOLD))
      .filter((r) => !q || r.text.toLowerCase().includes(q))
      .sort((a, b) => Number(highlight.includes(b.id)) - Number(highlight.includes(a.id)));
  }, [reviews, filter, query, highlight]);

  return (
    <section className="card" aria-labelledby="rev-title">
      <div className="row between">
        <h2 id="rev-title">4. Reviews</h2>
        <div className="row">
          <label htmlFor="rev-filter" className="visually-hidden">Filter</label>
          <select id="rev-filter" value={filter} onChange={(e) => setFilter(e.target.value as Filter)}>
            <option value="all">All</option>
            <option value="fake">Suspected fake</option>
            <option value="genuine">Likely genuine</option>
          </select>
          <label htmlFor="rev-search" className="visually-hidden">Search reviews</label>
          <input id="rev-search" placeholder="Search…" value={query} onChange={(e) => setQuery(e.target.value)} />
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th><th>Review</th><th>★</th><th>Fake?</th><th>Tone</th>
              {hasTruth && <th title="Label from the dataset, for checking the model">Truth</th>}
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => {
              const fake = (r.fake_prob ?? 0) >= FAKE_THRESHOLD;
              const right = r.true_label ? (r.true_label === "fake") === fake : null;
              return (
                <tr key={r.id} className={highlight.includes(r.id) ? "hl" : undefined}>
                  <td>{r.id}</td>
                  <td className="review-text">
                    {r.text}
                    {r.aspects?.length ? <div className="tags">{r.aspects.map((a) => <span key={a} className="tag">{a}</span>)}</div> : null}
                  </td>
                  <td>{r.rating ?? "–"}</td>
                  <td className={fake ? "neg" : "pos"}>{fake ? "⚠ " : ""}{Math.round((r.fake_prob ?? 0) * 100)}%</td>
                  <td className={(r.positive_prob ?? 0.5) >= 0.5 ? "pos" : "neg"}>
                    {(r.positive_prob ?? 0.5) >= 0.5 ? "positive" : "negative"}
                  </td>
                  {hasTruth && <td title={right ? "model was right" : "model was wrong"}>{r.true_label}{right ? " ✓" : " ✗"}</td>}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="muted small">Showing {shown.length} of {reviews.length}. “Fake?” is the model’s probability that a computer wrote the review.</p>
    </section>
  );
}
