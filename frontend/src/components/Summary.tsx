import type { Analysis } from "../types";

const pct = (x: number) => `${Math.round(x * 100)}%`;
const stars = (x: number | null) => (x == null ? "–" : `${x.toFixed(2)} ★`);

export default function Summary({ name, analysis }: { name: string; analysis: Analysis }) {
  const s = analysis.summary;
  const top = analysis.aspects.filter((a) => a.mentions >= 3);
  const best = [...top].sort((a, b) => b.score - a.score)[0];
  const worst = [...top].sort((a, b) => a.score - b.score)[0];
  return (
    <section className="card" aria-labelledby="sum-title">
      <h2 id="sum-title">2. {name}</h2>
      <div className="stats">
        <div className="stat"><div className="stat-label">Reviews</div><div className="stat-value">{s.total}</div></div>
        <div className="stat accent">
          <div className="stat-label">Suspected fake</div>
          <div className="stat-value">{s.suspected_fake} <span className="muted small">({pct(s.fake_share)})</span></div>
        </div>
        <div className="stat">
          <div className="stat-label">Avg rating: all → genuine only</div>
          <div className="stat-value">{stars(s.avg_rating_all)} → {stars(s.avg_rating_genuine)}</div>
        </div>
        {s.detector_accuracy_on_sample != null && (
          <div className="stat">
            <div className="stat-label">Detector accuracy here</div>
            <div className="stat-value">{pct(s.detector_accuracy_on_sample)}</div>
          </div>
        )}
      </div>
      {best && worst && best.aspect !== worst.aspect && (
        <p className="small">
          Genuine reviewers like the <strong>{best.aspect.toLowerCase()}</strong> most and complain about the{" "}
          <strong>{worst.aspect.toLowerCase()}</strong> most.
        </p>
      )}
    </section>
  );
}
