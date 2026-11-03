import { Bar, BarChart, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { AspectStat } from "../types";

// Diverging bars: positive mentions to the right, negative to the left.
export default function AspectChart({ aspects }: { aspects: AspectStat[] }) {
  const rows = aspects.slice(0, 8).map((a) => ({ aspect: a.aspect, Positive: a.positive, Negative: -a.negative }));
  // Symmetric whole-number axis, so "0" sits in the middle even with very few mentions.
  const max = Math.max(1, ...aspects.slice(0, 8).map((a) => Math.max(a.positive, a.negative)));
  const summary = aspects.slice(0, 8).map((a) => `${a.aspect}: ${a.positive} positive, ${a.negative} negative`).join("; ");
  if (!rows.length) return <p className="muted small">No aspects found in these reviews.</p>;
  return (
    <section className="card" aria-labelledby="asp-title">
      <h2 id="asp-title">3. What people praise and complain about</h2>
      <p className="muted small">Each sentence is scored separately, and suspected fake reviews are left out.</p>
      <div role="img" aria-label={`Aspect sentiment. ${summary}`}>
        <ResponsiveContainer width="100%" height={Math.max(220, rows.length * 36)}>
          <BarChart data={rows} layout="vertical" stackOffset="sign" margin={{ left: 20 }}>
            <XAxis type="number" domain={[-max, max]} allowDecimals={false}
              ticks={[...new Set([-max, -Math.round(max / 2), 0, Math.round(max / 2), max])]}
              tickFormatter={(v: number) => String(Math.abs(v))} />
            <YAxis type="category" dataKey="aspect" width={150} tick={{ fontSize: 12 }} />
            <Tooltip formatter={(v: number, n: string) => [Math.abs(v), n]} />
            <Legend />
            <Bar dataKey="Negative" stackId="s" fill="#ef4444" />
            <Bar dataKey="Positive" stackId="s" fill="#22c55e" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <details className="steps">
        <summary>Example quotes per aspect</summary>
        <ul className="quotes">
          {aspects.slice(0, 8).map((a) => (
            <li key={a.aspect}>
              <strong>{a.aspect}</strong>
              {a.best && a.best.positive_prob >= 0.5 && <div className="pos">+ “{a.best.sentence}” <span className="muted">#{a.best.review_id}</span></div>}
              {a.worst && a.worst.positive_prob < 0.5 && <div className="neg">− “{a.worst.sentence}” <span className="muted">#{a.worst.review_id}</span></div>}
            </li>
          ))}
        </ul>
      </details>
    </section>
  );
}
