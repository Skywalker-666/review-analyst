import type { ModelMetrics } from "../types";

const pct = (x?: number) => (x == null ? "–" : `${(x * 100).toFixed(1)}%`);

// The numbers come from a held-out test split the models never saw in training.
export default function ModelCard({ m }: { m: ModelMetrics | null }) {
  if (!m) return null;
  const t = m.transformer;
  const c = m.classic;
  return (
    <section className="card" aria-labelledby="model-title">
      <h2 id="model-title">How the models score</h2>
      <table>
        <thead><tr><th>Task</th><th>Model</th><th>Accuracy</th><th>F1</th></tr></thead>
        <tbody>
          {t && (
            <tr>
              <td>Fake review {m.detector === "transformer" && <span className="badge ok">in use</span>}</td>
              <td>Fine-tuned {t.model.split("/").pop()} ({t.params_millions}M params, int8 ONNX {t.onnx_int8_mb} MB)</td>
              <td>{pct(t.test_onnx_int8.accuracy)}</td><td>{pct(t.test_onnx_int8.f1)}</td>
            </tr>
          )}
          {c && (
            <tr>
              <td>Fake review {m.detector === "baseline" && <span className="badge ok">in use</span>}</td>
              <td>Baseline: {c.fake_baseline.model}</td>
              <td>{pct(c.fake_baseline.accuracy)}</td><td>{pct(c.fake_baseline.f1)}</td>
            </tr>
          )}
          {c && (
            <tr>
              <td>Sentiment</td><td>{c.sentiment.model}</td>
              <td>{pct(c.sentiment.accuracy)}</td><td>{pct(c.sentiment.f1_negative)} <span className="muted">(neg.)</span></td>
            </tr>
          )}
        </tbody>
      </table>
      <p className="muted small">
        Measured on a held-out test split ({t?.test_size ?? c?.fake_baseline.test_size} reviews) of Salminen et al. (2022).
        “Fake” here means computer-generated, which is what this dataset labels.
      </p>
    </section>
  );
}
