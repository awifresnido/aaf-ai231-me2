import { modelName, post, prettyIntent, prettyKey, useTick } from "../api";
import type { AppState } from "../types";

const RING_R = 120;
const RING_C = 2 * Math.PI * RING_R;
const LEVEL_FLOOR_DBFS = -70;

type Props = {
  state: AppState;
  level: number | null;
  phaseSince: { phase: string; at: number };
  presentation: boolean;
};

function ringTone(phase: string, accepted?: boolean) {
  if (phase === "disconnected") return "alarm";
  if (phase === "wake_detected" || phase === "listening") return "listen";
  if (phase === "inferencing") return "think";
  if (phase === "result") return accepted ? "ok" : "warn";
  return "idle";
}

export function AssistantCore({ state, level, phaseSince }: Props) {
  const phase = state.link.connected ? state.phase.phase ?? "idle" : "disconnected";
  const listening = phase === "listening";
  const vadMode = listening && state.phase.mode === "vad";
  useTick(100, listening || phase === "inferencing");
  const windowS = state.phase.window_s ?? state.edge?.window_s ?? 4;
  const elapsed = (Date.now() - phaseSince.at) / 1000;
  const remaining = listening && !vadMode && windowS > 0 ? Math.max(0, windowS - elapsed) : 0;
  const frac = listening && !vadMode && windowS > 0 ? remaining / windowS : phase === "idle" ? 0 : 1;
  const tone = ringTone(phase, state.phase.accepted);
  const wake = state.edge?.wake_engines.find((w) => w.id === state.edge?.active_wake);
  const manual = !wake || wake.kind === "manual";
  const wakeEv = [...state.timeline].reverse().find((ev) => ev.type === "wake_detected");
  const lv = level === null ? 0 : Math.max(0, Math.min(1, (level - LEVEL_FLOOR_DBFS) / -LEVEL_FLOOR_DBFS));
  const r = state.last_result;

  let headline = "Ready";
  let sub = manual ? "Press Wake, or replay a clip" : `Say “${wake?.phrase ?? "the wake word"}”, or press Wake`;
  if (phase === "disconnected") {
    headline = "Edge offline";
    sub = `Reconnecting to ${state.link.url}`;
  } else if (phase === "wake_detected") {
    headline = "Awake";
    const conf = wakeEv?.payload.confidence;
    sub = conf != null ? `Wake word heard (${Number(conf).toFixed(2)}). Speak after the tone` : "Speak after the tone";
  } else if (listening) {
    if (vadMode) {
      headline = "Listening…";
      sub = "I'll stop when you finish · max 4 s";
    } else {
      headline = windowS > 0 ? remaining.toFixed(1) : "Clip";
      sub = windowS > 0 ? "Listening, say the command" : "Reading replayed clip";
    }
  } else if (phase === "inferencing") {
    headline = "Classifying";
    sub = "VCM running on the edge";
  } else if (phase === "result" && r) {
    headline = r.accepted ? prettyIntent(r.primary.intent) : "Not recognised";
    sub = r.accepted ? r.primary.slot ?? "Done" : r.reject_reason ?? "";
  }

  return (
    <section className={`core tone-${tone} phase-${phase}`} aria-live="polite">
      {(phase === "wake_detected" || listening) && (
        <div className="awake-pill" role="status">
          <i aria-hidden /> {listening ? "Listening for your command" : "Device awake"}
        </div>
      )}
      <svg className="ring" viewBox="0 0 300 300" role="img" aria-label={`Assistant ${phase}`}>
        <circle className="ring-track" cx="150" cy="150" r={RING_R} />
        <circle
          className="ring-level"
          cx="150"
          cy="150"
          r={RING_R - 18}
          style={{ transform: `scale(${0.82 + 0.18 * lv})` }}
        />
        <circle
          className={`ring-arc ${phase === "inferencing" ? "spin" : ""}`}
          cx="150"
          cy="150"
          r={RING_R}
          strokeDasharray={phase === "inferencing" ? `${RING_C * 0.18} ${RING_C * 0.07}` : `${RING_C}`}
          strokeDashoffset={phase === "inferencing" ? 0 : RING_C * (1 - frac)}
        />
      </svg>
      <div className="core-text">
        <div className={`core-headline ${listening && !vadMode ? "digits" : ""}`}>{headline}</div>
        <div className="core-sub">{sub}</div>
      </div>
      <div className="core-actions">
        <button
          className="btn primary"
          onClick={() => post("/api/wake")}
          disabled={phase !== "idle"}
          title="Trigger the command window without the wake-word model"
        >
          Wake
        </button>
        <span className="hint">
          Wake source: {wake?.display_name ?? "—"}
        </span>
      </div>
    </section>
  );
}

function ConfidenceBar({ value, threshold }: { value: number; threshold: number }) {
  return (
    <div className="conf" aria-label={`Confidence ${(value * 100).toFixed(0)} percent`}>
      <div className="conf-fill" style={{ width: `${value * 100}%` }} data-pass={value >= threshold} />
      <div className="conf-thr" style={{ left: `${threshold * 100}%` }} title={`Threshold ${threshold.toFixed(2)}`} />
    </div>
  );
}

export function Decision({ state, presentation }: { state: AppState; presentation: boolean }) {
  const r = state.last_result;
  if (!r) {
    return (
      <section className="decision empty">
        <h2>Last command</h2>
        <p className="muted">No command yet. Press Wake, replay a WAV, or use Simulate in the engineering panel.</p>
      </section>
    );
  }
  const p = r.primary;
  const labels = state.ontology.leaf_labels;
  const setLabel = (cls: string | null) => post(`/api/commands/${r.command_id}/label`, { expected_class: cls });
  const logged = state.commands.find((c) => c.command_id === r.command_id);
  const expected = logged?.expected_class ?? r.expected_class;

  return (
    <section className="decision">
      <header className="decision-head">
        <h2>Last command</h2>
        <span className={`badge src-${r.source}`}>
          {r.source === "simulated" ? "Simulated, no model" : r.source === "replay" ? "Replayed WAV" : "Live audio"}
        </span>
      </header>
      <div className="intent-row">
        <div className="intent">{prettyIntent(p.intent)}</div>
        {p.slot && <div className="slot">{p.slot}</div>}
      </div>
      <div className="conf-row">
        <ConfidenceBar value={p.confidence} threshold={r.threshold} />
        <span className="num">{(p.confidence * 100).toFixed(1)}%</span>
      </div>
      <p className={`outcome ${r.outcome.executed ? "ok" : "warn"}`}>
        {r.outcome.executed ? r.outcome.action : `No action: ${r.outcome.detail ?? r.reject_reason ?? ""}`}
      </p>
      {!presentation && (
        <>
          <table className="topk">
            <tbody>
              {p.top_k.map(([k, v]) => (
                <tr key={k}>
                  <td>{prettyKey(k)}</td>
                  <td className="num">{(v * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
          <dl className="facts">
            <dt>Model</dt>
            <dd>
              {modelName(state.display_names, p.model_id)}
            </dd>
            <dt>Inference</dt>
            <dd className="num">
              {p.inference_ms.toFixed(1)} ms model + {p.feature_ms.toFixed(1)} ms features
            </dd>
            <dt>Clip</dt>
            <dd className="num">
              {(r.clip_ms / 1000).toFixed(2)} s at {r.clip_rms_dbfs?.toFixed(0)} dBFS
            </dd>
          </dl>
          {r.members && r.members.length > 0 && (
            <div className="compare">
              <h3>Double-Check members</h3>
              <table>
                <tbody>
                  {r.members.map((c) => (
                    <tr key={c.model_id}>
                      <td>
                        {modelName(state.display_names, c.model_id)}
                      </td>
                      <td>{prettyKey(c.class_key)}</td>
                      <td className="num">{(c.confidence * 100).toFixed(0)}%</td>
                      <td className="num">{c.inference_ms.toFixed(1)} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {r.compare.length > 0 && (
            <div className="compare">
              <h3>Challengers on the same clip</h3>
              <table>
                <tbody>
                  {[p, ...r.compare].map((c) => (
                    <tr key={c.model_id} className={c.model_id === p.model_id ? "is-active" : ""}>
                      <td>
                        {modelName(state.display_names, c.model_id)}
                      </td>
                      <td>{prettyKey(c.class_key)}</td>
                      <td className="num">{(c.confidence * 100).toFixed(0)}%</td>
                      <td className="num">{c.inference_ms.toFixed(1)} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {r.source !== "simulated" && (
            <div className="labeler">
              <span>What was said?</span>
              <button className="btn small" onClick={() => setLabel(p.class_key)} aria-pressed={expected === p.class_key}>
                Correct
              </button>
              <select value={expected ?? ""} onChange={(e) => setLabel(e.target.value || null)}>
                <option value="">Pick the true class</option>
                {labels.map((l) => (
                  <option key={l} value={l}>
                    {prettyKey(l)}
                  </option>
                ))}
              </select>
            </div>
          )}
        </>
      )}
    </section>
  );
}
