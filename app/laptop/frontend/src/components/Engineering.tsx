import { useRef, useState } from "react";
import { datasetName, modelName, post, prettyIntent, prettyKey } from "../api";
import type { AppState, ModelSummary, ScoreEntry } from "../types";

const kb = (b: number | null) => (b == null ? "—" : b < 1e6 ? `${(b / 1024).toFixed(0)} KB` : `${(b / 2 ** 20).toFixed(1)} MB`);

function statusText(m: ModelSummary) {
  if (m.status === "ready") return "Loaded";
  if (m.status === "not_loaded") return "Not loaded";
  return m.detail ?? m.status;
}

const SCORE_LABELS: Record<string, string> = {
  train_acc: "train acc",
  intent_acc: "intent acc",
  command_correct: "command correct",
  command_correct_tau090: "command correct",
  command_wrong: "command wrong",
  command_wrong_mean5: "command wrong (mean)",
  far: "false accept",
  far_mean5: "false accept (mean)",
  tuning_session_correct: "tuning session correct",
  final_test_session_correct: "final test session correct",
  command_macro_f1: "command macro-F1",
};

// headline metric preference; F1 and rates are shown as decimals, others as %.
const HEADLINE_ORDER = ["command_correct", "tuning_session_correct", "command_macro_f1",
                        "intent_acc", "train_acc", "command_wrong", "far"];

function fmtScore(key: string, v: number): string {
  if (key === "command_macro_f1") return v.toFixed(4);
  if (key.includes("wrong") || key === "far") return `${(v * 100).toFixed(2)}%`;
  return `${(v * 100).toFixed(1)}%`;
}

function headline(entry: ScoreEntry): [string, number] | null {
  if (entry.status !== "reported" || !entry.metrics) return null;
  for (const k of HEADLINE_ORDER) {
    if (k in entry.metrics) return [k, entry.metrics[k]];
  }
  const [k, v] = Object.entries(entry.metrics)[0];
  return [k, v];
}

function ScoreCell({ entry, names }: { entry?: ScoreEntry; names: AppState["display_names"] }) {
  if (!entry || entry.status !== "reported") {
    return <span className="muted">{entry?.status.replace(/_/g, " ") ?? "—"}</span>;
  }
  const hl = headline(entry);
  const ds = entry.dataset_id ? (names?.datasets[entry.dataset_id]?.name ?? entry.dataset_id) : "";
  const details = [
    ds || entry.dataset_id,
    entry.split ? `split ${entry.split}` : null,
    ...Object.entries(entry.metrics).map(
      ([k, v]) => `${SCORE_LABELS[k] ?? k} ${fmtScore(k, v)}`,
    ),
    entry.threshold != null ? `τ ${entry.threshold}` : null,
    entry.source ? `src: ${entry.source}` : null,
    entry.note ?? null,
  ].filter(Boolean).join("\n");
  return (
    <span className="small" title={details}>
      {hl ? `${SCORE_LABELS[hl[0]] ?? hl[0]} ${fmtScore(hl[0], hl[1])}` : "—"}
      {entry.threshold != null && hl && hl[0] !== "command_macro_f1" ? ` @ τ ${entry.threshold}` : ""}
      {entry.threshold != null && hl && hl[0] === "command_macro_f1" ? ` @ τ ${entry.threshold}` : ""}
    </span>
  );
}

function WakeMeter({ score, threshold }: { score: number | null; threshold: number }) {
  const v = score ?? 0;
  return (
    <span className="wake-meter" title="Live wake-word score (peak per 100 ms)">
      <span className="conf">
        <span className="conf-fill" style={{ width: `${v * 100}%` }} data-pass={v >= threshold} />
        <span className="conf-thr" style={{ left: `${threshold * 100}%` }} />
      </span>
      <span className="num">{score == null ? "—" : v.toFixed(2)}</span>
    </span>
  );
}

export function ModelRegistry({ state, wakeScore }: { state: AppState; wakeScore?: number | null }) {
  const e = state.edge;
  if (!e) return <p className="muted">Edge offline.</p>;
  const toggleCompare = (id: string, on: boolean) => {
    const ids = on ? [...e.compare_ids, id] : e.compare_ids.filter((x) => x !== id);
    post("/api/models/compare", { model_ids: ids });
  };
  return (
    <div className="eng-block">
      <header className="eng-head">
        <h3>Models on the edge</h3>
        <button className="btn small" onClick={() => post("/api/models/rescan")}>Rescan models folder</button>
      </header>
      <div className="scroll-x">
        <table className="grid">
          <thead>
            <tr>
              <th>Model</th>
              <th>Status</th>
              <th>Params</th>
              <th>Size</th>
              <th>Classes</th>
              <th>Threshold</th>
              <th className="left">Lineage</th>
              <th>Training</th>
              <th>Validation</th>
              <th>Test</th>
              <th>Active</th>
              <th>Compare</th>
            </tr>
          </thead>
          <tbody>
            {e.models.map((m) => (
              <tr key={m.id} className={m.id === e.active_vcm ? "is-active" : ""}>
                <td>
                  <div>{modelName(state.display_names, m.id)}</div>
                  {m.sha256_12 && <div className="muted small">sha {m.sha256_12}</div>}
                </td>
                <td className={m.status === "ready" ? "" : m.status === "not_loaded" ? "muted" : "warn-text"}>{statusText(m)}</td>
                <td className="num">{m.params?.toLocaleString() ?? "—"}</td>
                <td className="num">{kb(m.size_bytes)}</td>
                <td className="num">{m.n_classes}</td>
                <td className="num">{m.min_confidence.toFixed(2)}</td>
                <td className="small left">
                  {[
                    m.lineage.training_phase,
                    m.lineage.training_data ? datasetName(state.display_names, m.lineage.training_data) : null,
                  ]
                    .filter(Boolean)
                    .join(": ")}
                </td>
                <td>
                  <ScoreCell entry={m.scores?.training} names={state.display_names} />
                </td>
                <td>
                  <ScoreCell entry={m.scores?.validation} names={state.display_names} />
                </td>
                <td>
                  <ScoreCell entry={m.scores?.test} names={state.display_names} />
                </td>
                <td>
                  <input
                    type="radio"
                    name="active-model"
                    checked={m.id === e.active_vcm}
                    disabled={m.status === "unavailable" || m.status === "invalid"}
                    onChange={() => post("/api/models/select", { model_id: m.id })}
                    aria-label={`Use ${m.id}`}
                  />
                </td>
                <td>
                  <input
                    type="checkbox"
                    checked={e.compare_ids.includes(m.id)}
                    disabled={
                      m.id === e.active_vcm ||
                      m.status === "unavailable" ||
                      m.status === "invalid" ||
                      m.engine === "agreement"
                    }
                    onChange={(ev) => toggleCompare(m.id, ev.target.checked)}
                    aria-label={`Compare ${m.id}`}
                  />
                </td>
              </tr>
            ))}
            {e.models.length === 0 && (
              <tr>
                <td colSpan={12} className="muted">
                  No model packages found. Add one under models/vcm/ with scripts/register_model.py.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="row-controls">
        <label>
          Confidence threshold
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={e.threshold_override ?? e.models.find((m) => m.id === e.active_vcm)?.min_confidence ?? 0.7}
            onChange={(ev) => post("/api/threshold", { value: Number(ev.target.value) })}
          />
          <span className="num">
            {(e.threshold_override ?? e.models.find((m) => m.id === e.active_vcm)?.min_confidence ?? 0.7).toFixed(2)}
          </span>
          {e.threshold_override != null && (
            <button className="btn link" onClick={() => post("/api/threshold", { value: null })}>
              Use model default
            </button>
          )}
          {e.models.find((m) => m.id === e.active_vcm)?.engine === "agreement" && (
            <span className="muted small">applies to both Sweep and Focus</span>
          )}
        </label>
        <label>
          Wake source
          <select value={e.active_wake} onChange={(ev) => post("/api/wake/select", { wake_id: ev.target.value })}>
            {e.wake_engines.map((w) => (
              <option key={w.id} value={w.id} disabled={w.status !== "ready"}>
                {w.display_name}
                {w.status !== "ready" ? ` (${w.detail ?? w.status})` : ""}
              </option>
            ))}
          </select>
        </label>
        {(() => {
          const w = e.wake_engines.find((x) => x.id === e.active_wake);
          if (!w || w.kind === "manual") return null;
          return (
            <label>
              Wake score
              <WakeMeter score={wakeScore ?? null} threshold={w.threshold ?? 0.5} />
              <span className="muted small">
                threshold {w.threshold}, patience {w.patience} frames, then {e.pre_capture_delay_s} s tone gap
              </span>
            </label>
          );
        })()}
      </div>
    </div>
  );
}

export function Scoreboard({ state }: { state: AppState }) {
  return (
    <div className="eng-block">
      <header className="eng-head">
        <h3>Beta-test scoreboard</h3>
        <span className="muted small">Live and replayed clips only; label clips in the Last command panel.</span>
      </header>
      <div className="scroll-x">
        <table className="grid">
          <thead>
            <tr>
              <th>Model</th>
              <th>Clips</th>
              <th>Labelled</th>
              <th>Accuracy (intent + slot)</th>
              <th>Intent accuracy</th>
              <th>Mean confidence</th>
              <th>Mean inference</th>
            </tr>
          </thead>
          <tbody>
            {state.scoreboard.map((s) => (
              <tr key={s.model_id}>
                <td>
                  {modelName(state.display_names, s.model_id)}
                </td>
                <td className="num">{s.n_clips}</td>
                <td className="num">{s.n_labelled}</td>
                <td className="num">{s.accuracy == null ? "—" : `${(s.accuracy * 100).toFixed(0)}% (${s.n_correct}/${s.n_labelled})`}</td>
                <td className="num">
                  {s.intent_accuracy == null ? "—" : `${(s.intent_accuracy * 100).toFixed(0)}% (${s.n_intent_correct}/${s.n_labelled})`}
                </td>
                <td className="num">{s.mean_conf?.toFixed(2) ?? "—"}</td>
                <td className="num">{s.mean_inference_ms != null ? `${s.mean_inference_ms.toFixed(1)} ms` : "—"}</td>
              </tr>
            ))}
            {state.scoreboard.length === 0 && (
              <tr>
                <td colSpan={7} className="muted">No live or replayed commands yet.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <p className="small">
        Export: <a href="/api/export/commands.csv">commands.csv</a>, <a href="/api/export/predictions.csv">predictions.csv</a>
      </p>
    </div>
  );
}

function clockLabel(c: AppState["edge_clock"]): string {
  if (!c) return "—";
  const sign = c.offset_ms < 0 ? "\u2212" : "+";
  const abs = Math.abs(c.offset_ms);
  const s = Math.floor(abs / 1000) % 60;
  const m = Math.floor(abs / 60000) % 60;
  const h = Math.floor(abs / 3600000);
  const parts = [h ? `${h} h` : "", m ? `${m} min` : "", `${s} s`].filter(Boolean).join(" ");
  const corr = c.ready ? " (corrected)" : " (syncing\u2026)";
  return `${sign}${parts}${corr} \u00b7 \u00b1${c.spread_ms ?? "\u2014"} ms`;
}

export function Health({ state }: { state: AppState }) {
  const h = state.edge?.health;
  const fmt = (v: number | null | undefined, unit: string, d = 0) => (v == null ? "—" : `${v.toFixed(d)} ${unit}`);
  return (
    <div className="eng-block">
      <h3>Edge health</h3>
      <dl className="facts cols">
        <dt>Device</dt>
        <dd>{h?.device ?? "—"}</dd>
        <dt>Link</dt>
        <dd>
          {state.link.mode}, {state.link.url}
        </dd>
        <dt>Round trip</dt>
        <dd className="num">{fmt(state.link.rtt_ms, "ms", 1)}</dd>
        <dt>Clock offset</dt>
        <dd>{clockLabel(state.edge_clock)}</dd>
        <dt>CPU</dt>
        <dd className="num">{fmt(h?.cpu_percent, "%")}</dd>
        <dt>Process memory</dt>
        <dd className="num">{fmt(h?.process_rss_mb, "MB")}</dd>
        <dt>System memory used</dt>
        <dd className="num">{fmt(h?.system_mem_used_mb, "MB")}</dd>
        <dt>Mic level</dt>
        <dd className="num">{fmt(h?.level_dbfs, "dBFS")}</dd>
        <dt>Uptime</dt>
        <dd className="num">{fmt(h?.uptime_s, "s")}</dd>
        <dt>Endpointing</dt>
        <dd className="small">{state.edge?.endpointing ?? "—"}</dd>
        <dt>Last error</dt>
        <dd className="small">{h?.last_error ?? "None"}</dd>
      </dl>
    </div>
  );
}

export function CommandLog({ state }: { state: AppState }) {
  return (
    <div className="eng-block">
      <h3>Recent commands</h3>
      <div className="scroll-x">
        <table className="grid">
          <thead>
            <tr>
              <th>Time</th>
              <th>Source</th>
              <th className="left">Model</th>
              <th className="left">Prediction</th>
              <th>Conf.</th>
              <th>Inference</th>
              <th>Result to action</th>
              <th className="left">Action</th>
              <th className="left">True class</th>
            </tr>
          </thead>
          <tbody>
            {state.commands.map((c) => (
              <tr key={c.command_id}>
                <td className="num">{new Date(c.ts_ms).toLocaleTimeString([], { hour12: false })}</td>
                <td>{c.source}</td>
                <td className="small left">
                  {modelName(state.display_names, c.model_id)}
                </td>
                <td className="left">{prettyKey(c.class_key)}</td>
                <td className="num">{(c.confidence * 100).toFixed(0)}%</td>
                <td className="num">{c.inference_ms.toFixed(1)} ms</td>
                <td className="num">{c.result_to_action_ms.toFixed(0)} ms</td>
                <td className={`${c.executed ? "" : "muted"} left`}>{c.executed ? c.action : c.reject_reason ?? c.action}</td>
                <td className={`${c.expected_class ? (c.expected_class === c.class_key ? "" : "warn-text") : "muted"} left`}>
                  {c.expected_class ? prettyKey(c.expected_class) : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function DevTools({ state }: { state: AppState }) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [expected, setExpected] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const labels = state.ontology.leaf_labels;
  const groups = labels.reduce<Record<string, string[]>>((acc, k) => {
    const intent = k.split("|")[0];
    (acc[intent] ??= []).push(k);
    return acc;
  }, {});

  const replay = async () => {
    const files = fileRef.current?.files;
    if (!files || files.length === 0) return;
    const fd = new FormData();
    Array.from(files).forEach((f) => fd.append("files", f));
    if (expected) fd.append("expected_class", expected);
    const r = await fetch("/api/replay", { method: "POST", body: fd });
    setMsg(r.ok ? `Queued ${files.length} clip(s)` : `Replay failed: ${(await r.json()).detail}`);
  };

  return (
    <div className="eng-block">
      <h3>Test inputs</h3>
      <div className="replay">
        <label>
          Replay WAV files through the active model
          <input ref={fileRef} type="file" accept=".wav,audio/wav" multiple />
        </label>
        <label>
          True class for this batch
          <select value={expected} onChange={(e) => setExpected(e.target.value)}>
            <option value="">Unknown (label later)</option>
            {labels.map((l) => (
              <option key={l} value={l}>
                {prettyKey(l)}
              </option>
            ))}
          </select>
        </label>
        <button className="btn" onClick={replay}>Replay</button>
        {msg && <span className="muted small">{msg}</span>}
      </div>
      <p className="muted small">Simulate skips the model and marks the result as simulated. Use it to test the app, not the VCM.</p>
      <div className="sim-grid">
        {Object.entries(groups).map(([intent, keys]) => (
          <div key={intent} className="sim-group">
            <span className="sim-intent">{prettyIntent(intent)}</span>
            {keys.map((k) => (
              <button key={k} className="btn tiny" onClick={() => post("/api/simulate", { class_key: k })}>
                {k.includes("|") ? k.split("|")[1] : "Run"}
              </button>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
