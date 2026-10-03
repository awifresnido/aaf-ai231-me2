import { useCallback, useEffect, useState } from "react";
import { del, modelName, post, prettyKey } from "../api";
import type { AppState, BenchSummary, BenchSessionInfo } from "../types";

const OUTCOME_FLASH: Record<string, { icon: string; text: string }> = {
  correct: { icon: "✓", text: "Correct" },
  not_understood: { icon: "↺", text: "Not understood (safe)" },
  wrong: { icon: "✗", text: "Wrong" },
  false_accept: { icon: "✗", text: "Acted on it" },
  correct_reject: { icon: "✓", text: "Ignored" },
};

function useBenchmarkData(open: boolean, running: boolean) {
  const [summary, setSummary] = useState<BenchSummary | null>(null);
  const [sessions, setSessions] = useState<BenchSessionInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    try {
      const [sr, sr2] = await Promise.all([fetch("/api/bench/summary"), fetch("/api/bench/sessions")]);
      if (!sr.ok) throw new Error(`summary ${sr.status}`);
      setSummary(await sr.json());
      if (sr2.ok) setSessions(await sr2.json());
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    if (!open || running) return;
    reload();
    const id = window.setInterval(reload, 5000);
    return () => window.clearInterval(id);
  }, [open, running, reload]);

  return { summary, sessions, error, reload };
}

const pct1 = (v: number | null) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
const ms = (v: number | null) => (v == null ? "—" : `${v.toFixed(1)} ms`);

function Chip({ c }: { c: BenchSummary["chips"][number] }) {
  const tone = c.status === "pass" ? "ok" : c.status === "fail" ? "alarm" : "warn";
  const isMs = c.key === "pi_model_ms" || c.key === "pi_features_ms";
  const shown = c.value == null ? "—" : c.key === "params" ? c.value.toLocaleString()
    : c.key === "size_kb" ? `${c.value} KB` : isMs ? ms(c.value)
    : pct1(c.value);
  const ci = c.lo != null && c.hi != null ? ` [${pct1(c.lo)}–${pct1(c.hi)}]` : "";
  return (
    <div className={`bench-chip ${tone}`}>
      <div className="bench-chip-label">{c.label}</div>
      <div className="bench-chip-value">
        {shown}{ci} <span className="muted small">(n={c.n == null ? "—" : c.n})</span>
      </div>
      <div className="bench-chip-status muted small">
        {c.status === "too_few" ? "too few trials" : c.status === "pass" ? "pass" : "fail"}
        {" · target "}{c.key === "params" ? c.target.toLocaleString() : c.key === "size_kb" ? `${c.target} KB` : isMs ? ms(c.target) : pct1(c.target)}
      </div>
    </div>
  );
}

export function Benchmark({ open, presentation, state, onClose }: {
  open: boolean; presentation: boolean; state: AppState; onClose: () => void;
}) {
  const bench = state.benchmark;
  const { summary, sessions, error, reload } = useBenchmarkData(open, bench.running);
  const [setType, setSetType] = useState("fixed");
  const [alias, setAlias] = useState("");
  const [noise, setNoise] = useState("quiet");
  const [distance, setDistance] = useState("near");
  const [consent, setConsent] = useState(false);
  const [note, setNote] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [tab, setTab] = useState<"setup" | "results">("setup");

  const active = state.edge?.models.find((m) => m.id === state.edge?.active_vcm);
  const tau = state.edge?.threshold_override ?? active?.min_confidence ?? 0.8;
  const modelLabel = modelName(state.display_names, state.edge?.active_vcm ?? "");
  const wake = state.edge?.wake_engines.find((w) => w.id === state.edge?.active_wake);
  const wakePhrase = wake?.phrase ?? "Hey Rhasspy";

  useEffect(() => {
    if (open && sessions.length) {
      const n = sessions.length + 1;
      setAlias((a) => a || `Guest ${String(n).padStart(2, "0")}`);
    }
  }, [open, sessions.length]);

  const create = async () => {
    setMsg(null);
    const r = await post("/api/bench/sessions", {
      set_type: setType, cond_noise: noise, cond_distance: distance,
      consent_results: consent, consent_audio: false,
      alias: alias.trim() || undefined, note: note.trim() || undefined,
    });
    if (!r.ok) { setMsg((await r.json()).detail ?? "create failed"); return; }
    const sess = await r.json();
    const s2 = await post(`/api/bench/sessions/${sess.session_id}/start`);
    if (!s2.ok) setMsg((await s2.json()).detail ?? "start failed");
  };

  const act = async (path: string) => {
    setMsg(null);
    const r = await post(path);
    if (!r.ok) setMsg((await r.json()).detail ?? "action failed");
  };

  const remove = async (sid: string) => {
    if (!window.confirm(`Delete ${sid} and its trials?`)) return;
    await del(`/api/bench/sessions/${sid}`);
    reload();
  };

  if (!open) return null;

  const counts = bench.counts ?? {};

  return (
    <main className="benchmark" aria-label="Live Benchmark">
      {bench.running ? (
        <LiveRun bench={bench} wakePhrase={wakePhrase} counts={counts} presentation={presentation}
          onRetry={() => act("/api/bench/trial/retry")} onMisspoken={() => act("/api/bench/trial/misspoken")}
          onSkip={() => act("/api/bench/trial/skip")} onAbandon={() => act(`/api/bench/sessions/${bench.session_id}/abandon`)} />
      ) : (
        <div className="bench-panels">
          <div className="bench-tabs">
            <button className={`btn small ${tab === "setup" ? "" : ""}`} aria-pressed={tab === "setup"} onClick={() => setTab("setup")}>New participant</button>
            <button className="btn small" aria-pressed={tab === "results"} onClick={() => setTab("results")}>Results</button>
          </div>

          {tab === "setup" ? (
            <SetupForm setType={setType} setSetType={setSetType} alias={alias} setAlias={setAlias}
              noise={noise} setNoise={setNoise} distance={distance} setDistance={setDistance}
              consent={consent} setConsent={setConsent} note={note} setNote={setNote}
              modelLabel={modelLabel} tau={tau} msg={msg} onCreate={create} />
          ) : (
            <Results summary={summary} sessions={sessions} error={error} names={state.display_names}
              onDelete={remove} />
          )}
        </div>
      )}

      <footer className="about-foot muted small">
        Title: "How well the VCM understood each voice". Nothing identifying is collected — aliases only, no audio saved.
      </footer>
    </main>
  );
}

function SetupForm({ setType, setSetType, alias, setAlias, noise, setNoise, distance, setDistance,
  consent, setConsent, note, setNote, modelLabel, tau, msg, onCreate }: {
  setType: string; setSetType: (v: string) => void; alias: string; setAlias: (v: string) => void;
  noise: string; setNoise: (v: string) => void; distance: string; setDistance: (v: string) => void;
  consent: boolean; setConsent: (v: boolean) => void; note: string; setNote: (v: string) => void;
  modelLabel: string; tau: number; msg: string | null; onCreate: () => void;
}) {
  return (
    <section className="about-panel">
      <h2 className="about-title">How well the VCM understood each voice</h2>
      <p className="about-body">
        A visitor reads each prompt aloud, and the VCM answers on the Raspberry Pi. Results are compared
        across speakers, never ranked. Devices still act on recognised commands — that is part of the demo.
      </p>
      <div className="about-callout">
        <strong>Model:</strong> {modelLabel} · τ {tau.toFixed(2)}. Change the model on the dashboard before starting.
      </div>
      <form className="bench-form" onSubmit={(e) => { e.preventDefault(); onCreate(); }}>
        <label>
          <span>Alias <span className="muted small">(no real names)</span></span>
          <input value={alias} onChange={(e) => setAlias(e.target.value)} placeholder="Guest 01" />
        </label>
        <label>
          <span>Command set</span>
          <select value={setType} onChange={(e) => setSetType(e.target.value)}>
            <option value="fixed">Same set for everyone</option>
            <option value="random">Random set</option>
          </select>
        </label>
        <label>
          <span>Condition</span>
          <select value={noise} onChange={(e) => setNoise(e.target.value)}>
            <option value="quiet">quiet</option><option value="music">music playing</option>
            <option value="noise">noise</option><option value="other">other</option>
          </select>
        </label>
        <label>
          <span>Distance</span>
          <select value={distance} onChange={(e) => setDistance(e.target.value)}>
            <option value="near">near</option><option value="far">far</option>
          </select>
        </label>
        <label className="bench-consent">
          <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
          <span>My results are saved under this alias.</span>
        </label>
        <label>
          <input type="checkbox" disabled checked={false} />
          <span className="muted">audio saving not enabled in this version</span>
        </label>
        <label>
          <span>Note (optional)</span>
          <input value={note} onChange={(e) => setNote(e.target.value)} />
        </label>
        <button className="btn primary" disabled={!consent} type="submit">Start</button>
      </form>
      {msg && <p className="warn-text">{msg}</p>}
    </section>
  );
}

function LiveRun({ bench, wakePhrase, counts, presentation, onRetry, onMisspoken, onSkip, onAbandon }: {
  bench: AppState["benchmark"]; wakePhrase: string; counts: Record<string, number>; presentation: boolean;
  onRetry: () => void; onMisspoken: () => void; onSkip: () => void; onAbandon: () => void;
}) {
  const p = bench.prompt;
  const phase = bench.phase;
  const lo = bench.last_outcome;
  const step = phase === "await_wake" ? `Say “${wakePhrase}”`
    : phase === "await_result" && p
      ? (p.kind === "negative" ? `Now say something that is not a command: “${p.say}”` : `Now say: “${p.say}”`)
      : phase === "scored" ? "…" : "…";
  return (
    <section className="bench-live">
      <div className={`bench-prompt ${presentation ? "present" : ""}`}>
        <div className="muted small">Prompt {bench.trial_idx ?? 0} of {bench.trial_total ?? 0}</div>
        <div className="bench-say">{step}</div>
        {phase === "scored" && lo && (
          <div className={`bench-flash ${lo.outcome === "correct" || lo.outcome === "correct_reject" ? "ok" : lo.outcome === "not_understood" ? "warn" : "alarm"}`}>
            {OUTCOME_FLASH[lo.outcome]?.icon ?? "•"} {OUTCOME_FLASH[lo.outcome]?.text ?? lo.outcome}
            {lo.predicted && lo.outcome === "wrong" ? ` (it heard ${prettyKey(lo.predicted)})` : ""}
            <div className="muted small">confidence {(lo.confidence * 100).toFixed(0)}% · Pi {lo.inference_ms.toFixed(1)} ms</div>
          </div>
        )}
      </div>
      <div className="bench-counts muted small">
        ✓ {counts.correct ?? 0} correct · ↺ {counts.not_understood ?? 0} · ✗ {counts.wrong ?? 0} ·
        ignored {counts.correct_reject ?? 0} · acted {counts.false_accept ?? 0} · wake miss {counts.wake_miss ?? 0}
      </div>
      <div className="bench-ops">
        <button className="btn small" onClick={onRetry}>Retry</button>
        <button className="btn small" onClick={onMisspoken}>Mis-spoken</button>
        <button className="btn small" onClick={onSkip}>Skip</button>
        <button className="btn small danger" onClick={onAbandon}>Abandon</button>
      </div>
    </section>
  );
}

function Results({ summary, sessions, error, names, onDelete }: {
  summary: BenchSummary | null; sessions: BenchSessionInfo[]; error: string | null;
  names: AppState["display_names"]; onDelete: (sid: string) => void;
}) {
  if (error) return <p className="warn-text">Could not load results: {error}</p>;
  if (!summary) return <p className="muted">Loading results…</p>;
  const p = summary.pooled;
  return (
    <section className="about-panel">
      <h2 className="about-title">Results</h2>
      <div className="bench-chips">
        {summary.chips.map((c) => <Chip key={c.key} c={c} />)}
      </div>
      <p className="muted small">
        All speakers: {p.correct.value == null ? "—" : pct1(p.correct.value)} correct, {pct1(p.wrong.value)} wrong,
        {pct1(p.false_accept.value)} false accept, {pct1(p.wake_miss.value)} wake miss · Pi model p95 {ms(p.pi_model_ms.p95)} ·
        <a href="/api/bench/export.csv">CSV</a> · <a href="/api/bench/export.json">JSON</a>
      </p>
      <table className="grid about-table">
        <thead>
          <tr>
            <th>Alias</th><th>Set</th><th>Model</th><th>Condition</th><th>Distance</th>
            <th>Scored</th><th>Correct</th><th>Wrong</th><th>Neg. ignored</th>
            <th>Wake miss</th><th>Pi model med.</th><th>E2E med.</th><th></th>
          </tr>
        </thead>
        <tbody>
          {summary.sessions.map((s) => (
            <tr key={s.session_id}>
              <td>{s.alias}</td>
              <td>{s.set_type}</td>
              <td>{modelName(names, s.model_id)}</td>
              <td>{s.cond_noise}</td>
              <td>{s.cond_distance}</td>
              <td className="num">{s.scored}</td>
              <td className="num">{pct1(s.correct_rate)}</td>
              <td className="num">{pct1(s.wrong_rate)}</td>
              <td className="num">{s.neg_ignored}/{s.neg_total}</td>
              <td className="num">{s.wake_miss}</td>
              <td className="num">{ms(s.model_median_ms)}</td>
              <td className="num">{ms(s.e2e_median_ms)}</td>
              <td><button className="btn link" onClick={() => onDelete(s.session_id)}>Delete</button></td>
            </tr>
          ))}
        </tbody>
      </table>
      <h3 className="muted">Participants × prompts</h3>
      <div className="scroll-x">
        <table className="grid bench-heatmap">
          <thead><tr><th></th>{summary.heatmap.columns.map((c) => <th key={c} className="left">{c}</th>)}</tr></thead>
          <tbody>
            {summary.heatmap.rows.map((r) => (
              <tr key={r.session_id}>
                <td>{r.alias}</td>
                {r.cells.map((cell) => (
                  <td key={cell.col} className={`bench-cell ${cell.color}`} title={cell.col}>{cell.color === "green" ? "✓" : cell.color === "amber" ? "~" : cell.color === "red" ? "✗" : ""}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted small">green = correct / ignored · amber = not understood or intent right · red = wrong / acted · grey = excluded.</p>
    </section>
  );
}
