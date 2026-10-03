import { useEffect, useState } from "react";

// ---------------------------------------------------------------------------
// The About view. Every number comes from /api/about (about_data.json); nothing
// is typed into React. Figures are plain inline SVG, no chart library.
// ---------------------------------------------------------------------------

export type AboutData = {
  generated_at: string;
  app_commit: string;
  tvcm_commit: string;
  host: { composition: string };
  constraints: {
    n_classes: number;
    n_intents: number;
    n_fixed: number;
    n_slotted: number;
    params_default_k: number;
    onnx_kb_default: number;
    window_s: number;
    n_frames: number;
  };
  intents: string[];
  audit: { relabeled: number; dropped: number };
  story: { first_acc: string; fixed_acc: string };
  composition: {
    total_clips: number;
    datasets: Record<string, { name: string; train: number; validation: number; test: number }>;
  };
  headline: Record<
    string,
    { id: string; name: string; correct: number | null; correct_sd: number | null;
      wrong: number; wrong_sd: number | null; random_far: number | null; random_far_sd: number | null }
  >;
  table: Record<
    string,
    { id: string; name: string; correct: number | null; wrong: number; far: number;
      cmd_f1_val: number | null; cmd_f1_test: number | null; final_test: number | null }
  >;
  params: Record<string, number>;
  b0: { correct: number; wrong: number };
  mcnemar: { n: number; e1_b2_p: string };
  final_test: Record<string, number>;
  tuning_sd: { B2: number };
  agree: { wrong: number; far: number };
  parity: { B2_s0: number };
  deployment: Record<string, { name: string; onnx_kb: number; runtime: string; quantization: string }>;
  latency_pi: { status: string };
  repo_url: string | null;
};

const MODEL_ORDER = ["B2_s0", "E1_s0", "AGREE_B2_E1", "G2_s0"];
const DATASET_ORDER = ["synthetic_tts", "fsc", "gsc_v2", "gsc_noise", "personal_awi"];
const SPLIT_COLORS = ["var(--listen)", "var(--think)", "var(--warn)"]; // train / validation / test
const METRIC_COLORS = ["var(--ok)", "var(--alarm)", "var(--warn)"];    // correct / wrong / random-FAR

const pct = (x: number | null, d = 1) => (x == null ? "—" : `${(x * 100).toFixed(d)}%`);
const num = (x: number | null) => (x == null ? "—" : x.toLocaleString());

const useAboutData = (open: boolean) => {
  const [data, setData] = useState<AboutData | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!open) return;
    let stop = false;
    fetch("/api/about")
      .then((r) => {
        if (!r.ok) throw new Error(`/api/about -> ${r.status}`);
        return r.json();
      })
      .then((d: AboutData) => { if (!stop) { setData(d); setError(null); } })
      .catch((e) => { if (!stop) setError(String(e)); });
    return () => { stop = true; };
  }, [open]);
  return { data, error };
};

// ---- GitHub links ---------------------------------------------------------
const PANEL_LINKS: [string, string[]][] = [
  ["The problem and the data", ["docs/data.md", "docs/data.md#label-audit"]],
  ["Models and the training pipeline", ["docs/architectures.md", "docs/training.md", "docs/mlops.md"]],
  ["Results, model choice and edge deployment",
   ["docs/metrics.md", "docs/results.md", "docs/model-selection.md", "docs/deployment.md", "docs/limitations.md"]],
];

function GitHubLinks({ repoUrl, panel }: { repoUrl: string | null; panel: number }) {
  const links = PANEL_LINKS[panel][1];
  if (!repoUrl) {
    return <div className="about-github muted">Details on GitHub (coming soon)</div>;
  }
  return (
    <div className="about-github">
      {links.map((a) => {
        const label = a.split("#")[0].replace("docs/", "").replace(".md", "");
        return (
          <a key={a} href={`${repoUrl}/${a}`} target="_blank" rel="noreferrer">
            {label} →
          </a>
        );
      })}
    </div>
  );
}

// ---- figures --------------------------------------------------------------
function CompositionFigure({ d }: { d: AboutData }) {
  const rows = DATASET_ORDER.map((id) => d.composition.datasets[id]).filter(Boolean);
  const max = Math.max(...rows.map((r) => r.train + r.validation + r.test));
  const W = 620, ROW = 30, LBL = 150, H = rows.length * ROW + 40;
  return (
    <figure className="about-fig chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Blend v1 composition by split">
        {rows.map((r, i) => {
          const total = r.train + r.validation + r.test;
          const y = 30 + i * ROW;
          let x = LBL;
          const segs = [
            { v: r.train, c: SPLIT_COLORS[0] },
            { v: r.validation, c: SPLIT_COLORS[1] },
            { v: r.test, c: SPLIT_COLORS[2] },
          ];
          return (
            <g key={r.name}>
              <text x={LBL - 8} y={y + 14} textAnchor="end" fontSize="13" fill="currentColor">{r.name}</text>
              {segs.map((s, j) => {
                const w = (s.v / max) * (W - LBL - 70);
                const rect = (
                  <rect key={j} x={x} y={y} width={w} height={18} fill={s.c}
                        opacity={0.9} rx={j === segs.length - 1 ? 3 : 0} />
                );
                x += w;
                return rect;
              })}
              <text x={x + 6} y={y + 14} fontSize="12" fill="currentColor" opacity="0.7">
                {num(total)}
              </text>
            </g>
          );
        })}
        <g fontSize="11" fill="currentColor" opacity="0.7">
          <rect x={LBL} y={H - 14} width={10} height={10} fill={SPLIT_COLORS[0]} />
          <text x={LBL + 14} y={H - 5}>train</text>
          <rect x={LBL + 56} y={H - 14} width={10} height={10} fill={SPLIT_COLORS[1]} />
          <text x={LBL + 70} y={H - 5}>validation</text>
          <rect x={LBL + 148} y={H - 14} width={10} height={10} fill={SPLIT_COLORS[2]} />
          <text x={LBL + 162} y={H - 5}>test</text>
        </g>
      </svg>
      <figcaption className="muted small">
        Blend v1: {num(d.composition.total_clips)} clips across {rows.length} datasets, split by speaker
        (or by recording session for the Demo Speaker).
      </figcaption>
    </figure>
  );
}

// Diagrams are HTML/CSS, not SVG: text then wraps inside its box at the page's
// normal font size. (The old SVG versions used a ~650-unit viewBox stretched to
// full width, which scaled all text ~2.7x and let long labels spill over the
// neighbouring boxes.)
function ArchitectureFigure({ d }: { d: AboutData }) {
  const models = [
    { id: "B2_s0", name: "Sweep", tag: "default", based: "TC-ResNet (temporal convolution)",
      idea: "Scans the whole clip for sound patterns." },
    { id: "E1_s0", name: "Focus", tag: "alternative", based: "CRNN + attention",
      idea: "Reads the clip in order and focuses on the deciding word." },
    { id: "AGREE_B2_E1", name: "Double-Check", tag: "safety mode", based: "Sweep + Focus agreement",
      idea: "Acts only when both models agree and are confident." },
    { id: "G2_s0", name: "Benchmark", tag: "reference", based: "Hello-Edge DS-CNN",
      idea: "Field-standard single-keyword model, same size." },
  ];
  return (
    <figure className="about-fig arch" aria-label="Model architectures">
      <div className="arch-input">
        <strong>Input for every model:</strong> mel spectrogram, 40 bands × {d.constraints.n_frames} frames
        ({d.constraints.window_s} s of 16 kHz audio)
      </div>
      <div className="arch-grid">
        {models.map((m) => (
          <div key={m.id} className={`arch-card tag-${m.tag.replace(" ", "-")}`}>
            <div className="arch-head">
              <span className="arch-name">{m.name}</span>
              <span className="arch-tag">{m.tag}</span>
            </div>
            <div className="arch-based">{m.based}</div>
            <div className="arch-idea">{m.idea}</div>
            <div className="arch-params">
              {m.id === "AGREE_B2_E1" ? "no weights of its own" : `${num(d.params[m.id])} parameters`}
            </div>
          </div>
        ))}
      </div>
    </figure>
  );
}

type FlowStep = { label: string; side?: "laptop" | "pi" | "net" };

function FlowFigure({ steps, label, legend }: {
  steps: FlowStep[]; label: string; legend?: [FlowStep["side"], string][];
}) {
  return (
    <figure className="about-fig flow" aria-label={label}>
      <ol className="flow-steps">
        {steps.map((s, i) => (
          <li key={s.label} className={`flow-step ${s.side ?? ""}`}>
            <span className="flow-n">{i + 1}</span>
            <span>{s.label}</span>
          </li>
        ))}
      </ol>
      {legend && (
        <div className="flow-legend">
          {legend.map(([side, text]) => (
            <span key={text}><i className={`flow-key ${side}`} />{text}</span>
          ))}
        </div>
      )}
    </figure>
  );
}

function PipelineFigure() {
  const steps: FlowStep[] = [
    "data adapters", "versioned manifests", "label audit", "training on DGX (8 GPUs)", "evaluation",
    "pre-registered decision rule", "ONNX export + parity check", "app model registry", "Raspberry Pi",
  ].map((label) => ({ label }));
  return <FlowFigure steps={steps} label="MLOps pipeline" />;
}

// log10 mapping from rate (0.0001..1) to chart height
function HeadlineFigure({ d }: { d: AboutData }) {
  const LO = -3, HI = 0; // 0.001 .. 1.0
  // top margin 36 leaves room for the data label above a 100 % bar
  const W = 640, PLOT_X = 40, PLOT_W = W - PLOT_X - 10, PLOT_H = 220, BASE = PLOT_H + 36;
  const y = (v: number) => BASE - ((Math.log10(Math.max(v, 1e-4)) - LO) / (HI - LO)) * PLOT_H;
  const groupW = PLOT_W / MODEL_ORDER.length;
  const barW = groupW / 3.6;
  const metrics = [
    { key: "correct", sd: "correct_sd", label: "correct", c: METRIC_COLORS[0] },
    { key: "wrong", sd: "wrong_sd", label: "wrong action", c: METRIC_COLORS[1] },
    { key: "random_far", sd: "random_far_sd", label: "random-word false accept", c: METRIC_COLORS[2] },
  ] as const;
  const grid = [1, 0.1, 0.01, 0.001];
  return (
    <figure className="about-fig chart">
      <svg viewBox={`0 0 ${W} ${BASE + 44}`} role="img" aria-label="Headline results">
        {grid.map((v) => (
          <g key={v} fontSize="10" fill="currentColor" opacity="0.6">
            <line x1={PLOT_X} y1={y(v)} x2={PLOT_X + PLOT_W} y2={y(v)} stroke="var(--rule)" />
            <text x={PLOT_X - 6} y={y(v) + 3} textAnchor="end">{v * 100}%</text>
          </g>
        ))}
        {MODEL_ORDER.map((mid, gi) => {
          const row = d.headline[mid];
          const gx = PLOT_X + gi * groupW;
          return (
            <g key={mid}>
              {metrics.map((m, mi) => {
                const v = row[m.key];
                const bx = gx + mi * barW + 6;
                const cx = bx + (barW - 6) / 2;
                if (v == null) {
                  // make "no value" explicit instead of leaving a silent gap
                  return (
                    <text key={m.key} x={cx} y={BASE - 6} fontSize="9" textAnchor="middle"
                          fill="currentColor" opacity="0.55">n/a</text>
                  );
                }
                const top = y(v);
                const sd = row[m.sd];
                const barTop = sd != null ? Math.min(top, y(Math.min(v + sd, 1))) : top;
                // data label: same formatting as the table below the chart
                const label = m.key === "correct" ? pct(v, 1) : pct(v, 2);
                // tall bars (correct ≈ 80–100 %) reach the 100 % gridline; place the
                // label inside the bar so the top gridline doesn't cut through it
                const labelY = top < 60 ? top + 13 : barTop - 5;
                return (
                  <g key={m.key}>
                    <rect x={bx} y={top} width={barW - 6} height={BASE - top} fill={m.c} opacity={0.85} rx={2} />
                    {sd != null && (
                      <line x1={cx} y1={y(Math.max(v - sd, 1e-4))} x2={cx} y2={y(Math.min(v + sd, 1))}
                            stroke="currentColor" strokeWidth="1.5" />
                    )}
                    <text x={cx} y={labelY} fontSize="10.5" fontWeight="600" textAnchor="middle"
                          fill="currentColor">{label}</text>
                  </g>
                );
              })}
              <text x={gx + groupW / 2} y={BASE + 16} fontSize="12" fontWeight="600" textAnchor="middle"
                    fill="currentColor">{row.name}</text>
            </g>
          );
        })}
        <g fontSize="11" fill="currentColor" opacity="0.8" transform={`translate(${PLOT_X}, ${BASE + 36})`}>
          {metrics.map((m, i) => (
            <g key={m.key} transform={`translate(${i * 180}, 0)`}>
              <rect x={0} y={-8} width={10} height={10} fill={m.c} />
              <text x={14} y={1}>{m.label}</text>
            </g>
          ))}
        </g>
      </svg>
      <figcaption className="muted small">
        Real-voice validation, log scale (wrong actions and false accepts are tiny by design). Labels
        match the table below. Error bars are ±1 seed sd where seeds exist; Double-Check has no seeds,
        and Sweep/Focus “correct” is a single seed. “n/a” = not measured for that model.
      </figcaption>
    </figure>
  );
}

function EdgeFigure() {
  const steps: FlowStep[] = [
    { label: "laptop microphone", side: "laptop" },
    { label: "16 kHz audio stream over Ethernet", side: "net" },
    { label: "wake word", side: "pi" },
    { label: "4-s command capture", side: "pi" },
    { label: "TinyVCM model", side: "pi" },
    { label: "confidence gate (τ)", side: "pi" },
    { label: "intent + slot sent back", side: "net" },
    { label: "laptop app", side: "laptop" },
    { label: "simulated device acts", side: "laptop" },
  ];
  return (
    <FlowFigure
      steps={steps}
      label="Edge deployment"
      legend={[["laptop", "laptop"], ["pi", "Raspberry Pi 4 (all AI runs here)"], ["net", "Ethernet link"]]}
    />
  );
}

// ---- panels ---------------------------------------------------------------
function Panel1({ d }: { d: AboutData }) {
  const c = d.constraints;
  const dataRows: [string, string, string][] = [
    ["Synthetic Voices", "TTS in many cloned voices (classmate-contributed)", "every command, in many voices"],
    ["Home Commands", "Fluent Speech Commands (real speakers)", "real voices for lights, volume, music; realistic non-commands"],
    ["Single Words", "Google Speech Commands v2", "what is not a command; “stop”"],
    ["Background Noise", "Google Speech Commands noise", "silence, and noise for augmentation"],
    ["Demo Speaker", "own recordings, 3 sessions", "the voice the demo is tuned for"],
    ["Blend v1", "all of the above, audited", "the training set of the final models"],
  ];
  return (
    <section className="about-panel">
      <h2 className="about-title">A voice-command brain small enough for a Raspberry Pi</h2>
      <p className="about-lead">
        TinyVCM maps a spoken command straight to one of {c.n_classes} command classes: no speech-to-text,
        no language model, no cloud. Everything runs on a Raspberry Pi 4.
      </p>
      <div className="about-chips">
        <span className="chip">~{c.params_default_k}k parameters</span>
        <span className="chip">{c.onnx_kb_default} KB model</span>
        <span className="chip">16 kHz audio · {c.window_s}-s window</span>
        <span className="chip">on-device only</span>
        <span className="chip">no transcription</span>
      </div>
      <p className="about-body">
        {c.n_intents} commands ({c.n_fixed} fixed + {c.n_slotted} with values such as brightness or timer
        length) → {c.n_classes} classes, including <em>not a command</em> and <em>silence</em>.
      </p>
      <div className="about-intents">
        {d.intents.map((i) => <span key={i} className="chip intent">{i}</span>)}
      </div>
      <table className="grid about-table">
        <thead><tr><th>Dataset</th><th className="left">Source</th><th className="left">What it teaches</th></tr></thead>
        <tbody>
          {dataRows.map(([a, b, cc]) => (
            <tr key={a}><td>{a}</td><td className="muted left">{b}</td><td className="left">{cc}</td></tr>
          ))}
        </tbody>
      </table>
      <CompositionFigure d={d} />
      <div className="about-callout">
        <strong>Data quality</strong> — a label audit found commands labelled as <em>not a command</em>{" "}
        (e.g. “Pause”, and every “stop” in Single Words). A rule-based audit fixed them:{" "}
        {d.audit.relabeled} relabelled, {d.audit.dropped} dropped, 0 contradictions remaining.
      </div>
      <div className="about-callout">
        <strong>Fair splits</strong> — splits are by speaker, or by recording session for the Demo Speaker
        (Practice / Tuning / Final Test), so no voice is heard in both training and testing.
      </div>
      <p className="about-credits muted small">
        Fluent Speech Commands and Google Speech Commands (CC BY 4.0), used under their licences ·
        Synthetic Voices contributed by a classmate · weather by Open-Meteo
      </p>
      <GitHubLinks repoUrl={d.repo_url} panel={0} />
    </section>
  );
}

function Panel2({ d }: { d: AboutData }) {
  return (
    <section className="about-panel">
      <h2 className="about-title">Four models, one pipeline</h2>
      <ArchitectureFigure d={d} />
      <ol className="about-steps">
        <li>Learn the commands from Synthetic Voices.</li>
        <li>Train on Blend v1, with noise, reverb, speed, gain and mic augmentation, and “not a command” capped at 3× a typical command.</li>
        <li>Run 5 training seeds per model.</li>
        <li>Keep the checkpoint that best recognises real voices.</li>
        <li>Choose the confidence threshold τ so that every seed stays at or below 1 % wrong actions.</li>
      </ol>
      <PipelineFigure />
      <div className="about-chips">
        <span className="chip">pre-registered success criteria</span>
        <span className="chip">hard gates (stop on failure)</span>
        <span className="chip">data never edited in place</span>
        <span className="chip">per-run config sidecar</span>
        <span className="chip">git branch per experiment</span>
        <span className="chip">model registry with manifests</span>
      </div>
      <p className="about-body">
        First model: <strong>{d.story.first_acc}</strong> → found a one-filter bug and a padding artefact →
        fixed model: <strong>{d.story.fixed_acc}</strong> on synthetic voices → then closed the gap to real
        voices with the audit and augmentation.
      </p>
      <GitHubLinks repoUrl={d.repo_url} panel={1} />
    </section>
  );
}

function Panel3({ d }: { d: AboutData }) {
  return (
    <section className="about-panel">
      <h2 className="about-title">What works, what we chose, and how it runs on the edge</h2>
      <dl className="about-glossary">
        <dt>Correct</dt><dd>a command recognised correctly, and confidently enough to act</dd>
        <dt>Rejected</dt><dd>unsure, so it asks you to repeat (safe)</dd>
        <dt>Wrong action</dt><dd>did the wrong thing. <strong>This is the one that matters most.</strong></dd>
        <dt>False accept</dt><dd>acted on something that wasn’t a command</dd>
        <dt>Random-word false accept</dt><dd>acted on a random single word</dd>
        <dt>Command F1</dt><dd>balance of precision and recall across all commands</dd>
        <dt>τ</dt><dd>the minimum confidence needed to act</dd>
      </dl>
      <HeadlineFigure d={d} />
      <table className="grid about-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>correct (val)</th>
            <th>wrong (val)</th>
            <th>false accept (val)</th>
            <th>command F1 (test)</th>
            <th>Final Test Session</th>
          </tr>
        </thead>
        <tbody>
          {MODEL_ORDER.map((mid) => {
            const r = d.table[mid];
            return (
              <tr key={mid}>
                <td>{r.name}</td>
                <td className="num">{pct(r.correct)}</td>
                <td className="num">{pct(r.wrong, 2)}</td>
                <td className="num">{pct(r.far, 2)}</td>
                <td className="num">{r.cmd_f1_test == null ? "—" : r.cmd_f1_test.toFixed(4)}</td>
                <td className="num">{pct(r.final_test)}</td>
              </tr>
            );
          })}
        </tbody>
        <tfoot>
          <tr>
            <td colSpan={6} className="muted small">
              Sweep Preview (synthetic only): without real voices: {pct(d.b0.correct)} correct, {pct(d.b0.wrong)} wrong.
            </td>
          </tr>
        </tfoot>
      </table>
      <div className="about-callout">
        <strong>Is the difference real?</strong> — paired test on {d.mcnemar.n} identical real-voice clips:
        Focus vs Sweep p = {d.mcnemar.e1_b2_p}; both far ahead of the benchmark (p &lt; 10⁻¹⁸⁰).
      </div>
      <div className="about-callout">
        <strong>Honest limits</strong> — the test split was evaluated once, after the models were chosen;
        12 of the 19 commands have real-voice data only from the demo speaker, so other voices are
        unverified for those.
      </div>
      <div className="about-callout">
        <strong>Model choice</strong> — the demo serves one known speaker. The default is chosen on that
        speaker’s held-out sessions and on stability across training runs; ≤ 1 % wrong actions is a hard
        requirement.
        <ul>
          <li><strong>Sweep, default:</strong> highest Final Test Session score ({pct(d.final_test.B2_s0)}) and the most consistent across seeds (±{(d.tuning_sd.B2 * 100).toFixed(1)}). Confirmed in live testing.</li>
          <li><strong>Double-Check, safety mode:</strong> {pct(d.agree.wrong, 2)} wrong actions and {pct(d.agree.far, 2)} false accepts, at the cost of more “please repeat”.</li>
          <li><strong>Focus, alternative:</strong> best with other speakers and at ignoring random words. The recommended default for multi-speaker use.</li>
          <li><strong>Benchmark (DS-CNN), reference:</strong> built for 1-second single words, it underfits 1.7–3-second multi-word commands.</li>
        </ul>
      </div>
      <EdgeFigure />
      <div className="about-deploy">
        <div>ONNX export, with parity vs PyTorch max |Δprob| = {d.parity.B2_s0.toExponential(1)}</div>
        <table className="grid about-table">
          <thead><tr><th>Model</th><th>file size</th><th className="left">runtime</th></tr></thead>
          <tbody>
            {MODEL_ORDER.filter((m) => d.deployment[m]).map((m) => (
              <tr key={m}>
                <td>{d.deployment[m].name}</td>
                <td className="num">{d.deployment[m].onnx_kb} KB</td>
                <td className="left">{d.deployment[m].runtime} · {d.deployment[m].quantization}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div><strong>Pi 4 latency: pending measurement</strong> (budget ≤ 50 ms model, ≤ 30 ms features)</div>
      </div>
      <ul className="about-next">
        <li>classmates’ voices (Blend v2)</li>
        <li>Pi latency validation</li>
        <li>int8 path for Focus</li>
        <li>a stronger benchmark (BC-ResNet)</li>
      </ul>
      <GitHubLinks repoUrl={d.repo_url} panel={2} />
    </section>
  );
}

export function About({ open, presentation, onClose }: { open: boolean; presentation: boolean; onClose: () => void }) {
  const { data, error } = useAboutData(open);
  const [panel, setPanel] = useState(0);
  useEffect(() => setPanel(0), [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (ev: KeyboardEvent) => {
      if ((ev.target as HTMLElement)?.tagName.match(/INPUT|SELECT|TEXTAREA/)) return;
      if (ev.key === "Escape") { onClose(); return; }
      if (!presentation || !data) return;
      if (ev.key === "ArrowRight" || ev.key === "PageDown") setPanel((p) => Math.min(2, p + 1));
      if (ev.key === "ArrowLeft" || ev.key === "PageUp") setPanel((p) => Math.max(0, p - 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, presentation, data, onClose]);

  if (!open) return null;
  if (error) return <main className="about muted">Could not load About data: {error}</main>;
  if (!data) return <main className="about muted">Loading About…</main>;

  const panels = [<Panel1 key="p1" d={data} />, <Panel2 key="p2" d={data} />, <Panel3 key="p3" d={data} />];

  return (
    <main className="about" aria-label="About">
      {presentation ? (
        <div className="about-present">
          <div className="about-pager muted small">{panel + 1} / 3</div>
          {panels[panel]}
        </div>
      ) : (
        <>{panels}</>
      )}
      <footer className="about-foot muted small">
        Numbers generated {data.generated_at} from tiny-vcm @ {data.tvcm_commit}.
      </footer>
    </main>
  );
}
