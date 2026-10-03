import { modelName, prettyIntent } from "../api";
import type { AppState, ClockQuality, TimelineEvent } from "../types";

type Names = AppState["display_names"];

const LABELS: Record<string, (p: Record<string, any>, names?: Names) => string> = {
  wake_detected: (p) =>
    `Wake (${p.wake_engine}${p.confidence != null ? `, ${Number(p.confidence).toFixed(2)}` : ""})${p.clip_name ? `: ${p.clip_name}` : ""}`,
  media_ducked: () => "Music ducked",
  capture_started: (p) => (p.window_s > 0 ? `Listening ${p.window_s.toFixed(1)} s` : "Clip received"),
  vad_speech_started: (p) => `Speech started (+${p.at_ms} ms)`,
  vad_speech_ended: (p) => `Speech ended (+${p.at_ms} ms)`,
  vad_no_speech: () => "No speech heard",
  vad_max_window: () => "Still talking at the limit",
  capture_endpoint: (p) =>
    `Endpoint: ${p.reason} after ${(p.elapsed_ms / 1000).toFixed(1)} s · VAD ${p.vad_ms_mean} ms/chunk`,
  capture_finished: (p) => `Captured ${(p.clip_ms / 1000).toFixed(2)} s at ${p.rms_dbfs} dBFS`,
  capture_failed: (p) => `Capture failed: ${p.reason}`,
  inference_started: () => "VCM inference started",
  inference_error: (p, names) => `Inference error (${modelName(names, p.model_id)}): ${p.detail}`,
  result_received: (p) =>
    `${p.accepted ? "Accepted" : "Rejected"}: ${prettyIntent(p.intent)}${p.slot ? ` [${p.slot}]` : ""} at ${(p.confidence * 100).toFixed(0)}%`,
  action_completed: (p) => `Action: ${p.action}`,
  action_skipped: (p) => `No action${p.detail ? `: ${p.detail}` : ""}`,
  media_action: (p) => `Media ${p.status}, volume ${p.level}`,
  media_restored: (p) => `Music restored to volume ${p.level}`,
  model_selected: (p, names) => `Active model: ${modelName(names, p.model_id)}`,
  model_load_failed: (p, names) => `Model ${modelName(names, p.model_id)} failed to load: ${p.detail}`,
  edge_connected: () => "Edge connected",
  edge_disconnected: () => "Edge disconnected",
  timer_done: () => "Timer finished",
  alarm_ringing: (p) => `Alarm ${p.slot}`,
  call_connected: (p) => `Call connected to ${p.contact}`,
  edge_clock_jump: (p) =>
    `Pi clock changed by ${p.delta_ms >= 0 ? "+" : "−"}${fmtOffset(p.delta_ms)}; re-synced`,
};

const TRUSTED: ClockQuality[] = ["local", "accurate"];

function qualityLine(clock?: AppState["edge_clock"]): string | null {
  if (!clock || !clock.quality) return null;
  switch (clock.quality) {
    case "local":
      return "Timing: exact (edge runs in this app, one clock)";
    case "accurate":
      return `Timing: accurate (NTP-style sync${clock.spread_ms != null ? `, ±${Math.round(clock.spread_ms)} ms` : ""})`;
    case "approximate":
      return "Timing: approximate (the Pi sends no clock reply; update the Pi's edge code)";
    case "syncing":
      return "Timing: syncing…";
    default:
      return null;
  }
}

function fmtOffset(ms: number): string {
  const abs = Math.abs(ms);
  const h = Math.floor(abs / 3600000);
  const m = Math.floor((abs % 3600000) / 60000);
  const s = Math.floor((abs % 60000) / 1000);
  return `${h} h ${m} min ${s} s`;
}

export function Trace({ events, names, clock }: {
  events: TimelineEvent[]; names?: Names; clock?: AppState["edge_clock"];
}) {
  const lastCmd = [...events].reverse().find((e) => e.command_id)?.command_id;
  const shown = events.filter((e) => e.command_id === lastCmd || (!e.command_id && e.type !== "busy_ignored")).slice(-16);
  const cmdEvents = shown.filter((e) => e.command_id === lastCmd);
  const t0 = cmdEvents.find((e) => e.command_id === lastCmd)?.ts_ms;
  // cross-device deltas need every edge row of this command to be timing_valid;
  // they are exact only under a trusted clock (local / NTP-style), else marked ≈
  const edgeRows = cmdEvents.filter((e) => e.origin === "edge");
  const allValid = edgeRows.every((e) => e.timing_valid);
  const trusted = edgeRows.every((e) => !e.clock_quality || TRUSTED.includes(e.clock_quality));
  const edge0 = cmdEvents.find((e) => e.origin === "edge")?.ts_ms;
  const app0 = cmdEvents.find((e) => e.origin === "laptop")?.ts_ms;
  const skew = clock ? Math.abs(clock.offset_ms) > 2000 : false;
  const relFor = (e: TimelineEvent): string => {
    if (e.command_id !== lastCmd) return "";
    if (allValid) return t0 != null ? `${trusted ? "+" : "≈+"}${Math.round(e.ts_ms - t0)} ms` : "";
    const base = e.origin === "edge" ? edge0 : app0;
    return base != null ? `+${Math.round(e.ts_ms - base)} ms` : "";
  };
  const quality = qualityLine(clock);
  return (
    <section className="trace">
      <h2>Command trace</h2>
      {quality && <p className="muted small">{quality}</p>}
      {skew && clock && (
        <p className="muted small">Pi clock is off by {fmtOffset(clock.offset_ms)}; times corrected</p>
      )}
      {!allValid && cmdEvents.length > 0 && (
        <p className="muted small">clock syncing, cross-device timing hidden</p>
      )}
      {shown.length === 0 ? (
        <p className="muted">Each step of the next command appears here with its timing.</p>
      ) : (
        <ol>
          {shown.map((e, i) => {
            const fmt = LABELS[e.type];
            const rel = relFor(e);
            const piTime = e.ts_edge_ms != null
              ? `Pi time: ${new Date(e.ts_edge_ms).toLocaleTimeString([], { hour12: false })}`
              : undefined;
            return (
              <li key={`${e.ts_ms}-${i}`} className={`origin-${e.origin}`} title={piTime}>
                <time className="num">{new Date(e.ts_ms).toLocaleTimeString([], { hour12: false })}</time>
                <span className="rel num">{rel}</span>
                <span className="what">{fmt ? fmt(e.payload, names) : e.type.replace(/_/g, " ")}</span>
                <span className="who">{e.origin === "edge" ? "Edge" : "App"}</span>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}
