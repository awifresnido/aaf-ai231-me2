import { useEffect, useState } from "react";
import { modelName, orderModels, post, useAppState } from "./api";
import { useWakeChime } from "./sound";
import { AssistantCore, Decision } from "./components/Assistant";
import { AirconCard, ClockCard, LightCard, MediaCard, Overlays, PhoneCard, RemindersCard } from "./components/Devices";
import { About } from "./components/About";
import { Benchmark } from "./components/Benchmark";
import { CommandLog, DevTools, Health, ModelRegistry, Scoreboard } from "./components/Engineering";
import { Trace } from "./components/Trace";
import type { AppState } from "./types";

type Lamp = { label: string; tone: "ok" | "warn" | "alarm" | "off"; text: string; title?: string };

function lamps(s: AppState): Lamp[] {
  const e = s.edge;
  const h = e?.health;
  const active = e?.models.find((m) => m.id === e.active_vcm);
  return [
    (() => {
      const skew = Math.abs(s.edge_clock?.offset_ms ?? 0) > 2000;
      return {
        label: "Edge",
        tone: s.link.connected ? (skew ? "warn" : "ok") : "alarm",
        text: s.link.connected ? (skew ? "clock corrected" : (h?.device ?? "connected")) : "offline",
        title: skew ? "Pi clock not synced, corrected by the app" : undefined,
      };
    })(),
    {
      label: "Mic",
      tone: h?.audio_stream ? "ok" : s.mic.status === "error" ? "alarm" : "off",
      text: h?.audio_stream ? "streaming" : s.mic.status === "error" ? "error" : s.mic.enabled ? "silent" : "off",
    },
    (() => {
      const w = e?.wake_engines.find((x) => x.id === e.active_wake);
      if (!w || w.kind === "manual") return { label: "Wake", tone: "warn" as const, text: "manual" };
      return h?.wake_ready
        ? { label: "Wake", tone: "ok" as const, text: `“${w.phrase ?? w.id}”` }
        : { label: "Wake", tone: "alarm" as const, text: "unavailable" };
    })(),
    { label: "VCM", tone: h?.vcm_ready ? "ok" : "alarm", text: h?.vcm_ready ? (active ? modelName(s.display_names, active.id) : "ready") : "not loaded" },
    {
      label: "Speaker",
      tone: e?.media.backend === "mpv" && e.media.backend_ok ? "ok" : "warn",
      text: e?.media.backend === "mpv" && e.media.backend_ok ? "mpv" : "simulated",
    },
    { label: "Internet", tone: s.internet_ok == null ? "off" : s.internet_ok ? "ok" : "warn", text: s.internet_ok == null ? "unchecked" : s.internet_ok ? "ok" : "offline" },
  ];
}

export default function App() {
  const { state, online, level, wakeScore, events, phaseSince } = useAppState();
  const chime = useWakeChime(state?.link.connected ? state.phase.phase : undefined);
  const [presentation, setPresentation] = useState(() => new URLSearchParams(location.search).has("present"));
  const [eng, setEng] = useState(false);
  const [about, setAbout] = useState(() => new URLSearchParams(location.search).has("about"));
  const [benchmark, setBenchmark] = useState(() => new URLSearchParams(location.search).has("benchmark"));

  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if ((ev.target as HTMLElement)?.tagName.match(/INPUT|SELECT|TEXTAREA/)) return;
      if (ev.key === "e") setEng((x) => !x);
      if (ev.key === "a") setAbout((x) => !x);
      if (ev.key === "b") setBenchmark((x) => !x);
      if (ev.key === "p") setPresentation((x) => !x);
      if (ev.key === " " && state?.phase.phase === "idle") {
        ev.preventDefault();
        post("/api/wake");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [state?.phase.phase]);

  if (!state) {
    return <main className="boot">{online ? "Loading…" : "Connecting to the TinyVCM app server…"}</main>;
  }
  const e = state.edge;
  // benchmark-role models (e.g. Benchmark (DS-CNN)) are listed last
  const selectable = orderModels(state.display_names, e?.models.filter((m) => m.status !== "invalid") ?? []);

  return (
    <div className={`app ${presentation ? "presentation" : ""}`}>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden />
          <span>TinyVCM Control Center</span>
        </div>
        <ul className="lamps">
          {lamps(state).map((l) => (
            <li key={l.label} className={`lamp ${l.tone}`} title={l.title}>
              <i aria-hidden />
              <span className="lamp-label">{l.label}</span>
              <span className="lamp-text">{l.text}</span>
            </li>
          ))}
        </ul>
        <div className="top-actions">
          <label className="picker">
            <span>Model</span>
            <select
              value={e?.active_vcm ?? ""}
              disabled={!e}
              onChange={(ev) => post("/api/models/select", { model_id: ev.target.value })}
            >
              {selectable.map((m) => (
                <option
                  key={m.id}
                  value={m.id}
                  disabled={m.status === "unavailable"}
                  title={[m.description, m.based_on].filter(Boolean).join(" — ")}
                >
                  {modelName(state.display_names, m.id)}
                  {m.status === "unavailable" ? " (unavailable)" : ""}
                </option>
              ))}
            </select>
          </label>
          <button
            className="btn small"
            aria-pressed={chime.enabled}
            onClick={chime.toggle}
            title={chime.enabled && !chime.unlocked ? "Click anywhere once so the browser allows sound" : "Chime when the wake word is heard"}
          >
            {chime.enabled ? (chime.unlocked ? "Wake sound on" : "Wake sound: click to allow") : "Wake sound off"}
          </button>
          {!presentation && (
            <button className="btn small" aria-pressed={eng} onClick={() => setEng((x) => !x)}>
              Engineering
            </button>
          )}
          <button className="btn small" aria-pressed={about} onClick={() => setAbout((x) => !x)}>
            About
          </button>
          <button className="btn small" aria-pressed={benchmark} onClick={() => setBenchmark((x) => !x)}>
            Benchmark
          </button>
          <button className="btn small" aria-pressed={presentation} onClick={() => setPresentation((x) => !x)}>
            {presentation ? "Exit presentation" : "Present"}
          </button>
        </div>
      </header>

      {!online && <div className="banner alarm-bg">Lost connection to the app server. Retrying.</div>}

      {about ? (
        <About open={about} presentation={presentation} onClose={() => setAbout(false)} />
      ) : benchmark ? (
        <Benchmark open={benchmark} presentation={presentation} state={state} onClose={() => setBenchmark(false)} />
      ) : (
        <>
          <main className="stage">
            <AssistantCore state={state} level={level} phaseSince={phaseSince} presentation={presentation} />
            <Decision state={state} presentation={presentation} />
            <Trace events={events} names={state.display_names} clock={state.edge_clock} />
          </main>

          <section className="devices">
            <LightCard light={state.devices.light} />
            <AirconCard t={state.devices.thermostat} />
            <MediaCard m={e?.media} />
            <ClockCard timer={state.devices.timer} alarm={state.devices.alarm} />
            <RemindersCard r={state.devices.reminders} />
            <PhoneCard p={state.devices.phone} />
          </section>
        </>
      )}

      {!about && eng && !presentation && (
        <section className="engineering" aria-label="Engineering view">
          <ModelRegistry state={state} wakeScore={wakeScore} />
          <Scoreboard state={state} />
          <div className="eng-split">
            <DevTools state={state} />
            <Health state={state} />
          </div>
          <CommandLog state={state} />
        </section>
      )}
      {!about && eng && presentation && (
        <section className="engineering" aria-label="Engineering view">
          <ModelRegistry state={state} wakeScore={wakeScore} />
          <Health state={state} />
        </section>
      )}

      <footer className="foot muted small">
        Ontology {state.ontology.schema_version}. Keys: Space wakes, E engineering, A About, B Benchmark, P presentation.
        Audio is classified on the edge; this app only executes the returned intent.
      </footer>
      <Overlays state={state} />
    </div>
  );
}
