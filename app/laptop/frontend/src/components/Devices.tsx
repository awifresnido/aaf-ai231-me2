import { useState } from "react";
import { del, post, useTick } from "../api";
import type { AppState, Media } from "../types";

const BULB_COLORS: Record<string, string> = {
  Red: "#E0473A",
  Blue: "#3A7BE0",
  Green: "#3DAA5C",
};
const WARM_WHITE = "#FFC857";

function Card({ title, children, focus }: { title: string; children: React.ReactNode; focus?: boolean }) {
  return (
    <section className={`card ${focus ? "focus" : ""}`}>
      <h2>{title}</h2>
      {children}
    </section>
  );
}

export function LightCard({ light }: { light: AppState["devices"]["light"] }) {
  const col = light.color ? BULB_COLORS[light.color] ?? WARM_WHITE : WARM_WHITE;
  const glow = light.power ? 0.25 + 0.75 * (light.brightness_pct / 100) : 0;
  return (
    <Card title="Smart light">
      <div className="light">
        <svg viewBox="0 0 120 150" className="bulb" aria-hidden>
          <defs>
            <radialGradient id="glow">
              <stop offset="0%" stopColor={col} stopOpacity={glow} />
              <stop offset="100%" stopColor={col} stopOpacity={0} />
            </radialGradient>
          </defs>
          <circle cx="60" cy="58" r="58" fill="url(#glow)" />
          <path
            d="M60 14c-22 0-38 16-38 37 0 14 7 23 14 31 5 6 8 11 8 18h32c0-7 3-12 8-18 7-8 14-17 14-31 0-21-16-37-38-37z"
            fill={light.power ? col : "#C9CED3"}
            fillOpacity={light.power ? 0.35 + 0.65 * (light.brightness_pct / 100) : 1}
            stroke="#56606A"
            strokeWidth="2"
          />
          <rect x="44" y="104" width="32" height="8" rx="2" fill="#8A939B" />
          <rect x="46" y="114" width="28" height="8" rx="2" fill="#8A939B" />
          <rect x="50" y="124" width="20" height="8" rx="3" fill="#6E777F" />
        </svg>
        <dl className="facts">
          <dt>Power</dt>
          <dd>{light.power ? "On" : "Off"}</dd>
          <dt>Brightness</dt>
          <dd>{light.brightness}</dd>
          <dt>Colour</dt>
          <dd>{light.color ?? "Warm white"}</dd>
        </dl>
      </div>
    </Card>
  );
}

export function AirconCard({ t }: { t: AppState["devices"]["thermostat"] }) {
  return (
    <Card title="Air conditioner">
      <div className="aircon">
        <div className="setpoint">
          <span className="num big">{t.setpoint_c}</span>
          <span className="unit">°C</span>
        </div>
        <div className="scale" aria-label="Available set-points">
          {t.options_c.map((c) => (
            <span key={c} className={c === t.setpoint_c ? "on" : ""}>
              {c}
            </span>
          ))}
        </div>
        <p className="muted">Cooling. The model can only choose these set-points.</p>
      </div>
    </Card>
  );
}

export function MediaCard({ m }: { m: Media | undefined }) {
  if (!m) return <Card title="Music">{<p className="muted">Waiting for the edge.</p>}</Card>;
  const act = (a: string) => post(`/api/media/${a}`);
  return (
    <Card title="Music">
      <div className="media">
        <div className="track">
          <span className={`eq ${m.status === "playing" && !m.ducked ? "on" : ""}`} aria-hidden>
            <i /> <i /> <i />
          </span>
          <div>
            <div className="title">{m.track_title}</div>
            <div className="muted">
              {m.status === "playing" ? (m.ducked ? "Ducked while listening" : "Playing") : m.status === "paused" ? "Paused" : "Stopped"}
              {m.track_count > 0 ? `, track ${m.track_index + 1} of ${m.track_count}` : ""}
            </div>
          </div>
        </div>
        <div className="vol" aria-label={`Volume ${m.level} of ${m.max_level}`}>
          {Array.from({ length: m.max_level }, (_, i) => (
            <span key={i} className={i < m.level ? "on" : ""} style={{ height: `${10 + i * 5}px` }} />
          ))}
          <span className="num muted">{m.level_pct}%</span>
        </div>
        <div className="controls">
          <button className="btn small" onClick={() => act(m.status === "playing" ? "pause" : "play")}>
            {m.status === "playing" ? "Pause" : "Play"}
          </button>
          <button className="btn small" onClick={() => act("next")}>Next</button>
          <button className="btn small" onClick={() => act("stop")}>Stop</button>
          <button className="btn small" onClick={() => act("volume_down")} aria-label="Volume down">−</button>
          <button className="btn small" onClick={() => act("volume_up")} aria-label="Volume up">+</button>
        </div>
        {!m.backend_ok || m.backend === "simulated" ? (
          <p className="muted small">{m.backend_detail ?? "Speaker unavailable"}</p>
        ) : null}
      </div>
    </Card>
  );
}

export function ClockCard({ timer, alarm }: { timer: AppState["devices"]["timer"]; alarm: AppState["devices"]["alarm"] }) {
  useTick(250, timer.status === "running");
  const left = timer.ends_at_ms ? Math.max(0, (timer.ends_at_ms - Date.now()) / 1000) : 0;
  const mmss = `${String(Math.floor(left / 60)).padStart(2, "0")}:${String(Math.floor(left % 60)).padStart(2, "0")}`;
  return (
    <Card title="Timer and alarm">
      <div className="clock">
        <div className={`timer ${timer.status}`}>
          <span className="num big">{timer.status === "running" ? mmss : timer.status === "done" ? "00:00" : "--:--"}</span>
          <span className="muted">
            {timer.status === "running" ? `Timer for ${timer.label}` : timer.status === "done" ? "Timer complete" : "No timer"}
          </span>
          {timer.status !== "idle" && (
            <button className="btn small" onClick={() => post("/api/timer/cancel")}>
              {timer.status === "done" ? "Clear" : "Cancel"}
            </button>
          )}
        </div>
        <div className={`alarm ${alarm.status}`}>
          <span className="muted">Next alarm</span>
          <span className="num mid">{alarm.slot ?? "None set"}</span>
          {alarm.status !== "none" && (
            <button className="btn small" onClick={() => post("/api/alarm/dismiss")}>
              {alarm.status === "ringing" ? "Stop alarm" : "Delete"}
            </button>
          )}
        </div>
      </div>
    </Card>
  );
}

export function RemindersCard({ r }: { r: AppState["devices"]["reminders"] }) {
  useTick(1000, r.focus_until_ms > Date.now());
  const [text, setText] = useState("");
  const focus = r.focus_until_ms > Date.now();
  const add = async () => {
    if (!text.trim()) return;
    await post("/api/reminders", { text });
    setText("");
  };
  return (
    <Card title="Reminders" focus={focus}>
      <ul className="reminders">
        {r.items.length === 0 && <li className="muted">Nothing yet. Say “Remind me to study”, or add one below.</li>}
        {r.items.slice(0, 7).map((it) => (
          <li key={it.id} className={it.done ? "done" : ""}>
            <label>
              <input type="checkbox" checked={!!it.done} onChange={(e) => post(`/api/reminders/${it.id}/toggle?done=${e.target.checked}`)} />
              {it.text}
            </label>
            <span className="muted small">{it.source === "voice" ? "voice" : "typed"}</span>
            <button className="btn link" onClick={() => del(`/api/reminders/${it.id}`)} aria-label={`Delete ${it.text}`}>
              Delete
            </button>
          </li>
        ))}
      </ul>
      <form
        className="add"
        onSubmit={(e) => {
          e.preventDefault();
          add();
        }}
      >
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder="Add a reminder" aria-label="New reminder" />
        <button className="btn small" type="submit">Add</button>
      </form>
    </Card>
  );
}

export function PhoneCard({ p }: { p: AppState["devices"]["phone"] }) {
  useTick(1000, p.call_status === "connected");
  const secs = p.call_started_ms ? Math.floor((Date.now() - p.call_started_ms) / 1000) : 0;
  return (
    <Card title="Phone">
      <div className={`phone ${p.call_status}`}>
        {p.call_status === "idle" ? (
          <p className="muted">
            Calls and messages go to {p.contact.name}. The VCM has no contact slot.
          </p>
        ) : (
          <div className="call">
            <div className="who">{p.contact.name}</div>
            <div className="muted num">{p.contact.phone}</div>
            <div className="status">{p.call_status === "calling" ? "Calling" : `Connected ${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`}</div>
            <button className="btn danger small" onClick={() => post("/api/phone/end")}>End call</button>
          </div>
        )}
        {p.messages.length > 0 && (
          <ul className="bubbles">
            {p.messages.slice(-3).map((m) => (
              <li key={m.ts_ms}>
                <span>{m.text}</span>
                <time className="muted small">{new Date(m.ts_ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Card>
  );
}

export function Overlays({ state }: { state: AppState }) {
  const { toast, weather, timer, alarm, reminders } = state.devices;
  useTick(500, !!toast || !!weather);
  const open = reminders.items.filter((it) => !it.done);
  const done = reminders.items.filter((it) => !!it.done);
  return (
    <>
      {toast?.kind === "reminders" && (
        <div className="overlay reminders" role="status">
          <h2>Your reminders</h2>
          <p className="muted">
            {open.length} open, {done.length} done
          </p>
          <ul>
            {[...open.slice(0, 6), ...done.slice(0, Math.max(0, 6 - open.length))]
              .slice(0, 6)
              .map((it) => (
                <li key={it.id} className={it.done ? "done" : ""}>
                  <span>{it.text}</span>
                  <span className="muted small">{it.source === "voice" ? "voice" : "typed"}</span>
                </li>
              ))}
            {open.length + done.length > 6 && (
              <li className="muted small">+{open.length + done.length - 6} more</li>
            )}
            {reminders.items.length === 0 && (
              <li className="muted">No reminders yet. Say &ldquo;Remind me to study&rdquo;.</li>
            )}
          </ul>
        </div>
      )}
      {toast?.kind === "time" && (
        <div className="overlay time" role="status">
          <div className="num huge">
            {new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit", timeZone: toast.tz })}
          </div>
          <div>{new Date().toLocaleDateString([], { weekday: "long", month: "long", day: "numeric", timeZone: toast.tz })}</div>
        </div>
      )}
      {weather && (
        <div className="overlay weather" role="status">
          <h2>{weather.location} weather</h2>
          {weather.available ? (
            <>
              <div className="wx-main">
                <span className="num huge">{weather.temperature_c?.toFixed(0)}°</span>
                <span>{weather.condition}</span>
              </div>
              <dl className="facts">
                <dt>Feels like</dt>
                <dd className="num">{weather.apparent_c?.toFixed(0)} °C</dd>
                <dt>Humidity</dt>
                <dd className="num">{weather.humidity_pct}%</dd>
                <dt>Wind</dt>
                <dd className="num">{weather.wind_kmh?.toFixed(0)} km/h</dd>
              </dl>
              {weather.cached && <p className="badge warn">Offline fixture, not live: {weather.cached_note}</p>}
              <p className="muted small source">
                {weather.mode === "live"
                  ? `Source: ${weather.provider} API (open-meteo.com), no API key, live data` +
                    (weather.observed ? `, observed ${String(weather.observed).slice(11, 16)}` : "")
                  : `Source: saved ${weather.provider} sample (live API unreachable)`}
              </p>
            </>
          ) : (
            <>
              <p>{weather.error}</p>
              <p className="muted small source">
                Source: {weather.provider ?? "Open-Meteo"} API, request failed
              </p>
            </>
          )}
        </div>
      )}
      {(timer.status === "done" || alarm.status === "ringing") && (
        <div className="banner" role="alert">
          {timer.status === "done" ? "Timer complete" : `Alarm: ${alarm.slot}`}
          <button className="btn small" onClick={() => post(timer.status === "done" ? "/api/timer/cancel" : "/api/alarm/dismiss")}>
            Dismiss
          </button>
        </div>
      )}
    </>
  );
}
