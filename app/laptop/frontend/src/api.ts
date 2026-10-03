import { useEffect, useRef, useState } from "react";
import type { AppState, TimelineEvent } from "./types";

export async function post(path: string, body?: unknown): Promise<Response> {
  return fetch(path, {
    method: "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export async function del(path: string): Promise<Response> {
  return fetch(path, { method: "DELETE" });
}

/** Live app state over /ws/ui with automatic reconnect. */
export function useAppState() {
  const [state, setState] = useState<AppState | null>(null);
  const [online, setOnline] = useState(false);
  const [level, setLevel] = useState<number | null>(null);
  const [wakeScore, setWakeScore] = useState<number | null>(null);
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const phaseSince = useRef<{ phase: string; at: number }>({ phase: "", at: Date.now() });

  useEffect(() => {
    let ws: WebSocket | null = null;
    let stop = false;
    let retry: number | undefined;
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws/ui`);
      ws.onopen = () => setOnline(true);
      ws.onclose = () => {
        setOnline(false);
        if (!stop) retry = window.setTimeout(connect, 1500);
      };
      ws.onmessage = (m) => {
        const msg = JSON.parse(m.data);
        if (msg.kind === "state") {
          const s: AppState = msg.state;
          if (s.phase.phase !== phaseSince.current.phase) {
            phaseSince.current = { phase: s.phase.phase, at: Date.now() };
          }
          setState(s);
          setEvents(s.timeline);
        } else if (msg.kind === "event") {
          setEvents((ev) => [...ev.slice(-59), msg.event]);
        } else if (msg.kind === "level") {
          setLevel(msg.dbfs);
          setWakeScore(msg.wake_score ?? null);
        }
      };
    };
    connect();
    return () => {
      stop = true;
      window.clearTimeout(retry);
      ws?.close();
    };
  }, []);

  return { state, online, level, wakeScore, events, phaseSince: phaseSince.current };
}

/** Re-render every `ms` milliseconds (countdowns). */
export function useTick(ms: number, active = true) {
  const [, set] = useState(0);
  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => set((x) => x + 1), ms);
    return () => window.clearInterval(id);
  }, [ms, active]);
}

export const prettyIntent = (intent: string) =>
  intent
    .toLowerCase()
    .split("_")
    .map((w, i) => (i === 0 ? w[0].toUpperCase() + w.slice(1) : w))
    .join(" ");

export const prettyKey = (key: string) => {
  const [intent, slot] = key.split("|");
  return slot ? `${prettyIntent(intent)}: ${slot}` : prettyIntent(intent);
};

/** Friendly name for a model id (falls back to the id). */
export const modelName = (
  names: { models: Record<string, { name: string }> } | undefined,
  id: string,
) => names?.models[id]?.name ?? id;

/** Optional role tag for a model id ("" when absent). Only "benchmark" today. */
export const modelRole = (
  names: { models: Record<string, { name: string; role?: string }> } | undefined,
  id: string,
) => names?.models[id]?.role ?? "";

/** Display order for model ids: normal models first, benchmark-role models last. */
export const orderModels = <T extends { id: string }>(
  names: { models: Record<string, { name: string; role?: string }> } | undefined,
  models: T[],
): T[] =>
  [...models].sort(
    (a, b) => Number(modelRole(names, a.id) === "benchmark") - Number(modelRole(names, b.id) === "benchmark"),
  );

/** Friendly name for a dataset id (falls back to the id). */
export const datasetName = (
  names: { datasets: Record<string, { name: string }> } | undefined,
  id: string,
) => names?.datasets[id]?.name ?? id;

/** Public alias for a dataset id (falls back to the id). */
export const datasetDisplayId = (
  names: { datasets: Record<string, { name: string; display_id?: string }> } | undefined,
  id: string,
) => names?.datasets[id]?.display_id ?? id;
