import { useEffect, useRef, useState } from "react";

/* Wake chime, synthesised with WebAudio (no audio asset to license or load).
   Same design as edge/chime.py: a soft rising fifth, A5 -> E6, ~0.2 s total.
   Browsers only allow audio after a user gesture, so the context is unlocked
   on the first click/keypress anywhere on the page. */

const TONES_HZ = [880, 1318.5];
const TONE_S = 0.09;
const PEAK_GAIN = 0.12;
const STORAGE_KEY = "tinyvcm.wakeSound";

let ctx: AudioContext | null = null;

function ensureContext(): AudioContext | null {
  if (!ctx) {
    const AC = window.AudioContext ?? (window as any).webkitAudioContext;
    if (!AC) return null;
    ctx = new AC();
  }
  if (ctx.state === "suspended") void ctx.resume();
  return ctx;
}

export function playWakeChime(): boolean {
  const c = ensureContext();
  if (!c || c.state !== "running") return false;
  const t0 = c.currentTime + 0.01;
  TONES_HZ.forEach((f, i) => {
    const start = t0 + i * TONE_S;
    const osc = c.createOscillator();
    const gain = c.createGain();
    osc.type = "sine";
    osc.frequency.value = f;
    gain.gain.setValueAtTime(0, start);
    gain.gain.linearRampToValueAtTime(PEAK_GAIN, start + 0.005);
    gain.gain.exponentialRampToValueAtTime(0.0001, start + TONE_S + 0.12);
    osc.connect(gain).connect(c.destination);
    osc.start(start);
    osc.stop(start + TONE_S + 0.15);
  });
  return true;
}

/** Plays the chime on every transition into `wake_detected` while enabled. */
export function useWakeChime(phase: string | undefined) {
  const [enabled, setEnabled] = useState(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) !== "off";
    } catch {
      return true;
    }
  });
  const [unlocked, setUnlocked] = useState(false);
  const prev = useRef<string | undefined>(undefined);

  useEffect(() => {
    const unlock = () => {
      const c = ensureContext();
      if (c) c.resume().then(() => setUnlocked(c.state === "running"));
    };
    window.addEventListener("pointerdown", unlock);
    window.addEventListener("keydown", unlock);
    return () => {
      window.removeEventListener("pointerdown", unlock);
      window.removeEventListener("keydown", unlock);
    };
  }, []);

  useEffect(() => {
    if (phase === "wake_detected" && prev.current !== "wake_detected" && enabled) playWakeChime();
    prev.current = phase;
  }, [phase, enabled]);

  const toggle = () => {
    const next = !enabled;
    setEnabled(next);
    try {
      localStorage.setItem(STORAGE_KEY, next ? "on" : "off");
    } catch {
      /* storage unavailable: setting lasts for this session only */
    }
    if (next) playWakeChime();   // audible confirmation when switching on
  };

  return { enabled, unlocked, toggle };
}
