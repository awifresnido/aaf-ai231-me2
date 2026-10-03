"""The single authoritative intent router. React never routes commands."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from vcm_common.ontology import Ontology
from vcm_common.protocol import InferenceResult

from .services.clocks import AlarmService, TimerService
from .services.devices import LightService, ThermostatService
from .services.phone import PhoneService
from .services.reminders import ReminderStore
from .services.weather import WeatherService

EDGE_MEDIA_INTENTS = {"PLAY_MUSIC", "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN"}
REMINDER_FOCUS_MS = 6000
REMINDERS_POPUP_MS = 8000
TIME_TOAST_MS = 3000
WEATHER_CARD_MS = 8000


@dataclass
class Outcome:
    executed: bool
    action: str
    target: Optional[str] = None
    detail: Optional[str] = None


@dataclass
class Services:
    light: LightService
    thermostat: ThermostatService
    timer: TimerService
    alarm: AlarmService
    reminders: ReminderStore
    phone: PhoneService
    weather: WeatherService


class Dispatcher:
    def __init__(self, ontology: Ontology, services: Services, tz: str):
        self.o = ontology
        self.s = services
        self.tz = tz
        self.toast: Optional[dict] = None
        self.weather_card_until_ms = 0.0

    async def dispatch(self, r: InferenceResult, now_ms: float) -> Outcome:
        p = r.primary
        if not r.accepted:
            return Outcome(False, "no action", detail=r.reject_reason)
        err = self.o.validate(p.intent, p.slot)          # defence in depth: edge validated too
        if err:
            return Outcome(False, "no action", detail=f"laptop validation: {err}")
        i, v = p.intent, p.slot
        s = self.s
        if i in EDGE_MEDIA_INTENTS:
            if r.media_handled:
                return Outcome(True, "media handled on edge", "media")
            return Outcome(False, "no action", "media", "edge did not handle media intent")
        if i == "LIGHT_ON":
            return Outcome(True, s.light.on(), "light")
        if i == "LIGHT_OFF":
            return Outcome(True, s.light.off(), "light")
        if i == "BRIGHTNESS":
            return Outcome(True, s.light.brightness(v), "light")
        if i == "COLOR":
            return Outcome(True, s.light.color(v), "light")
        if i == "TEMPERATURE":
            return Outcome(True, s.thermostat.set(v, now_ms), "thermostat")
        if i == "TIMER":
            return Outcome(True, s.timer.start(v, now_ms), "clocks")
        if i == "ALARM":
            return Outcome(True, s.alarm.set(v), "clocks")
        if i == "CREATE_REMINDER":
            s.reminders.add(v, source="voice")
            s.reminders.focus_until_ms = now_ms + REMINDER_FOCUS_MS
            return Outcome(True, f"reminder added: {v}", "reminders")
        if i == "LIST_REMINDERS":
            s.reminders.focus_until_ms = now_ms + REMINDER_FOCUS_MS
            self.toast = {"kind": "reminders", "until_ms": now_ms + REMINDERS_POPUP_MS,
                          "at_ms": now_ms}
            return Outcome(True, "reminders shown", "reminders")
        if i == "CALL":
            return Outcome(True, s.phone.call(now_ms), "phone")
        if i == "MESSAGE":
            return Outcome(True, s.phone.message(now_ms), "phone")
        if i == "TIME":
            self.toast = {"kind": "time", "until_ms": now_ms + TIME_TOAST_MS, "at_ms": now_ms,
                          "tz": self.tz}
            return Outcome(True, "time shown", "time")
        if i == "WEATHER":
            w = await s.weather.fetch(now_ms)
            self.weather_card_until_ms = now_ms + WEATHER_CARD_MS
            if not w.get("available"):
                return Outcome(False, "weather unavailable", "weather", w.get("error"))
            note = "cached fixture" if w.get("cached") else "live"
            return Outcome(True, f"weather shown ({note})", "weather")
        return Outcome(False, "no action", detail=f"no route for {i}")
