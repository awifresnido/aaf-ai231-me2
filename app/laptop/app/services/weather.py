"""Weather behind a provider interface: Open-Meteo (no API key) with an
offline fixture fallback that is always labelled as cached in the UI."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import httpx

log = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
PROVIDER_NAME = "Open-Meteo"
PROVIDER_URL = "https://open-meteo.com"

# WMO weather interpretation codes (Open-Meteo docs), condensed
WMO_TEXT = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain", 80: "Rain showers",
    81: "Heavy showers", 82: "Violent showers", 95: "Thunderstorm",
    96: "Thunderstorm with hail", 99: "Severe thunderstorm",
}


class WeatherService:
    def __init__(self, cfg: dict, fixture_path: Path):
        self.cfg = cfg
        self.fixture_path = fixture_path
        self.last: Optional[dict] = None
        self.internet_ok: Optional[bool] = None

    async def fetch(self, now_ms: float) -> dict:
        params = {
            "latitude": self.cfg["latitude"], "longitude": self.cfg["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                       "weather_code,wind_speed_10m",
            "timezone": "auto",
        }
        try:
            async with httpx.AsyncClient(timeout=float(self.cfg.get("timeout_s", 4.0))) as c:
                r = await c.get(OPEN_METEO_URL, params=params)
                r.raise_for_status()
                cur = r.json()["current"]
            self.internet_ok = True
            data = self._shape(cur, cached=False)
            data.update(provider=PROVIDER_NAME, provider_url=PROVIDER_URL,
                        endpoint=OPEN_METEO_URL, mode="live")
        except Exception as exc:  # noqa: BLE001 -- the demo must not crash on network loss
            self.internet_ok = False
            log.warning("live weather unavailable: %s", exc)
            if not self.cfg.get("fallback_enabled", True):
                data = {"available": False, "error": "Live weather unavailable",
                        "provider": PROVIDER_NAME, "provider_url": PROVIDER_URL,
                        "endpoint": OPEN_METEO_URL, "mode": "unavailable"}
            else:
                fx = json.loads(self.fixture_path.read_text(encoding="utf-8"))
                data = self._shape(fx["current"], cached=True)
                data["cached_note"] = fx.get("note", "cached fixture")
                data.update(provider=PROVIDER_NAME, provider_url=PROVIDER_URL,
                            endpoint=OPEN_METEO_URL, mode="cached_fixture")
        data["location"] = self.cfg.get("location_name", "")
        data["fetched_ms"] = now_ms
        self.last = data
        return data

    @staticmethod
    def _shape(cur: dict, cached: bool) -> dict:
        code = int(cur.get("weather_code", -1))
        return {
            "available": True, "cached": cached,
            "temperature_c": cur.get("temperature_2m"),
            "apparent_c": cur.get("apparent_temperature"),
            "humidity_pct": cur.get("relative_humidity_2m"),
            "wind_kmh": cur.get("wind_speed_10m"),
            "code": code, "condition": WMO_TEXT.get(code, f"WMO code {code}"),
            "observed": cur.get("time"),
        }
