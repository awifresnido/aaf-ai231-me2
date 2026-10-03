import json
from pathlib import Path

import httpx
import pytest

from laptop.app.services.weather import WeatherService

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "data" / "weather_fixture.json"


def _svc(**kw):
    cfg = {"latitude": 10.32, "longitude": 123.9, "timeout_s": 0.5, "fallback_enabled": True,
           "location_name": "UP Cebu"}
    cfg.update(kw)
    return WeatherService(cfg, FIXTURE)


class MockResponse(httpx.Response):
    def raise_for_status(self):
        return None


def _live_current():
    return {"temperature_2m": 29.1, "apparent_temperature": 31.0,
            "relative_humidity_2m": 70, "weather_code": 1, "wind_speed_10m": 8.0,
            "time": "2026-09-29T12:00"}


def test_live_success_sets_provider_and_mode(monkeypatch):
    async def fake_get(self, url, params=None):
        return MockResponse(200, json={"current": _live_current()})

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    async def run():
        return await _svc().fetch(0)
    import asyncio
    d = asyncio.run(run())
    assert d["provider"] == "Open-Meteo"
    assert d["mode"] == "live"
    assert d["available"] is True


def test_fallback_sets_cached_fixture(monkeypatch):
    async def fake_get(self, url, params=None):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    import asyncio
    d = asyncio.run(_svc().fetch(0))
    assert d["mode"] == "cached_fixture"
    assert d["provider"] == "Open-Meteo"
    assert d["available"] is True and d["cached"] is True


def test_fallback_disabled_sets_unavailable(monkeypatch):
    async def fake_get(self, url, params=None):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    import asyncio
    d = asyncio.run(_svc(fallback_enabled=False).fetch(0))
    assert d["mode"] == "unavailable"
    assert d["provider"] == "Open-Meteo"
    assert d["available"] is False
