import asyncio
from pathlib import Path

import pytest

from laptop.app.command_log import CommandLog
from laptop.app.dispatcher import Dispatcher, Outcome, Services
from laptop.app.services.clocks import AlarmService, TimerService
from laptop.app.services.devices import LightService, ThermostatService
from laptop.app.services.phone import PhoneService
from laptop.app.services.reminders import ReminderStore
from laptop.app.services.weather import WeatherService
from vcm_common.ontology import get_ontology
from vcm_common.protocol import InferenceResult, ModelPrediction

REPO = Path(__file__).resolve().parents[1]
O = get_ontology()
MEDIA = {"PLAY_MUSIC", "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN"}


async def _noop(*_a, **_k):
    return None


def make(tmp_path):
    s = Services(
        light=LightService(O), thermostat=ThermostatService(O), timer=TimerService(_noop),
        alarm=AlarmService(_noop, "Asia/Manila"),
        reminders=ReminderStore(tmp_path / "r.sqlite"),
        phone=PhoneService({"name": "Demo", "phone": "+63"}, "hi", _noop),
        weather=WeatherService({"latitude": "invalid", "longitude": 0, "timeout_s": 0.01,
                                "fallback_enabled": True},
                               REPO / "data" / "weather_fixture.json"),
    )
    return Dispatcher(O, s, "Asia/Manila"), s


def pred(key, model="m", conf=0.95):
    intent, slot = O.decode(key)
    return ModelPrediction(model_id=model, class_key=key, intent=intent, slot=slot, confidence=conf)


def result(key, accepted=True, media_handled=False, conf=0.95, compare=()):
    return InferenceResult(command_id=f"c-{key}", source="replay", primary=pred(key, conf=conf),
                           accepted=accepted, threshold=0.7, clip_ms=1000, t_capture_end_ms=0,
                           t_result_ms=1, media_handled=media_handled, compare=list(compare))


@pytest.mark.parametrize("key", [k for k in O.leaf_labels() if O.is_command(O.decode(k)[0])])
def test_every_command_routes(tmp_path, key):
    async def run():
        d, s = make(tmp_path)
        out = await d.dispatch(result(key, media_handled=O.decode(key)[0] in MEDIA), 0.0)
        assert out.executed, (key, out)
        s.timer.cancel()
        s.phone.end()
    asyncio.run(run())


def test_state_effects(tmp_path):
    async def run():
        d, s = make(tmp_path)
        await d.dispatch(result("BRIGHTNESS|60 percent"), 0)
        assert s.light.state.power and s.light.state.brightness == "60 percent"
        await d.dispatch(result("COLOR|Blue"), 0)
        assert s.light.state.color == "Blue"
        await d.dispatch(result("LIGHT_OFF"), 0)
        assert not s.light.state.power
        await d.dispatch(result("TEMPERATURE|18 degrees"), 0)
        assert s.thermostat.setpoint == "18 degrees"
        await d.dispatch(result("CREATE_REMINDER|Study"), 0)
        assert s.reminders.all()[0]["text"] == "Study"
        await d.dispatch(result("TIMER|10 seconds"), 1000)
        assert s.timer.ends_at_ms == 11000
        s.timer.cancel()
    asyncio.run(run())


def test_rejected_does_nothing(tmp_path):
    async def run():
        d, s = make(tmp_path)
        out = await d.dispatch(result("LIGHT_ON", accepted=False), 0)
        assert not out.executed and not s.light.state.power
    asyncio.run(run())


def test_laptop_revalidates_slot(tmp_path):
    async def run():
        d, _ = make(tmp_path)
        r = result("TIMER|10 seconds")
        r.primary.slot = "2 hours"  # malformed edge message
        out = await d.dispatch(r, 0)
        assert not out.executed and "not an allowed" in out.detail
    asyncio.run(run())


def test_list_reminders_sets_focus_and_toast(tmp_path):
    async def run():
        d, s = make(tmp_path)
        await d.dispatch(result("LIST_REMINDERS"), 1000)
        assert s.reminders.focus_until_ms == 1000 + 6000
        assert d.toast is not None
        assert d.toast["kind"] == "reminders"
        assert d.toast["until_ms"] == 1000 + 8000
        assert d.toast["at_ms"] == 1000
    asyncio.run(run())


def test_reminders_persist(tmp_path):
    ReminderStore(tmp_path / "r.sqlite").add("Exercise", "voice")
    assert ReminderStore(tmp_path / "r.sqlite").all()[0]["text"] == "Exercise"


def test_scoreboard_intent_vs_full_accuracy(tmp_path):
    log = CommandLog(tmp_path / "c.sqlite")
    r = result("TIMER|10 seconds", compare=[pred("TIMER", model="intent_model"),
                                            pred("TIMER|30 seconds", model="leaf_b")])
    log.record(r, Outcome(True, "x"), 2)
    log.label(r.command_id, "TIMER|10 seconds")
    board = {row["model_id"]: row for row in log.scoreboard()}
    assert board["m"]["accuracy"] == 1.0
    assert board["intent_model"]["accuracy"] == 0.0 and board["intent_model"]["intent_accuracy"] == 1.0
    assert board["leaf_b"]["accuracy"] == 0.0 and board["leaf_b"]["intent_accuracy"] == 1.0
