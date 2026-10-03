"""Smart light and air-conditioner simulators.

Their state can only take values the VCM ontology can produce, so the UI can
never show a setting the model could not have commanded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from vcm_common.ontology import Ontology


@dataclass
class LightState:
    power: bool = False
    brightness: str = "100 percent"
    color: Optional[str] = None      # None = warm white (initial, before any COLOR command)


class LightService:
    def __init__(self, ontology: Ontology):
        self.o = ontology
        self.state = LightState(brightness=ontology.slot_values["BRIGHTNESS"][-1])

    def on(self) -> str:
        self.state.power = True
        return "light power on"

    def off(self) -> str:
        self.state.power = False
        return "light power off"

    def brightness(self, value: str) -> str:
        assert value in self.o.slot_values["BRIGHTNESS"]
        self.state.brightness, self.state.power = value, True
        return f"light brightness {value}"

    def color(self, value: str) -> str:
        assert value in self.o.slot_values["COLOR"]
        self.state.color, self.state.power = value, True
        return f"light color {value}"

    def snapshot(self) -> dict:
        d = asdict(self.state)
        d["brightness_pct"] = int(self.state.brightness.split()[0])
        return d


class ThermostatService:
    """Air conditioner: set-point restricted to the ontology values."""

    def __init__(self, ontology: Ontology):
        self.o = ontology
        values = ontology.slot_values["TEMPERATURE"]
        self.setpoint = values[-1]
        self.changed_at_ms: Optional[float] = None

    def set(self, value: str, now_ms: float) -> str:
        assert value in self.o.slot_values["TEMPERATURE"]
        self.setpoint, self.changed_at_ms = value, now_ms
        return f"air-con set-point {value}"

    def snapshot(self) -> dict:
        return {
            "setpoint": self.setpoint,
            "setpoint_c": int(self.setpoint.split()[0]),
            "options_c": [int(v.split()[0]) for v in self.o.slot_values["TEMPERATURE"]],
            "mode": "cool",
            "changed_at_ms": self.changed_at_ms,
        }
