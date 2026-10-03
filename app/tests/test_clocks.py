from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from laptop.app.services.clocks import next_alarm, parse_duration_s
from vcm_common.ontology import get_ontology

TZ = ZoneInfo("Asia/Manila")


def test_every_timer_slot_parses():
    got = [parse_duration_s(v) for v in get_ontology().slot_values["TIMER"]]
    assert got == [10, 30, 60]


def test_bad_duration():
    with pytest.raises(ValueError):
        parse_duration_s("soon")


def test_next_alarm_rolls_over():
    now = datetime(2026, 9, 25, 7, 0, tzinfo=TZ)
    assert next_alarm("6:00 AM", now) == datetime(2026, 9, 26, 6, 0, tzinfo=TZ)
    assert next_alarm("8:00 AM", now) == datetime(2026, 9, 25, 8, 0, tzinfo=TZ)
    assert next_alarm("9:00 PM", now).hour == 21
