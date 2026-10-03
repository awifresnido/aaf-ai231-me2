import pytest

from edge.media import MediaController, SimulatedBackend


class SpyBackend(SimulatedBackend):
    def __init__(self):
        self.volumes: list[int] = []

    def set_volume_percent(self, pct: int) -> None:
        self.volumes.append(pct)


def playing(level=4):
    mc = MediaController(backend=SpyBackend(), level=level)
    mc.play()
    return mc


def test_duck_only_when_playing():
    mc = MediaController(backend=SpyBackend())
    assert mc.duck() is False
    mc = playing()
    assert mc.duck() is True
    assert mc.backend.volumes[-1] == 10 and mc.level == 4


@pytest.mark.parametrize("intent,status,level", [
    (None, "playing", 4), ("LIGHT_ON", "playing", 4), ("VOLUME_UP", "playing", 5),
    ("VOLUME_DOWN", "playing", 3), ("PAUSE", "paused", 4), ("STOP", "stopped", 4),
    ("NEXT", "playing", 4), ("PLAY_MUSIC", "playing", 4),
])
def test_restore_rules(intent, status, level):
    mc = playing(4)
    mc.duck()
    mc.apply(intent)
    assert (mc.status, mc.level, mc.ducked) == (status, level, False)
    assert mc.backend.volumes[-1] == mc.volume_steps_pct[level - 1]  # never left ducked


def test_volume_bounds():
    mc = playing(5)
    mc.apply("VOLUME_UP")
    assert mc.level == 5
    mc = playing(1)
    mc.apply("VOLUME_DOWN")
    assert mc.level == 1


def test_next_advances_track():
    mc = playing()
    idx = mc.track_index
    mc.apply("NEXT")
    assert mc.track_index == idx + 1
