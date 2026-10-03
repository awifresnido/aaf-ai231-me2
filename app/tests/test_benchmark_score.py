"""score_trial: every scored outcome."""
from laptop.app.benchmark.runner import score_trial


def test_command_correct():
    assert score_trial("command", "LIGHT_ON", True, "LIGHT_ON") == ("correct", True)


def test_command_not_understood_below_tau():
    assert score_trial("command", "LIGHT_ON", False, "UNKNOWN") == ("not_understood", False)


def test_command_wrong():
    assert score_trial("command", "LIGHT_ON", True, "LIGHT_OFF") == ("wrong", False)


def test_command_wrong_intent_ok_slot_wrong():
    assert score_trial("command", "BRIGHTNESS|60 percent", True,
                       "BRIGHTNESS|20 percent") == ("wrong", True)


def test_negative_correct_reject():
    assert score_trial("negative", None, False, "UNKNOWN") == ("correct_reject", False)


def test_negative_false_accept():
    assert score_trial("negative", None, True, "TIME") == ("false_accept", False)
