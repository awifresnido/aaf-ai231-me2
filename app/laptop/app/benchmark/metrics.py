"""Pooled benchmark metrics: Wilson intervals, per-session rows, heatmap, chips."""
from __future__ import annotations

import math
from typing import Optional

from ..clock_sync import timing_trusted

VCM_OUTCOMES = ("correct", "not_understood", "wrong")
NEG_OUTCOMES = ("false_accept", "correct_reject")
WAKE_OR_RESULT = VCM_OUTCOMES + NEG_OUTCOMES + ("wake_miss", "no_result")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (95%). n == 0 -> (0.0, 0.0)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def pctile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    i = (len(s) - 1) * q
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def trial_color(outcome: Optional[str], intent_correct: int = 0) -> str:
    """green / amber / red / grey per the heatmap legend."""
    if outcome in ("correct", "correct_reject"):
        return "green"
    if outcome == "not_understood":
        return "amber"
    if outcome == "wrong":
        return "amber" if intent_correct else "red"
    if outcome == "false_accept":
        return "red"
    return "grey"


def effective(trials: list[dict]) -> list[dict]:
    """Drop superseded, misspoken, skipped, and abandoned-session trials."""
    return [t for t in trials
            if not t.get("superseded") and not t.get("misspoken")
            and t.get("outcome") != "skipped" and t.get("status") != "abandoned"]


def _rate(trials: list[dict], key: str) -> dict:
    n = len(trials)
    k = sum(1 for t in trials if t["outcome"] == key)
    lo, hi = wilson(k, n)
    return {"n": n, "k": k, "value": (k / n) if n else None, "lo": lo, "hi": hi}


def _ms(trials: list[dict], field: str) -> dict:
    vals = [t[field] for t in trials if t.get(field) is not None]
    return {"n": len(vals), "p50": pctile(vals, 0.5), "p95": pctile(vals, 0.95)}

def _e2e_ms(trials: list[dict]) -> dict:
    """End-to-end = capture_to_result + network + result_to_action, only for trials
    timed under a settled, accurate clock (NTP-style or local; see clock_sync)."""
    vals = [t["capture_to_result_ms"] + t["network_ms"] + t["result_to_action_ms"]
            for t in trials
            if timing_trusted(t.get("timing_valid"), t.get("clock_quality"))
            and t.get("capture_to_result_ms") is not None
            and t.get("network_ms") is not None and t.get("result_to_action_ms") is not None]
    return {"n": len(vals), "p50": pctile(vals, 0.5), "p95": pctile(vals, 0.95)}


def _intent_of(expected: Optional[str]) -> Optional[str]:
    return expected.split("|")[0] if expected else None


def _aggregate_color(cells: list[str]) -> str:
    if not cells:
        return "grey"
    if "red" in cells:
        return "red"
    if "amber" in cells:
        return "amber"
    return "green"


def heatmap(trials: list[dict], fixed_prompt_labels: dict[int, str]) -> dict:
    """participants x prompts. Fixed sessions use prompt labels; random sessions
    aggregate per intent."""
    eff = effective(trials)
    sessions: dict[str, dict] = {}
    for t in eff:
        s = sessions.setdefault(t["session_id"], {"alias": t["alias"], "set_type": t["set_type"],
                                                  "cells": {}})
        if t["set_type"] == "fixed":
            col = fixed_prompt_labels.get(t["idx"], f"#{t['idx']}")
        else:
            col = _intent_of(t["expected_class"]) or "—"
        c = trial_color(t["outcome"], t.get("intent_correct") or 0)
        existing = s["cells"].get(col)
        s["cells"][col] = _aggregate_color([existing, c]) if existing else c
    columns = sorted({col for s in sessions.values() for col in s["cells"]})
    # keep fixed labels in prompt order first, then random intents
    fixed_cols = [l for _, l in sorted(fixed_prompt_labels.items())]
    order = fixed_cols + [c for c in columns if c not in fixed_cols]
    rows = [{"session_id": sid, "alias": s["alias"], "set_type": s["set_type"],
             "cells": [{"col": c, "color": s["cells"].get(c, "grey")} for c in order]}
            for sid, s in sessions.items()]
    return {"columns": order, "rows": rows}


def per_session(trials: list[dict]) -> list[dict]:
    eff = effective(trials)
    out: dict[str, dict] = {}
    for t in eff:
        s = out.setdefault(t["session_id"], {
            "session_id": t["session_id"], "alias": t["alias"], "set_type": t["set_type"],
            "model_id": t["model_id"], "cond_noise": t["cond_noise"],
            "cond_distance": t["cond_distance"], "created_ms": t["created_ms"],
            "scored": 0, "correct": 0, "wrong": 0, "neg_ignored": 0, "neg_total": 0,
            "wake_miss": 0, "inference_ms": [], "e2e_ms": []})
        s = out[t["session_id"]]
        o = t["outcome"]
        if t["kind"] == "command" and o in VCM_OUTCOMES:
            s["scored"] += 1
            if o == "correct":
                s["correct"] += 1
            elif o == "wrong":
                s["wrong"] += 1
        elif t["kind"] == "negative" and o in NEG_OUTCOMES:
            s["neg_total"] += 1
            if o == "correct_reject":
                s["neg_ignored"] += 1
        if o == "wake_miss":
            s["wake_miss"] += 1
        if t.get("inference_ms") is not None:
            s["inference_ms"].append(t["inference_ms"])
        if (timing_trusted(t.get("timing_valid"), t.get("clock_quality"))
                and t.get("capture_to_result_ms") is not None
                and t.get("network_ms") is not None and t.get("result_to_action_ms") is not None):
            s["e2e_ms"].append(t["capture_to_result_ms"] + t["network_ms"] + t["result_to_action_ms"])
    rows = []
    for s in out.values():
        rows.append({
            "session_id": s["session_id"], "alias": s["alias"], "set_type": s["set_type"],
            "model_id": s["model_id"], "cond_noise": s["cond_noise"],
            "cond_distance": s["cond_distance"], "created_ms": s["created_ms"],
            "scored": s["scored"],
            "correct_rate": (s["correct"] / s["scored"]) if s["scored"] else None,
            "wrong_rate": (s["wrong"] / s["scored"]) if s["scored"] else None,
            "neg_ignored": s["neg_ignored"], "neg_total": s["neg_total"],
            "wake_miss": s["wake_miss"],
            "model_median_ms": pctile(s["inference_ms"], 0.5),
            "e2e_median_ms": pctile(s["e2e_ms"], 0.5),
        })
    return sorted(rows, key=lambda r: r["created_ms"])


def chips(pooled: dict, targets: dict, model_facts: dict) -> list[dict]:
    def rate_chip(key: str, label: str, target_key: str, higher_is_better: bool) -> dict:
        m = pooled[key]
        n = m["n"]
        status = "too_few" if n < 20 else ("pass" if _meets(m["value"], targets[target_key],
                                                            higher_is_better) else "fail")
        return {"key": key, "label": label, "value": m["value"], "lo": m["lo"], "hi": m["hi"],
                "n": n, "target": targets[target_key], "status": status}

    def latency_chip(key: str, label: str, target_key: str) -> dict:
        m = pooled[key]
        n = m["n"]
        status = "too_few" if n < 20 else ("pass" if (m["p95"] or 0) <= targets[target_key] else "fail")
        return {"key": key, "label": label, "value": m["p95"], "lo": m["p50"], "hi": None,
                "n": n, "target": targets[target_key], "status": status}

    def static_chip(key: str, label: str, target_key: str) -> dict:
        v = model_facts.get(key)
        status = "pass" if (v is not None and v <= targets[target_key]) else "fail"
        return {"key": key, "label": label, "value": v, "lo": None, "hi": None, "n": None,
                "target": targets[target_key], "status": status}

    return [
        rate_chip("correct", "correct", "correct_min", True),
        rate_chip("wrong", "wrong", "wrong_max", False),
        rate_chip("false_accept", "false accept", "false_accept_max", False),
        rate_chip("wake_miss", "wake miss", "wake_miss_max", False),
        latency_chip("pi_model_ms", "Pi model p95", "pi_model_p95_ms_max"),
        latency_chip("pi_features_ms", "Pi features p95", "pi_features_p95_ms_max"),
        static_chip("params", "params", "params_max"),
        static_chip("size_kb", "model size", "model_size_kb_max"),
    ]


def _meets(value: Optional[float], target: float, higher_is_better: bool) -> bool:
    if value is None:
        return False
    return value >= target if higher_is_better else value <= target


def compute(trials: list[dict], targets: dict, model_facts: dict,
            false_wakes_total: int, fixed_prompt_labels: dict[int, str]) -> dict:
    """Return {pooled, sessions, heatmap, chips} over the filtered trials."""
    eff = effective(trials)
    command = [t for t in eff if t["kind"] == "command" and t["outcome"] in VCM_OUTCOMES]
    negative = [t for t in eff if t["kind"] == "negative" and t["outcome"] in NEG_OUTCOMES]
    prompted = [t for t in eff if t["outcome"] in WAKE_OR_RESULT]
    scored_with_ms = [t for t in eff if t["outcome"] in VCM_OUTCOMES + NEG_OUTCOMES]

    pooled = {
        "correct": _rate(command, "correct"),
        "not_understood": _rate(command, "not_understood"),
        "wrong": _rate(command, "wrong"),
        "false_accept": _rate(negative, "false_accept"),
        "wake_miss": _rate(prompted, "wake_miss"),
        "no_result": sum(1 for t in eff if t["outcome"] == "no_result"),
        "false_wakes": false_wakes_total,
        "pi_model_ms": _ms(scored_with_ms, "inference_ms"),
        "pi_features_ms": _ms(scored_with_ms, "feature_ms"),
        "e2e_ms": _e2e_ms(scored_with_ms),
        "model": model_facts,
    }

    return {
        "pooled": pooled,
        "sessions": per_session(trials),
        "heatmap": heatmap(trials, fixed_prompt_labels),
        "chips": chips(pooled, targets, model_facts),
    }
