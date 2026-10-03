"""Build an EdgePipeline from config/demo.yaml (shared by server and local mode)."""
from __future__ import annotations

from pathlib import Path

from .chime import EdgeChime
from .media import build_media
from .pipeline import EdgePipeline, EdgeSettings, Emit
from .registry import ModelRegistry
from .vad import EndpointingConfig, build_endpointer
from .wake import build_wake_engines


def build_pipeline(cfg: dict, repo_root: Path, emit: Emit) -> EdgePipeline:
    ecfg = cfg["edge_service"]

    def resolve(p: str | None) -> str | None:
        if p is None:
            return None
        path = Path(p).expanduser()
        return str(path if path.is_absolute() else (repo_root / path).resolve())

    registry = ModelRegistry(
        models_dir=Path(resolve(ecfg.get("models_dir", "models"))),
        engine_settings={"tiny_vcm_root": resolve(ecfg.get("tiny_vcm_root"))},
    )
    registry.scan()
    media_cfg = dict(ecfg.get("media", {}))
    media_cfg["folder"] = resolve(media_cfg.get("folder"))
    wake = build_wake_engines(ecfg.get("wake_engines", []), resolve)
    default_wake = ecfg.get("default_wake", "manual")
    if default_wake not in wake or wake[default_wake].info.status != "ready":
        default_wake = "manual"   # stock/custom wake word missing -> button still works
    benchmark_mode = bool(ecfg.get("benchmark_mode", False))
    settings = EdgeSettings(
        window_s=float(ecfg.get("window_s", 4.0)),
        active_vcm=ecfg.get("active_vcm"),
        compare_ids=list(ecfg.get("compare_ids", [])),
        active_wake=default_wake,
        result_hold_s=float(ecfg.get("result_hold_s", 1.5)),
        pre_capture_delay_s=float(ecfg.get("pre_capture_delay_s", 0.35)),
        benchmark_mode=benchmark_mode,
    )
    if benchmark_mode:
        media_cfg["backend"] = "simulated"   # benchmark: music off
    chime_cfg = ecfg.get("chime", {}) or {}
    chime = EdgeChime(bool(chime_cfg.get("play_on_edge", False)) and not benchmark_mode,
                      chime_cfg.get("player", "aplay"))
    ep_cfg = dict(ecfg.get("endpointing", {}) or {})
    if ep_cfg.get("model_path"):
        ep_cfg["model_path"] = resolve(ep_cfg["model_path"])
    ep_cfg["search_roots"] = [resolve(r) for r in ep_cfg.get("search_roots", [])]
    endpointer, ep_status = build_endpointer(EndpointingConfig.from_dict(ep_cfg))
    return EdgePipeline(registry, wake, build_media(media_cfg), settings, emit,
                        device_label=ecfg.get("device_label", "edge"), chime=chime,
                        endpointer=endpointer, endpointing_status=ep_status)
