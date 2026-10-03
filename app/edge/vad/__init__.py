"""VAD endpointing plugin: stop listening when the user stops talking.

See docs/VAD_PLUGIN_GUIDE.md. Public API:
    EndpointingConfig.from_dict(cfg)  -> config
    build_endpointer(config)          -> (CaptureEndpointer | None, status)
    CaptureEndpointer.start() / push(frame) / done / result() / summary()
"""
from .config import EndpointingConfig, build_endpointer
from .endpointer import EndpointParams, Endpointer, EndpointResult
from .engines import EnergyVAD, SileroVAD, find_silero_model
from .stream import CaptureEndpointer

__all__ = ["EndpointingConfig", "build_endpointer", "EndpointParams", "Endpointer",
           "EndpointResult", "EnergyVAD", "SileroVAD", "find_silero_model", "CaptureEndpointer"]
