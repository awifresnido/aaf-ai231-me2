"""CPU / memory readings. Uses psutil when present; otherwise /proc on Linux."""
from __future__ import annotations

import os
from typing import Optional

try:
    import psutil  # type: ignore
except ImportError:  # optional dependency
    psutil = None

_last_cpu: Optional[tuple[int, int]] = None


def cpu_percent() -> Optional[float]:
    global _last_cpu
    if psutil is not None:
        return float(psutil.cpu_percent(interval=None))
    try:
        with open("/proc/stat") as fh:
            parts = [int(x) for x in fh.readline().split()[1:]]
    except OSError:
        return None
    idle, total = parts[3] + parts[4], sum(parts)
    if _last_cpu is None:
        _last_cpu = (idle, total)
        return None
    d_idle, d_total = idle - _last_cpu[0], total - _last_cpu[1]
    _last_cpu = (idle, total)
    return round(100.0 * (1 - d_idle / d_total), 1) if d_total > 0 else None


def process_rss_mb() -> Optional[float]:
    if psutil is not None:
        return round(psutil.Process(os.getpid()).memory_info().rss / 2**20, 1)
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        return None
    return None


def system_mem_used_mb() -> Optional[float]:
    if psutil is not None:
        vm = psutil.virtual_memory()
        return round((vm.total - vm.available) / 2**20, 1)
    try:
        info = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0])
        return round((info["MemTotal"] - info["MemAvailable"]) / 1024, 1)
    except (OSError, KeyError):
        return None
