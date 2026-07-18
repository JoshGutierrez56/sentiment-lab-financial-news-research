"""NVIDIA telemetry sampled during local inference."""

from __future__ import annotations

import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True)
class GPUSample:
    monotonic_time: float
    utilization_percent: float
    memory_used_mib: float
    power_watts: float


@dataclass(frozen=True)
class GPUTelemetrySummary:
    sample_count: int
    duration_seconds: float
    average_utilization_percent: float
    maximum_utilization_percent: float
    maximum_memory_used_mib: float
    average_power_watts: float
    maximum_power_watts: float
    energy_kwh: float
    electricity_cost_usd: float


class NvidiaTelemetrySampler:
    """Background sampler using stable CSV fields from nvidia-smi."""

    def __init__(self, *, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self._samples: list[GPUSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _sample() -> GPUSample:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        first_gpu = completed.stdout.strip().splitlines()[0]
        utilization, memory, power = (float(item.strip()) for item in first_gpu.split(","))
        return GPUSample(time.monotonic(), utilization, memory, power)

    def _run(self) -> None:
        while not self._stop.is_set():
            with suppress(OSError, subprocess.SubprocessError, ValueError, IndexError):
                self._samples.append(self._sample())
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Telemetry sampler has already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, *, electricity_rate_usd_per_kwh: float) -> GPUTelemetrySummary:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
        samples = list(self._samples)
        if not samples:
            return GPUTelemetrySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        duration = max(samples[-1].monotonic_time - samples[0].monotonic_time, 0.0)
        powers = [item.power_watts for item in samples]
        if len(samples) == 1:
            energy_wh = powers[0] * self.interval_seconds / 3600.0
        else:
            energy_wh = 0.0
            for left, right in pairwise(samples):
                seconds = right.monotonic_time - left.monotonic_time
                energy_wh += ((left.power_watts + right.power_watts) / 2.0) * seconds / 3600.0
        energy_kwh = energy_wh / 1000.0
        return GPUTelemetrySummary(
            sample_count=len(samples),
            duration_seconds=duration,
            average_utilization_percent=sum(item.utilization_percent for item in samples)
            / len(samples),
            maximum_utilization_percent=max(item.utilization_percent for item in samples),
            maximum_memory_used_mib=max(item.memory_used_mib for item in samples),
            average_power_watts=sum(powers) / len(powers),
            maximum_power_watts=max(powers),
            energy_kwh=energy_kwh,
            electricity_cost_usd=energy_kwh * electricity_rate_usd_per_kwh,
        )
