from __future__ import annotations

import pytest

from sentiment_lab.hybrid.hardware import GPUSample, NvidiaTelemetrySampler


def test_gpu_telemetry_integrates_power_and_cost() -> None:
    sampler = NvidiaTelemetrySampler(interval_seconds=1.0)
    sampler._samples = [
        GPUSample(1.0, 50.0, 10_000.0, 100.0),
        GPUSample(3.0, 100.0, 20_000.0, 300.0),
    ]
    summary = sampler.stop(electricity_rate_usd_per_kwh=0.25)
    assert summary.sample_count == 2
    assert summary.maximum_memory_used_mib == 20_000.0
    assert summary.energy_kwh == pytest.approx((200.0 * 2.0 / 3600.0) / 1000.0)
    assert summary.electricity_cost_usd == pytest.approx(summary.energy_kwh * 0.25)


def test_empty_and_single_sample_telemetry() -> None:
    empty = NvidiaTelemetrySampler().stop(electricity_rate_usd_per_kwh=0.25)
    assert empty.sample_count == 0
    sampler = NvidiaTelemetrySampler(interval_seconds=2.0)
    sampler._samples = [GPUSample(1.0, 50.0, 1000.0, 90.0)]
    single = sampler.stop(electricity_rate_usd_per_kwh=0.25)
    assert single.sample_count == 1
    assert single.energy_kwh > 0


def test_sampler_cannot_start_twice() -> None:
    sampler = NvidiaTelemetrySampler(interval_seconds=0.01)
    sampler.start()
    with pytest.raises(RuntimeError, match="already started"):
        sampler.start()
    sampler.stop(electricity_rate_usd_per_kwh=0.25)
