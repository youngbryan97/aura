"""A metrics surface that drops samples in silence goes blind quietly.

Eight collectors in `core/observability/metrics.py` caught their source
being unavailable and passed. That is the right BEHAVIOUR — psutil missing,
the substrate not loaded yet, the drive engine absent: each leaves a gap in
the series rather than an error. What was missing is that the gap said
nothing, so a scrape reporting half its metrics looked exactly like a
healthy one on a quiet runtime.

The gap is a metric now.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.observability.metrics import MetricsCollector


def test_a_clean_collect_reports_no_gaps() -> None:
    """The null. A counter that is always there says nothing."""
    collector = MetricsCollector()
    samples = collector.collect()
    assert samples
    assert not any(s.name == "aura_metric_unsampled_total" for s in samples)
    assert collector._unsampled == {}


def test_a_source_that_cannot_be_read_becomes_a_counted_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _absent(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("psutil is not installed here")

    monkeypatch.setattr(
        "core.runtime.service_registry.get_runtime_service", _absent, raising=False
    )
    collector = MetricsCollector()
    samples = collector.collect()
    gaps = [s for s in samples if s.name == "aura_metric_unsampled_total"]
    assert gaps, "a source that could not be read left no trace"
    assert {s.labels["metric"] for s in gaps} & {
        "aura_substrate",
        "aura_will",
        "aura_drive_level",
    }
    for gap in gaps:
        assert gap.metric_type == "counter"
        assert gap.value >= 1.0


def test_the_gap_counts_up_across_scrapes(monkeypatch: pytest.MonkeyPatch) -> None:
    collector = MetricsCollector()
    collector._sample_unavailable("aura_cpu_percent", OSError("gone"))
    collector._sample_unavailable("aura_cpu_percent", OSError("gone"))
    gaps = {
        s.labels["metric"]: s.value
        for s in collector.collect()
        if s.name == "aura_metric_unsampled_total"
    }
    assert gaps["aura_cpu_percent"] == 2.0


def test_it_is_logged_once_not_every_scrape(caplog: pytest.LogCaptureFixture) -> None:
    """A collector runs on every scrape; a line per scrape is a flood."""
    collector = MetricsCollector()
    with caplog.at_level("INFO", logger="Aura.Observability.Metrics"):
        for _ in range(5):
            collector._sample_unavailable("aura_cpu_percent", OSError("gone"))
    said = [r for r in caplog.records if "aura_cpu_percent" in r.getMessage()]
    assert len(said) == 1
    assert collector._unsampled["aura_cpu_percent"] == 5


def test_the_gap_renders_in_prometheus_text() -> None:
    collector = MetricsCollector()
    collector._sample_unavailable("aura_db_size_bytes", OSError("no path"))
    text = collector.render_prometheus()
    assert "aura_metric_unsampled_total" in text
    assert 'metric="aura_db_size_bytes"' in text
