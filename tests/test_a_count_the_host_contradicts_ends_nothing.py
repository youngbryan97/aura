"""A memory count the host contradicts ends neither a worker nor her (26 November 2026).

Her watchdog sums the footprints of her process tree, and a footprint counts memory that is not resident: wired Metal
buffers, weights mapped by more than one process. LIVE 26 November 2026, 12:10, it read 57,805 MB while the host was
56.6% used with 27.8 GB available, terminated her active cortex worker, which came straight back, and ended her; the
boot after did the same within twenty minutes. The tiers that end something act on her count only when the host
agrees it is short. A host that gave no reading leaves the count to act alone, as before.
"""
from __future__ import annotations

import pytest

from core.resilience.memory_watchdog import MemorySample, MemoryWatchdog, _Thresholds

pytestmark = pytest.mark.unit

#: The live ceilings of 26 November 2026 on the 64 GB host.
_LIVE = _Thresholds(soft_mb=47_104.0, hard_mb=53_248.0, lethal_mb=57_180.0, swap_hard_gb=16.0, boot_grace_s=0.0)


def _the_tombstone(*, available_gb: float = 27.77, swap_free_gb: float = 1.42) -> MemorySample:
    """The last sample before she was ended (data/error_logs/memory/oom_tombstone_1795723834.json)."""
    return MemorySample(core_rss_mb=56_876.4, child_rss_mb=3_043.9, swap_used_gb=2.58, swap_free_gb=swap_free_gb,
                        available_gb=available_gb, system_percent=56.6, total_ram_gb=64.0, sampled_at=1795723834.09)


def _watchdog(ended: list[str]) -> MemoryWatchdog:
    dog = MemoryWatchdog(thresholds=_LIVE, lethal_action="exit", sample_interval_s=999.0, sampler=_the_tombstone,
                         worker_terminator=lambda **_: ended.append("worker") or 1, gc_collect=lambda: 0,
                         ladder_shed=lambda: (0, 0), process_exit=lambda code: ended.append(f"exit {code}"))
    dog._schedule_governor_sweep = lambda: None
    return dog


def test_the_count_that_ended_her_ends_nothing_while_the_host_has_room():
    ended: list[str] = []
    dog = _watchdog(ended)
    tiers = [dog._evaluate(_the_tombstone(), now=1_000.0 + 3 * n) for n in range(6)]
    assert ended == [] and all(tier.startswith("soft") for tier in tiers), (tiers, ended)


def test_the_same_count_with_the_host_short_still_acts():
    ended: list[str] = []
    dog = _watchdog(ended)
    tiers = [dog._evaluate(_the_tombstone(available_gb=3.0), now=1_000.0 + 3 * n) for n in range(6)]
    assert "worker" in ended and tiers[0] == "lethal_reclaim", (tiers, ended)


def test_a_host_that_gave_no_reading_leaves_the_count_to_act_alone():
    ended: list[str] = []
    dog = _watchdog(ended)
    assert dog._evaluate(_the_tombstone(available_gb=0.0), now=1_000.0) == "lethal_reclaim" and "worker" in ended
