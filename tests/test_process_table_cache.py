"""The host process-table scan must not run on every observation call.

The Jul 24 boot-stall dumps caught psutil.process_iter running ON the
event loop via resource_observation.processes(); a full host scan costs
seconds of syscalls on macOS, and several subsystems call it per tick.
A short-TTL cache bounds the cost for every caller at once; failures
are never cached so recovery retries immediately.
"""
from __future__ import annotations

from core.runtime.resource_observation import HostResourceObserver


def test_process_table_is_cached_within_ttl(monkeypatch):
    observer = HostResourceObserver()
    calls = {"n": 0}
    real_iter = __import__("psutil").process_iter

    def counting_iter(*args, **kwargs):
        calls["n"] += 1
        return real_iter(*args, **kwargs)

    monkeypatch.setattr("psutil.process_iter", counting_iter)
    first = observer.process_table()
    second = observer.process_table()
    assert calls["n"] == 1, "the second call within the TTL must hit the cache"
    assert second is first

    observer._process_table_cache = (
        observer._process_table_cache[0] - 10.0,
        observer._process_table_cache[1],
    )
    observer.process_table()
    assert calls["n"] == 2, "an expired cache must rescan"


def test_scan_failures_are_not_cached(monkeypatch):
    observer = HostResourceObserver()

    def broken_iter(*args, **kwargs):
        raise OSError("simulated scan failure")

    monkeypatch.setattr("psutil.process_iter", broken_iter)
    failed = observer.process_table()
    assert failed.available is False
    assert observer._process_table_cache is None, (
        "a failed scan must not be served to later callers"
    )


# The Jul 24 pass bounded processes()/process_table(), which reach the host
# through process_iter. memory()'s tree walk reaches it through
# Process.children(recursive=True) — the same host-wide enumeration, and it
# was left uncached. On Jul 29 it froze the loop for 63.5s from
# record_degradation. Same defect, same remedy.


def test_process_tree_walk_is_cached_within_ttl(monkeypatch):
    import psutil

    observer = HostResourceObserver()
    calls = {"n": 0}
    real_children = psutil.Process.children

    def counting_children(self, *args, **kwargs):
        calls["n"] += 1
        return real_children(self, *args, **kwargs)

    monkeypatch.setattr(psutil.Process, "children", counting_children)

    first = observer.memory()
    second = observer.memory()
    assert calls["n"] == 1, "the second tree observation within the TTL must hit the cache"
    assert second.process_tree_rss_bytes >= second.process_rss_bytes

    root = next(iter(observer._tree_children_rss_cache))
    stamp, total = observer._tree_children_rss_cache[root]
    observer._tree_children_rss_cache[root] = (stamp - 10.0, total)
    observer.memory()
    assert calls["n"] == 2, "an expired cache must rescan"

    # Own RSS is cheap and must stay live rather than riding the cache.
    assert first.process_rss_bytes > 0


def test_own_rss_never_pays_for_the_tree_walk(monkeypatch):
    import psutil

    observer = HostResourceObserver()

    def forbidden(*args, **kwargs):  # pragma: no cover - only runs on regression
        raise AssertionError("include_process_tree=False must not enumerate the host")

    # Armed for the measurement alone: the fixtures that close the test down
    # list processes, and with the tripwire still armed the test errored.
    with monkeypatch.context() as tripwire:
        tripwire.setattr(psutil.Process, "children", forbidden)
        tripwire.setattr(psutil, "pids", forbidden)
        own = observer.memory(include_process_tree=False)
    assert own.process_rss_bytes > 0


def test_tree_walk_failures_are_not_cached(monkeypatch):
    import psutil

    observer = HostResourceObserver()

    def broken_children(self, *args, **kwargs):
        raise OSError("simulated tree walk failure")

    with monkeypatch.context() as broken:
        broken.setattr(psutil.Process, "children", broken_children)
        observer.memory()
    assert not observer._tree_children_rss_cache, (
        "a failed tree walk must not be served to later callers"
    )
