from core.memory.cold_store import ColdMemoryStore


def test_cold_store_persists_and_recalls_across_instances(tmp_path):
    path = tmp_path / "cold.db"
    first = ColdMemoryStore(path)

    assert first.add_memory(
        "Bryan's favorite animal is the orca.",
        {"kind": "preference", "owner": "Bryan"},
    )
    assert first.count() == 1

    second = ColdMemoryStore(path)
    results = second.search("favorite orca", limit=3)

    assert second.is_ready() is True
    assert len(results) == 1
    assert results[0]["content"].endswith("orca.")
    assert results[0]["metadata"]["kind"] == "preference"


def test_cold_store_rejects_empty_and_bounds_query_count(tmp_path):
    store = ColdMemoryStore(tmp_path / "cold.db")

    assert store.add_memory("", {}) is False
    for index in range(3):
        assert store.add_memory(f"bounded archival record {index}", {"i": index})

    assert len(store.search("archival", limit=2)) == 2


def test_cold_store_count_is_a_cached_health_read_not_sqlite_io(tmp_path, monkeypatch):
    store = ColdMemoryStore(tmp_path / "cold.db")
    assert store.add_memory("one") is True

    def database_access_would_block():
        raise AssertionError("count reopened SQLite")

    monkeypatch.setattr(store, "_connect", database_access_would_block)

    assert store.count() == 1
    assert store.health_item_count == 1


def test_cold_store_reloads_cached_count_from_durable_state(tmp_path):
    path = tmp_path / "cold.db"
    first = ColdMemoryStore(path)
    assert first.add_memory("one") is True
    assert first.add_memory("two") is True

    reopened = ColdMemoryStore(path)

    assert reopened.count() == 2
    assert reopened.health_item_count == 2
