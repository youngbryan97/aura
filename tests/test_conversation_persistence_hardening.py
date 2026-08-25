from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import sys
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.conversation import persistence as persistence_module
from core.conversation.persistence import (
    ConversationPersistence,
    ConversationRevisionConflictError,
)
from core.runtime.sqlite_support import connecting


def _install_event_bus(monkeypatch, bus):
    module = types.ModuleType("core.event_bus")
    module.get_event_bus = lambda: bus
    monkeypatch.setitem(sys.modules, "core.event_bus", module)


def test_conversation_persistence_records_turn_and_publishes_threadsafe(monkeypatch, tmp_path):
    published: list[tuple[str, dict[str, object]]] = []

    class Bus:
        def publish_threadsafe(self, topic, payload):
            published.append((topic, payload))

    _install_event_bus(monkeypatch, Bus())

    store = ConversationPersistence(tmp_path / "conversations.db")
    session_id = store.start_session({"non_json": object()})
    turn_id = store.record_turn(
        "user\x00",
        "hello from persistence",
        origin="text",
        cid="cid-123",
    )

    history = store.get_session_history(session_id, limit="10000")

    assert turn_id
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hello from persistence"
    assert len(history) == 1
    assert published[0][0] == "turn_recorded"
    assert published[0][1]["session_id"] == session_id
    assert published[0][1]["turn_id"] == turn_id
    assert published[0][1]["content_chars"] == len("hello from persistence")


def test_conversation_persistence_records_exchange_atomically(monkeypatch, tmp_path):
    published: list[tuple[str, dict[str, object]]] = []

    class Bus:
        def publish_threadsafe(self, topic, payload):
            published.append((topic, payload))

    _install_event_bus(monkeypatch, Bus())

    store = ConversationPersistence(tmp_path / "conversations.db")
    session_id = store.start_session()
    turn_ids = store.record_exchange(
        "Remember the live desktop path.",
        "I will carry it across restart.",
        origin="desktop_ui",
        cid="exchange-42",
    )

    history = store.get_session_history(session_id)

    assert len(turn_ids) == 2
    assert [row["role"] for row in history] == ["user", "aura"]
    assert [row["cid"] for row in history] == [
        "exchange-42:user",
        "exchange-42:aura",
    ]
    assert len(published) == 2


def test_completed_exchange_metadata_survives_a_new_persistence_instance(tmp_path):
    db_path = tmp_path / "conversation-evidence.db"
    first = ConversationPersistence(db_path)
    session_id = first.start_session()
    episode = {
        "objective": "Open MissingApp.",
        "capability": "desktop_task",
        "status": "desktop_objective_failed",
        "succeeded": False,
        "failure_detail": "No installed application matches 'MissingApp'",
        "authority_kind": "governed_action_episode",
        "authority_proven": True,
        "authority_reason": "governed_executor_reported_failure",
    }
    first.record_exchange(
        "Open MissingApp.",
        "The application was not found.",
        cid="durable-action",
        session_id=session_id,
        exchange_metadata={"action_episode": episode},
    )

    history = ConversationPersistence(db_path).get_session_history(session_id)

    assert history[0]["metadata"] == {}
    assert history[1]["metadata"]["action_episode"] == episode


def test_existing_conversation_schema_gains_bounded_turn_metadata(tmp_path):
    db_path = tmp_path / "legacy-conversation.db"
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                started_at REAL NOT NULL,
                last_active REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            );
            CREATE TABLE turns (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                origin TEXT,
                created_at REAL NOT NULL,
                cid TEXT
            );
            """
        )
    store = ConversationPersistence(db_path)
    session_id = store.start_session()
    store.record_turn(
        "aura",
        "A verified state read.",
        session_id=session_id,
        metadata={"authority_proven": True},
    )

    with sqlite3.connect(db_path) as con:
        columns = {row[1] for row in con.execute("PRAGMA table_info(turns)")}
    assert "metadata_json" in columns
    assert store.get_session_history(session_id)[0]["metadata"] == {
        "authority_proven": True
    }


def test_completed_exchange_atomically_enqueues_durable_memory_log(tmp_path):
    store = ConversationPersistence(tmp_path / "memory-outbox.db")
    session_id = store.start_session()

    store.record_exchange(
        "Durably learn this turn",
        "This answer must survive worker restart.",
        origin="desktop_ui",
        cid="outbox-1",
        session_id=session_id,
        principal_id="bryan",
        principal_surface="owner",
        enqueue_memory_log=True,
    )

    assert store.memory_log_outbox_status() == {
        "pending": 1,
        "processing": 0,
        "completed": 0,
        "rejected": 0,
        "failed": 0,
    }
    claimed = store.claim_memory_log_batch(limit=4)
    assert len(claimed) == 1
    assert claimed[0]["operation_id"] == f"{session_id}:outbox-1:r1"
    assert claimed[0]["user_content"] == "Durably learn this turn"
    assert claimed[0]["aura_content"] == "This answer must survive worker restart."
    assert claimed[0]["principal_id"] == "bryan"
    assert claimed[0]["principal_surface"] == "owner"
    assert claimed[0]["attempts"] == 1

    assert store.settle_memory_log_item(
        claimed[0]["operation_id"],
        outcome="completed",
    ) == "completed"
    assert store.memory_log_outbox_status()["completed"] == 1


def test_memory_log_outbox_reclaims_expired_lease_and_bounds_poison_retry(tmp_path):
    store = ConversationPersistence(tmp_path / "memory-outbox-retry.db")
    session_id = store.start_session()
    store.record_exchange(
        "Retry this turn",
        "The durable outbox owns it.",
        cid="outbox-retry",
        session_id=session_id,
        enqueue_memory_log=True,
    )
    operation_id = f"{session_id}:outbox-retry:r1"
    first = store.claim_memory_log_batch(limit=1)[0]
    assert first["operation_id"] == operation_id

    with connecting(sqlite3.connect(tmp_path / "memory-outbox-retry.db")) as con:
        con.execute(
            "UPDATE conversation_memory_outbox SET claimed_at = ? "
            "WHERE operation_id = ?",
            (0.0, operation_id),
        )
        con.commit()

    reclaimed = store.claim_memory_log_batch(limit=1, lease_s=1.0)[0]
    assert reclaimed["operation_id"] == operation_id
    assert reclaimed["attempts"] == 2
    for _ in range(7):
        state = store.settle_memory_log_item(
            operation_id,
            outcome="retry",
            error="transient memory service failure",
        )
        if state == "failed":
            break
        assert state == "pending"
        store.claim_memory_log_batch(limit=1)

    assert state == "failed"
    assert store.memory_log_outbox_status()["failed"] == 1


def test_memory_log_outbox_persists_individual_effect_stages(tmp_path):
    store = ConversationPersistence(tmp_path / "memory-outbox-stages.db")
    session_id = store.start_session()
    store.record_exchange(
        "Stage this turn",
        "Each completed effect survives a worker retry.",
        cid="outbox-stages",
        session_id=session_id,
        enqueue_memory_log=True,
    )
    operation_id = f"{session_id}:outbox-stages:r1"
    first = store.claim_memory_log_batch(limit=1)[0]
    assert first["episodic_logged"] == 0
    assert first["experience_recorded"] == 0
    assert first["consciousness_updated"] == 0

    assert store.mark_memory_log_stage(operation_id, stage="episodic") is True
    assert store.mark_memory_log_stage(operation_id, stage="experience") is True
    assert store.mark_memory_log_stage(operation_id, stage="experience") is False

    with connecting(sqlite3.connect(tmp_path / "memory-outbox-stages.db")) as con:
        con.execute(
            "UPDATE conversation_memory_outbox SET claimed_at = 0 WHERE operation_id = ?",
            (operation_id,),
        )
        con.commit()
    replay = store.claim_memory_log_batch(limit=1, lease_s=1.0)[0]
    assert replay["episodic_logged"] == 1
    assert replay["experience_recorded"] == 1
    assert replay["consciousness_updated"] == 0


def test_memory_log_outbox_migrates_stage_columns_for_existing_database(tmp_path):
    db_path = tmp_path / "memory-outbox-migration.db"
    ConversationPersistence(db_path)
    with connecting(sqlite3.connect(db_path)) as con:
        con.execute("ALTER TABLE conversation_memory_outbox DROP COLUMN episodic_logged")
        con.execute("ALTER TABLE conversation_memory_outbox DROP COLUMN experience_recorded")
        con.execute("ALTER TABLE conversation_memory_outbox DROP COLUMN consciousness_updated")
        con.commit()

    store = ConversationPersistence(db_path)
    session_id = store.start_session()
    store.record_exchange(
        "Migrate existing durable state",
        "The outbox now checkpoints each effect stage.",
        cid="migrated-stage-columns",
        session_id=session_id,
        enqueue_memory_log=True,
    )

    claimed = store.claim_memory_log_batch(limit=1)[0]
    assert claimed["episodic_logged"] == 0
    assert claimed["experience_recorded"] == 0
    assert claimed["consciousness_updated"] == 0


def test_regenerated_revision_gets_distinct_memory_log_identity(tmp_path):
    store = ConversationPersistence(tmp_path / "memory-outbox-revision.db")
    session_id = store.start_session()
    store.record_exchange(
        "Revise the answer",
        "Original answer",
        cid="outbox-regen",
        session_id=session_id,
        enqueue_memory_log=True,
    )
    store.replace_aura_turn(
        exchange_id="outbox-regen",
        session_id=session_id,
        replacement_content="Replacement answer",
        expected_revision=1,
        expected_content_sha256=hashlib.sha256(b"Original answer").hexdigest(),
    )

    claimed = store.claim_memory_log_batch(limit=4)
    assert [item["revision"] for item in claimed] == [1, 2]
    assert [item["aura_content"] for item in claimed] == [
        "Original answer",
        "Replacement answer",
    ]
    assert [item["operation_id"] for item in claimed] == [
        f"{session_id}:outbox-regen:r1",
        f"{session_id}:outbox-regen:r2",
    ]


def test_conversation_regeneration_is_atomic_and_append_only(monkeypatch, tmp_path):
    published: list[tuple[str, dict[str, object]]] = []

    class Bus:
        def publish_threadsafe(self, topic, payload):
            published.append((topic, payload))

    _install_event_bus(monkeypatch, Bus())
    db_path = tmp_path / "regeneration.db"
    store = ConversationPersistence(db_path)
    session_id = store.start_session()
    store.record_exchange(
        "Explain the result.",
        "Original answer",
        origin="desktop_ui",
        cid="regen-42",
        session_id=session_id,
    )
    original_sha = hashlib.sha256(b"Original answer").hexdigest()

    receipt = store.replace_aura_turn(
        exchange_id="regen-42",
        session_id=session_id,
        replacement_content="Stronger replacement",
        expected_revision=1,
        expected_content_sha256=original_sha,
    )

    assert receipt["applied"] is True
    assert receipt["previous_revision"] == 1
    assert receipt["revision"] == 2
    history = store.get_session_history(session_id)
    aura = next(row for row in history if row["role"] == "aura")
    assert aura["content"] == "Stronger replacement"
    assert aura["revision"] == 2
    assert aura["content_sha256"] == hashlib.sha256(
        b"Stronger replacement"
    ).hexdigest()

    with connecting(sqlite3.connect(db_path)) as con:
        revisions = con.execute(
            "SELECT revision, content, previous_content_sha256, content_sha256, "
            "origin, actor_principal_id, actor_principal_surface "
            "FROM turn_revisions ORDER BY revision"
        ).fetchall()
    assert revisions == [
        (1, "Original answer", "", original_sha, "initial_delivery", "", ""),
        (
            2,
            "Stronger replacement",
            original_sha,
            hashlib.sha256(b"Stronger replacement").hexdigest(),
            "regenerate",
            "",
            "",
        ),
    ]
    event = next(payload for topic, payload in published if topic == "turn_regenerated")
    assert event["exchange_id"] == "regen-42"
    assert event["previous_revision"] == 1
    assert event["revision"] == 2
    assert event["previous_content_sha256"] == original_sha
    assert event["content_sha256"] == hashlib.sha256(
        b"Stronger replacement"
    ).hexdigest()


def test_conversation_regeneration_rejects_stale_revision_without_mutation(tmp_path):
    store = ConversationPersistence(tmp_path / "stale-regeneration.db")
    session_id = store.start_session()
    store.record_exchange(
        "Question",
        "Original",
        cid="regen-stale",
        session_id=session_id,
    )
    original_sha = hashlib.sha256(b"Original").hexdigest()
    store.replace_aura_turn(
        exchange_id="regen-stale",
        session_id=session_id,
        replacement_content="First replacement",
        expected_revision=1,
        expected_content_sha256=original_sha,
    )

    with pytest.raises(ConversationRevisionConflictError, match="source revision changed"):
        store.replace_aura_turn(
            exchange_id="regen-stale",
            session_id=session_id,
            replacement_content="Stale overwrite",
            expected_revision=1,
            expected_content_sha256=original_sha,
        )

    aura = next(
        row
        for row in store.get_session_history(session_id)
        if row["role"] == "aura"
    )
    assert aura["content"] == "First replacement"
    assert aura["revision"] == 2


def test_conversation_regeneration_enforces_principal_binding(tmp_path):
    store = ConversationPersistence(tmp_path / "principal-regeneration.db")
    session_id = "paired-regeneration"
    store.record_exchange(
        "Private question",
        "Private answer",
        cid="regen-private",
        session_id=session_id,
        principal_id="paired-device:a",
        principal_surface="paired_device",
    )

    with pytest.raises(PermissionError, match="principal mismatch"):
        store.replace_aura_turn(
            exchange_id="regen-private",
            session_id=session_id,
            replacement_content="Unauthorized replacement",
            expected_revision=1,
            expected_content_sha256=hashlib.sha256(b"Private answer").hexdigest(),
            principal_id="paired-device:b",
            principal_surface="paired_device",
        )

    aura = next(
        row
        for row in store.get_session_history(
            session_id,
            principal_id="paired-device:a",
            principal_surface="paired_device",
        )
        if row["role"] == "aura"
    )
    assert aura["content"] == "Private answer"


def test_conversation_persistence_deduplicates_turn_by_cid(monkeypatch, tmp_path):
    published: list[tuple[str, dict[str, object]]] = []

    class Bus:
        def publish_threadsafe(self, topic, payload):
            published.append((topic, payload))

    _install_event_bus(monkeypatch, Bus())

    store = ConversationPersistence(tmp_path / "conversations.db")
    session_id = store.start_session()
    first_id = store.record_turn("user", "same live prompt", origin="desktop_ui", cid="live-1:user")
    second_id = store.record_turn("user", "same live prompt", origin="desktop_ui", cid="live-1:user")

    history = store.get_session_history(session_id)

    assert second_id == first_id
    assert len(history) == 1
    assert history[0]["cid"] == "live-1:user"
    assert len(published) == 1


def test_conversation_persistence_scopes_cid_idempotency_to_session(tmp_path):
    store = ConversationPersistence(tmp_path / "session-scoped-cids.db")
    first_session = store.start_session()
    second_session = store.start_session()

    first_id = store.record_turn(
        "user",
        "first session",
        cid="shared-cid:user",
        session_id=first_session,
    )
    second_id = store.record_turn(
        "user",
        "second session",
        cid="shared-cid:user",
        session_id=second_session,
    )

    assert first_id != second_id
    assert store.get_session_history(first_session)[0]["content"] == "first session"
    assert store.get_session_history(second_session)[0]["content"] == "second session"


def test_conversation_persistence_rejects_cid_content_collision(tmp_path):
    store = ConversationPersistence(tmp_path / "cid-content-collision.db")
    session_id = store.start_session()
    original_id = store.record_turn(
        "user",
        "original prompt",
        cid="immutable-exchange:user",
        session_id=session_id,
    )

    with pytest.raises(ValueError, match="conversation turn cid conflict"):
        store.record_turn(
            "user",
            "different prompt",
            cid="immutable-exchange:user",
            session_id=session_id,
        )

    history = store.get_session_history(session_id)
    assert [(row["id"], row["content"]) for row in history] == [
        (original_id, "original prompt")
    ]


def test_conversation_persistence_without_cid_never_collapses_exchanges(tmp_path):
    store = ConversationPersistence(tmp_path / "cidless-exchanges.db")
    session_id = store.start_session()

    first_ids = store.record_exchange("question one", "answer one")
    second_ids = store.record_exchange("question two", "answer two")
    history = store.get_session_history(session_id)

    assert set(first_ids).isdisjoint(second_ids)
    assert [row["content"] for row in history] == [
        "question one",
        "answer one",
        "question two",
        "answer two",
    ]
    assert all(not row["cid"] for row in history)


def test_conversation_persistence_exchange_reuses_prelogged_user_by_cid(monkeypatch, tmp_path):
    published: list[tuple[str, dict[str, object]]] = []

    class Bus:
        def publish_threadsafe(self, topic, payload):
            published.append((topic, payload))

    _install_event_bus(monkeypatch, Bus())

    store = ConversationPersistence(tmp_path / "conversations.db")
    session_id = store.start_session()
    prelogged_user_id = store.record_turn(
        "user",
        "foreground prompt",
        origin="desktop_ui",
        cid="race-42:user",
    )
    user_id, aura_id = store.record_exchange(
        "foreground prompt",
        "foreground answer",
        origin="desktop_ui",
        cid="race-42",
    )

    history = store.get_session_history(session_id)

    assert user_id == prelogged_user_id
    assert aura_id
    assert [row["role"] for row in history] == ["user", "aura"]
    assert [row["cid"] for row in history] == ["race-42:user", "race-42:aura"]
    assert [event[1]["role"] for event in published] == ["user", "aura"]


def test_conversation_persistence_bounded_history_returns_newest_turns(tmp_path):
    store = ConversationPersistence(tmp_path / "conversations.db")
    session_id = store.start_session()
    for index in range(8):
        store.record_turn("user", f"turn-{index}")

    history = store.get_session_history(session_id, limit=3)

    assert [row["content"] for row in history] == ["turn-5", "turn-6", "turn-7"]


def test_conversation_persistence_serializes_concurrent_writers(monkeypatch, tmp_path):
    class Bus:
        def publish_threadsafe(self, _topic, _payload):
            return None

    _install_event_bus(monkeypatch, Bus())
    store = ConversationPersistence(tmp_path / "concurrent-conversations.db")
    session_id = store.start_session()

    def write_turn(index: int) -> str:
        return store.record_turn(
            "user",
            f"concurrent-turn-{index}",
            origin="desktop_ui",
            cid=f"concurrent-{index}",
            session_id=session_id,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        turn_ids = list(pool.map(write_turn, range(40)))

    history = store.get_session_history(session_id, limit=100)

    assert len(set(turn_ids)) == 40
    assert len(history) == 40
    assert {row["cid"] for row in history} == {
        f"concurrent-{index}" for index in range(40)
    }


def test_conversation_persistence_async_publish_is_scheduled(monkeypatch, tmp_path):
    published: list[tuple[str, dict[str, object]]] = []
    scheduled: list[str] = []

    class Bus:
        async def publish(self, topic, payload):
            await asyncio.sleep(0)
            published.append((topic, payload))

    class Tracker:
        def create_task(self, coro, name=None):
            scheduled.append(name or "")
            return asyncio.create_task(coro)

    _install_event_bus(monkeypatch, Bus())
    monkeypatch.setattr(persistence_module, "get_task_tracker", lambda: Tracker())

    async def scenario():
        store = ConversationPersistence(tmp_path / "async-conversations.db")
        turn_id = store.record_turn("aura", "scheduled event", cid="cid-async")
        await asyncio.sleep(0.01)
        return turn_id

    turn_id = asyncio.run(scenario())

    assert turn_id
    assert scheduled == ["conversation.turn_recorded.publish"]
    assert published[0][0] == "turn_recorded"
    assert published[0][1]["cid"] == "cid-async"


def test_conversation_persistence_scheduler_failure_records_receipt(monkeypatch, tmp_path):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    class TaskSpec:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Scheduler:
        async def register(self, _spec):
            self.attempted = True
            raise RuntimeError("scheduler unavailable")

    def record_degradation(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    scheduler_module = types.ModuleType("core.scheduler")
    scheduler_module.TaskSpec = TaskSpec
    scheduler_module.scheduler = Scheduler()
    monkeypatch.setitem(sys.modules, "core.scheduler", scheduler_module)
    monkeypatch.setattr(persistence_module, "record_degradation", record_degradation)

    store = ConversationPersistence(tmp_path / "scheduler-conversations.db")
    asyncio.run(store.on_start_async())

    assert store.get_retention_status()["last_persist_error_at"] > 0
    assert recorded
    assert recorded[0][0] == "persistence"
    assert recorded[0][1] == "RuntimeError"
    assert recorded[0][2]["receipt_required"] is True
    assert "scheduled conversation pruning" in str(recorded[0][2]["action"])


def test_empty_boot_sessions_do_not_displace_real_conversations(tmp_path):
    """LIVE DEFECT 2026-08-10: a whole day's conversation went unrecallable.

    Every boot opens a session row before anything is said in it. Durable
    recall scans a bounded number of recent sessions, so each restart spent one
    of those slots on an empty row. Five empty sessions were created on the day
    this was found; three would have been enough to hide the conversation
    entirely, and she reported "my state was reset and I have no memory of it"
    with all 34 turns of it on disk.
    """
    store = ConversationPersistence(tmp_path / "conversations.db")

    real = store.start_session({"kind": "morning"})
    store.record_turn("user", "we were talking about senses", origin="text")
    store.record_turn("aura", "you said you'd give up the screen", origin="text")

    # Three restarts, none of them said anything.
    for _ in range(3):
        store.start_session({"kind": "boot"})

    unfiltered = store.get_recent_sessions(limit=3)
    assert all(int(s["turn_count"]) == 0 for s in unfiltered), (
        "boot rows should be the newest — this is the condition that hid history"
    )

    with_content = store.get_recent_sessions(limit=3, with_turns_only=True)
    assert [s["id"] for s in with_content] == [real]
    assert int(with_content[0]["turn_count"]) == 2


def test_recover_last_session_skips_empty_boot_rows(tmp_path):
    store = ConversationPersistence(tmp_path / "conversations.db")

    real = store.start_session({"kind": "real"})
    store.record_turn("user", "remember this one", origin="text")
    store.start_session({"kind": "boot"})

    assert store.recover_last_session() == real


def test_recover_last_session_returns_none_when_nothing_was_ever_said(tmp_path):
    store = ConversationPersistence(tmp_path / "conversations.db")
    store.start_session({"kind": "boot"})
    assert store.recover_last_session() is None


def test_conversation_persistence_enforces_exact_session_principal(tmp_path):
    store = ConversationPersistence(tmp_path / "principal-bound.db")
    session_id = "paired-session-a"
    store.record_turn(
        "user",
        "principal A secret",
        session_id=session_id,
        principal_id="paired-device:a",
        principal_surface="paired_device",
    )

    assert store.get_session_history(
        session_id,
        principal_id="paired-device:a",
        principal_surface="paired_device",
    )[0]["content"] == "principal A secret"
    assert store.get_session_history(
        session_id,
        principal_id="paired-device:b",
        principal_surface="paired_device",
    ) == []
    with pytest.raises(PermissionError, match="principal mismatch"):
        store.record_turn(
            "user",
            "principal B takeover",
            session_id=session_id,
            principal_id="paired-device:b",
            principal_surface="paired_device",
        )


def test_paired_surface_cannot_adopt_legacy_unbound_transcript(tmp_path):
    store = ConversationPersistence(tmp_path / "legacy-binding.db")
    session_id = "legacy-owner-session"
    store.record_turn("user", "owner legacy history", session_id=session_id)

    assert store.get_session_history(
        session_id,
        principal_id="paired-device:a",
        principal_surface="paired_device",
    ) == []
    with pytest.raises(PermissionError, match="only be adopted by the owner"):
        store.record_turn(
            "user",
            "paired claim",
            session_id=session_id,
            principal_id="paired-device:a",
            principal_surface="paired_device",
        )

    store.record_turn(
        "aura",
        "owner continuity",
        session_id=session_id,
        principal_id="bryan",
        principal_surface="owner",
    )
    assert len(
        store.get_session_history(
            session_id,
            principal_id="bryan",
            principal_surface="owner",
        )
    ) == 2


def test_recent_sessions_are_principal_scoped(tmp_path):
    store = ConversationPersistence(tmp_path / "principal-recent.db")
    for suffix in ("a", "b"):
        store.record_turn(
            "user",
            f"secret {suffix}",
            session_id=f"session-{suffix}",
            principal_id=f"paired-device:{suffix}",
            principal_surface="paired_device",
        )

    visible = store.get_recent_sessions(
        limit=10,
        with_turns_only=True,
        principal_id="paired-device:a",
        principal_surface="paired_device",
    )
    assert [session["id"] for session in visible] == ["session-a"]
