import asyncio
import json
import queue
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def _content_item(title: str):
    from core.autonomy.curated_media_loader import ContentItem

    return ContentItem(
        category="Science education",
        title=title,
        creator=None,
        url=None,
        description="runtime contract topic",
    )


def _raise_for_boundary_test(exc: BaseException):
    if type(exc).__name__:
        raise exc
    return None


@pytest.mark.asyncio
async def test_model_manager_evict_loop_stops_at_capacity(monkeypatch):
    from core.utils import model_manager
    from core.utils.model_manager import ModelManager

    closed: list[str] = []

    class ManagedModel:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    manager = ModelManager(lambda name, _opts: ManagedModel(name), max_models=1)
    manager._models["a"] = ManagedModel("a")
    manager._models["b"] = ManagedModel("b")
    manager._meta["a"] = {}
    manager._meta["b"] = {}
    manager._last_used["a"] = 1.0
    manager._last_used["b"] = 2.0

    monkeypatch.setattr(
        model_manager.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=0.0),
    )

    await manager.evict_if_needed()

    assert list(manager._models) == ["b"]
    assert closed == ["a"]


@pytest.mark.asyncio
async def test_model_manager_holds_lane_lease_until_unload(monkeypatch):
    from core.runtime import model_lane_control
    from core.utils import model_manager
    from core.utils.model_manager import ModelManager

    captured: list[dict[str, object]] = []

    class _Lease:
        released = False

        async def release(self, *, reason):
            captured.append({"release_reason": reason})
            self.released = True
            return True

    lease = _Lease()

    async def _acquire(**kwargs):
        captured.append(kwargs)
        return lease

    monkeypatch.setattr(model_lane_control, "acquire_in_process_model_lane", _acquire)
    monkeypatch.setattr(
        model_manager.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=20.0),
    )
    manager = ModelManager(lambda _name, _opts: object(), max_models=1)

    loaded = await manager.load_model(
        "coder",
        {"model_path": "/models/coder-7b", "declared_gb": 5.0},
    )

    assert loaded is not None
    assert manager._meta["coder"]["lane_lease"] is lease
    assert captured[0]["model_path"] == "/models/coder-7b"

    assert await manager.unload_model("coder") is True
    assert lease.released is True


@pytest.mark.asyncio
async def test_model_manager_cancellation_after_load_cleans_object_and_lane(
    monkeypatch,
) -> None:
    from core.runtime import model_lane_control
    from core.utils import model_manager
    from core.utils.model_manager import ModelManager

    events: list[str] = []

    class _Lease:
        async def release(self, *, reason):
            events.append(f"release:{reason}")
            return True

    class _Model:
        def close(self) -> None:
            events.append("model:closed")

    async def _acquire(**_kwargs):
        return _Lease()

    manager: ModelManager
    loaded = asyncio.Event()

    async def _load(_name, _opts):
        await manager._lock.acquire()
        loaded.set()
        return _Model()

    monkeypatch.setattr(model_lane_control, "acquire_in_process_model_lane", _acquire)
    monkeypatch.setattr(
        model_manager.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=20.0),
    )
    manager = ModelManager(_load, max_models=1)
    task = asyncio.create_task(manager.load_model("cancelled"))
    await loaded.wait()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    manager._lock.release()

    assert manager.list_loaded() == []
    assert events == ["model:closed", "release:model_manager_publish_cancelled"]


@pytest.mark.asyncio
async def test_streaming_coordinator_subscriber_exits_on_close() -> None:
    from core.multimodal.coordinator import StreamingCoordinator

    coordinator = StreamingCoordinator()
    timeline = await coordinator.open_turn(turn_id="turn-test")

    async def collect_events() -> list[str]:
        events: list[str] = []
        async for event in coordinator.subscribe(timeline.turn_id):
            events.append(event.kind)
        return events

    task = asyncio.create_task(collect_events())
    await asyncio.sleep(0)
    await coordinator.emit(timeline.turn_id, "text_token", {"text": "hello"})
    await coordinator.close(timeline.turn_id)

    assert await asyncio.wait_for(task, timeout=1.0) == ["text_token"]


def test_singleton_process_metadata_records_unobserved_pid(resource_observer) -> None:
    """Identity flows through the canonical observer (71b5598f): an
    unobserved pid must record an honest identity_error, and an observed
    one must carry observation provenance."""
    from core.utils import singleton

    metadata = singleton._current_process_metadata("unit-test", 123)

    assert metadata["identity_error"] == "RuntimeError: process_identity_unavailable"
    assert "create_time" not in metadata

    from core.runtime.resource_observation import ProcessObservation

    resource_observer.configure_processes([
        ProcessObservation(
            provenance=resource_observer.provenance,
            pid=123,
            ppid=1,
            create_time=1_700_000_000.0,
            status="running",
            name="unit-test-proc",
            cmdline=("unit-test",),
            rss_bytes=1024,
            username="bryan",
        )
    ])
    observed = singleton._current_process_metadata("unit-test", 123)

    assert observed["observation_source"] == "simulated"
    assert observed["create_time"] == 1_700_000_000.0


def test_runtime_loop_sources_have_explicit_boundaries() -> None:
    source_expectations = {
        "core/agency/repl_daemon.py": ["while True", "except BaseException"],
        "core/utils/model_manager.py": ["while True", "pass  # no-op"],
        "core/utils/singleton.py": ["except Exception"],
        "core/multimodal/coordinator.py": ["while True", "pass  # no-op"],
        "core/resilience/immune_system.py": ["while True"],
        "core/sandbox/bash_daemon.py": ["while True"],
        "core/senses/sensory_worker.py": ["while True", "except Exception"],
        "core/brain/llm/mlx_worker.py": ["while True"],
        "core/brain/llm/mlx_vision_worker.py": ["while True"],
        "core/brain/llm/nucleus_manager.py": ["while True"],
        "core/consciousness/aura_protocol.py": ["while True"],
        "core/consciousness/unified_audit.py": ["while True"],
        "core/skills/sovereign_terminal.py": ["while True"],
    }

    for source_path, forbidden_fragments in source_expectations.items():
        source = Path(source_path).read_text(encoding="utf-8")
        for forbidden in forbidden_fragments:
            assert forbidden not in source


def test_runtime_exception_boundaries_do_not_use_broad_catches() -> None:
    source_expectations = {
        "core/affect/affective_circumplex.py": ["except Exception", "except BaseException"],
        "core/agency/agency_core.py": ["except Exception", "except BaseException"],
        "core/consciousness/closed_loop.py": ["except Exception", "except BaseException"],
        "core/consciousness/executive_closure.py": ["except Exception", "except BaseException"],
        "core/consciousness/hierarchical_phi.py": ["except Exception", "except BaseException"],
        "core/morphogenesis/cell.py": ["except Exception", "except BaseException"],
        "core/phases/response_generation.py": ["except Exception", "except BaseException"],
        "core/resilience/error_boundary.py": ["except Exception", "except BaseException"],
        "core/resilience/phenomenal_error_map.py": ["except Exception", "except BaseException"],
        "core/skills/base_skill.py": ["except Exception", "except BaseException"],
        "core/soma/resilience_engine.py": ["except Exception", "except BaseException"],
        "main_daemon.py": ["except Exception", "except BaseException"],
    }

    for source_path, forbidden_fragments in source_expectations.items():
        source = Path(source_path).read_text(encoding="utf-8")
        for forbidden in forbidden_fragments:
            assert forbidden not in source


@pytest.mark.asyncio
async def test_error_boundary_preserves_invariant_failures() -> None:
    from core.resilience.error_boundary import error_boundary

    @error_boundary(name="unit-runtime-error", fallback_value="fallback")
    async def recoverable_failure():
        return _raise_for_boundary_test(RuntimeError("runtime dependency unavailable"))

    @error_boundary(name="unit-invariant-error", fallback_value="fallback")
    async def invariant_failure():
        return _raise_for_boundary_test(AssertionError("invariant broken"))

    assert await recoverable_failure() == "fallback"
    with pytest.raises(AssertionError, match="invariant broken"):
        await invariant_failure()


@pytest.mark.asyncio
async def test_phenomenal_error_map_preserves_invariant_failures() -> None:
    from core.resilience.phenomenal_error_map import PhenomenalRaise, phenomenal

    @phenomenal()
    async def recoverable_failure():
        return _raise_for_boundary_test(RuntimeError("network offline"))

    @phenomenal()
    async def invariant_failure():
        return _raise_for_boundary_test(AssertionError("phenomenal invariant broken"))

    with pytest.raises(PhenomenalRaise):
        await recoverable_failure()
    with pytest.raises(AssertionError, match="phenomenal invariant broken"):
        await invariant_failure()


@pytest.mark.asyncio
async def test_closed_loop_thread_task_preserves_invariant_failures() -> None:
    from core.consciousness.closed_loop import ClosedCausalLoop

    loop = ClosedCausalLoop()

    def invariant_failure():
        return _raise_for_boundary_test(AssertionError("closed-loop invariant broken"))

    with pytest.raises(AssertionError, match="closed-loop invariant broken"):
        await loop._safe_run_thread_task(
            invariant_failure,
            action="unit test invariant escape",
            stage="unit_test",
        )


def test_evolution_orchestrator_does_not_swallow_invariant_failures() -> None:
    source = Path("core/evolution/evolution_orchestrator.py").read_text(encoding="utf-8")

    assert "except Exception" not in source
    assert "_EVOLUTION_RECOVERABLE_ERRORS" in source


@pytest.mark.asyncio
async def test_evolution_orchestrator_invariant_failure_escapes(monkeypatch, tmp_path) -> None:
    from core.evolution.evolution_orchestrator import EvolutionOrchestrator

    monkeypatch.setattr(EvolutionOrchestrator, "_STATE_FILE", tmp_path / "evolution_state.json")
    orchestrator = EvolutionOrchestrator()
    calls: list[str] = []

    async def invariant_failure():
        calls.append("called")
        raise AssertionError("evolution invariant broken")

    monkeypatch.setattr(orchestrator, "_eval_self_awareness", invariant_failure)

    with pytest.raises(AssertionError, match="evolution invariant broken"):
        await orchestrator.tick()
    assert calls == ["called"]


def test_research_trigger_payload_is_json_safe(tmp_path) -> None:
    from core.autonomy import research_triggers

    class NonSerializable:
        def __repr__(self) -> str:
            return "<non-serializable-marker>"

    trigger_path = tmp_path / "research-triggers.jsonl"
    research_triggers.emit_research_trigger(
        topic="runtime payload hygiene",
        source_intent_id="intent-runtime",
        payload_hint={"object": NonSerializable(), "nested": {"set": {1, 2}}},
        path=trigger_path,
    )

    [trigger] = research_triggers.drain_pending_triggers(path=trigger_path)
    assert trigger.payload_hint["object"] == "<non-serializable-marker>"
    assert trigger.payload_hint["nested"]["set"] == "{1, 2}"


def test_research_trigger_default_path_is_runtime_state() -> None:
    from core.autonomy import research_triggers

    path = research_triggers.DEFAULT_TRIGGER_PATH

    # The point of this test is its name: the default is RUNTIME STATE, not a
    # path inside the checkout. state_root() is what expresses that — it is
    # ~/.aura for a live runtime and an isolated sibling for a test one, so
    # pinning Path.home()/".aura" demanded that a TEST run default into the
    # live instance's data directory.
    from core.runtime.state_ownership import state_root

    assert "live-source" not in str(path)
    assert path == state_root() / "data" / "autonomy" / "research-triggers.jsonl"


def test_research_trigger_env_path_is_resolved_at_call_time(tmp_path, monkeypatch) -> None:
    from core.autonomy import research_triggers

    trigger_path = tmp_path / "runtime" / "research-triggers.jsonl"
    monkeypatch.setenv("AURA_RESEARCH_TRIGGER_PATH", str(trigger_path))

    research_triggers.emit_research_trigger(
        topic="runtime path hygiene",
        source_intent_id="env-runtime",
        payload_hint={"content": "store outside the source tree"},
    )

    [trigger] = research_triggers.drain_pending_triggers()
    assert trigger.source_intent_id == "env-runtime"
    assert trigger_path.exists()


def test_pending_chat_default_path_is_runtime_state(monkeypatch) -> None:
    from core.conversation import chat_preflight
    from core.runtime.state_ownership import state_root

    monkeypatch.delenv("AURA_PENDING_CHAT_QUEUE_PATH", raising=False)
    path = chat_preflight._resolve_pending_queue_path()

    assert "live-source" not in str(path)
    assert path == state_root() / "data" / "conversation" / "pending-chat-queue.jsonl"


def test_pending_chat_env_path_is_resolved_at_call_time(tmp_path, monkeypatch) -> None:
    from core.conversation import chat_preflight

    queue_path = tmp_path / "runtime" / "pending-chat-queue.jsonl"
    monkeypatch.setenv("AURA_PENDING_CHAT_QUEUE_PATH", str(queue_path))

    chat_preflight.enqueue("session-runtime", "answer this later")

    assert queue_path.exists()
    assert chat_preflight.has_unanswered_for_session("session-runtime")


def test_autonomy_runtime_state_paths_stay_out_of_source_tree(tmp_path, monkeypatch) -> None:
    from core.autonomy.autonomous_research_orchestrator import AutonomousResearchOrchestrator
    from core.autonomy.content_progress_tracker import ProgressEntry, ProgressLog, load
    from core.autonomy.curated_media_loader import load_corpus
    from core.autonomy.memory_persister import MemoryPersister

    progress_path = tmp_path / "data" / "curated-progress.json"
    queue_path = tmp_path / "data" / "persist-retry.jsonl"
    dedup_path = tmp_path / "data" / "persist-dedup.json"
    sessions_dir = tmp_path / "data" / "research-sessions"
    corpus_path = tmp_path / "source" / "bryan-curated-media.md"
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_text(
        "# The library\n\n## Test category\n- **A test item** — https://example.com — Useful.\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("AURA_CURATED_MEDIA_PROGRESS_PATH", str(progress_path))
    monkeypatch.setenv("AURA_MEMORY_PERSIST_RETRY_QUEUE_PATH", str(queue_path))
    monkeypatch.setenv("AURA_MEMORY_PERSIST_DEDUP_PATH", str(dedup_path))
    monkeypatch.setenv("AURA_RESEARCH_SESSIONS_DIR", str(sessions_dir))
    monkeypatch.setenv("AURA_CURATED_MEDIA_CORPUS_PATH", str(corpus_path))

    log = ProgressLog()
    log.add_entry(
        ProgressEntry(
            title="A test item",
            started_at="2026-06-22T00:00:00Z",
            method_priority_level=1,
            method_detail="unit",
        )
    )
    log.save()

    loaded = load()
    persister = MemoryPersister()
    orchestrator = AutonomousResearchOrchestrator()
    items = load_corpus()

    assert progress_path.exists()
    assert loaded.find("A test item") is not None
    assert persister._queue_path == queue_path
    assert persister._dedup_path == dedup_path
    assert orchestrator._sessions_dir == sessions_dir
    assert len(items) == 1
    for path in (progress_path, queue_path, dedup_path, sessions_dir):
        assert "live-source" not in str(path)


def test_research_triggers_reject_live_reply_failure_contamination(tmp_path) -> None:
    from core.autonomy import research_triggers

    trigger_path = tmp_path / "research-triggers.jsonl"
    research_triggers.emit_research_trigger(
        topic="write_memory:interaction_commit",
        source_intent_id="bad-live-reply",
        payload_hint={
            "content": (
                "Ok. Just checking. I'll be back, ok? -> conversation_reply -> "
                "I lost the reply lane for a moment. Ask that again and I'll answer cleanly."
            )
        },
        path=trigger_path,
    )

    assert not trigger_path.exists() or trigger_path.read_text(encoding="utf-8") == ""
    assert research_triggers.drain_pending_triggers(path=trigger_path) == []


def test_research_trigger_drain_skips_preexisting_contaminated_lines(tmp_path) -> None:
    from core.autonomy import research_triggers

    trigger_path = tmp_path / "research-triggers.jsonl"
    research_triggers.emit_research_trigger(
        topic="valid curiosity",
        source_intent_id="valid",
        payload_hint={"content": "look up bounded runtime recovery patterns"},
        path=trigger_path,
    )
    with trigger_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "topic": "write_memory:interaction_commit",
                    "source_intent_id": "old-bad-line",
                    "contested_count": 1,
                    "payload_hint": {
                        "content": "As an AI language model, I do not have personal feelings."
                    },
                    "emitted_at": time.time(),
                    "consumed_at": None,
                }
            )
            + "\n"
        )

    triggers = research_triggers.drain_pending_triggers(path=trigger_path)
    assert [trigger.source_intent_id for trigger in triggers] == ["valid"]


def test_curiosity_scheduler_degrades_known_reader_errors_but_surfaces_invariants() -> None:
    from core.autonomy.content_progress_tracker import ProgressLog
    from core.autonomy.curiosity_scheduler import CuriosityScheduler

    def typed_failure_reader() -> dict[str, float]:
        observed = "substrate unavailable"
        raise RuntimeError(observed)

    scheduler = CuriosityScheduler(
        corpus_loader=lambda: [_content_item("bounded curiosity")],
        progress_loader=lambda: ProgressLog(),
        substrate_reader=typed_failure_reader,
        trigger_drainer=lambda: [],
    )

    assert scheduler.pick_next() is not None

    invariant_calls: list[str] = []

    def invariant_failure_reader() -> dict[str, float]:
        invariant_calls.append("called")
        raise AssertionError("substrate invariant broken")

    invariant_scheduler = CuriosityScheduler(
        corpus_loader=lambda: [_content_item("surfaced invariant")],
        progress_loader=lambda: ProgressLog(),
        substrate_reader=invariant_failure_reader,
        trigger_drainer=lambda: [],
    )

    with pytest.raises(AssertionError, match="substrate invariant broken"):
        invariant_scheduler.pick_next()
    assert invariant_calls == ["called"]


@pytest.mark.asyncio
async def test_process_tcell_patrol_has_stop_contract_and_cancels_stale_named_tasks() -> None:
    from core.resilience.immune_system import ProcessTCell

    async def sleeper() -> None:
        await asyncio.sleep(30)

    task = asyncio.create_task(sleeper(), name="runtime_stale_task")
    tcell = ProcessTCell(max_lifespan_seconds=0.1, patrol_interval=0.001)
    tcell._first_seen[id(task)] = time.monotonic() - 1.0

    assert tcell.patrol_once(current_task=asyncio.current_task()) == 1
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(tcell.patrol_bloodstream(max_cycles=1), timeout=0.5)


@pytest.mark.asyncio
async def test_bash_daemon_reader_exits_on_delimiter_without_forever_loop() -> None:
    from core.sandbox.bash_daemon import PersistentBashSession

    class FakeStdout:
        def __init__(self, lines: list[bytes]) -> None:
            self._items = queue.SimpleQueue()
            for line in lines:
                self._items.put(line)

        async def readline(self) -> bytes:
            if self._items.empty():
                return b""
            return self._items.get()

    session = PersistentBashSession(cwd="/tmp")
    session._process = SimpleNamespace(
        stdout=FakeStdout([b"hello\n", f"{session._delimiter}:0\n".encode()])
    )

    assert await session._read_until_delimiter() == ("hello", 0)


@pytest.mark.asyncio
async def test_bash_daemon_execute_preserves_legacy_timeout_keyword(monkeypatch) -> None:
    from core.sandbox.bash_daemon import PersistentBashSession

    class FakeStdin:
        def __init__(self) -> None:
            self.writes: list[bytes] = []

        def write(self, payload: bytes) -> None:
            self.writes.append(payload)

        async def drain(self) -> None:
            return None

    process = SimpleNamespace(stdin=FakeStdin(), returncode=None)
    session = PersistentBashSession(cwd="/tmp")
    session._process = process
    monkeypatch.setattr(session, "_read_until_delimiter", lambda: asyncio.sleep(0, result=("ok", 0)))

    assert await session.execute("echo ok", timeout=0.1) == (True, "ok")


@pytest.mark.asyncio
async def test_bash_daemon_starts_process_through_subprocess_gateway(monkeypatch) -> None:
    import core.sandbox.bash_daemon as bash_daemon
    from core.sandbox.bash_daemon import PersistentBashSession

    spawn_calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    class FakeStdout:
        def __init__(self, line: bytes) -> None:
            self._line = line
            self._used = False

        async def readline(self) -> bytes:
            if self._used:
                return b""
            self._used = True
            return self._line

    class FakeStdin:
        def __init__(self) -> None:
            self.writes: list[bytes] = []

        def write(self, payload: bytes) -> None:
            self.writes.append(payload)

        async def drain(self) -> None:
            return None

    class FakeSubprocessGateway:
        async def spawn_async(self, argv, **kwargs):
            spawn_calls.append((tuple(argv), kwargs))
            return SimpleNamespace(
                stdin=FakeStdin(),
                stdout=FakeStdout(f"{session._delimiter}:0\n".encode()),
                returncode=None,
            )

    session = PersistentBashSession(cwd="/tmp")
    monkeypatch.setattr(
        bash_daemon,
        "get_subprocess_gateway",
        lambda: FakeSubprocessGateway(),
    )

    await session._start()

    assert spawn_calls
    argv, kwargs = spawn_calls[0]
    assert argv == ("bash", "--noprofile", "--norc")
    assert kwargs["source"] == "core.sandbox.bash_daemon.persistent_bash"
    assert session._process.stdin.writes
    assert b"PROMPT_COMMAND" in session._process.stdin.writes[0]


def test_sensory_worker_ping_and_exit_contract() -> None:
    from core.senses.sensory_worker import sensory_worker_loop

    request_queue: queue.Queue[dict[str, str]] = queue.Queue()
    response_queue: queue.Queue[dict[str, str]] = queue.Queue()
    request_queue.put({"command": "ping"})
    request_queue.put({"command": "exit"})

    sensory_worker_loop(request_queue, response_queue)

    assert response_queue.get(timeout=0.1) == {"status": "ok", "msg": "pong"}


class TestTheSettingsRefreshLaneCanBeStopped:
    """`while True` in a daemon thread has no way to be told to stop.

    The runtime-settings refresh lane looped forever. At interpreter
    shutdown its last iteration ran against half-torn-down imports, and a
    test that needed a quiet process had no way to end it. The loop now
    reads a stop event, and there is a function that sets it and waits.
    """

    def test_the_worker_leaves_the_loop_when_asked(self) -> None:
        from core.runtime import runtime_settings

        runtime_settings._ensure_refresh_worker()
        thread = runtime_settings._refresh_thread
        assert thread is not None and thread.is_alive()

        assert runtime_settings.stop_refresh_worker(timeout_s=5.0) is True
        assert thread.is_alive() is False
        assert runtime_settings._refresh_thread is None

    def test_reading_settings_again_restarts_the_lane(self) -> None:
        from core.runtime import runtime_settings

        runtime_settings._ensure_refresh_worker()
        runtime_settings.stop_refresh_worker(timeout_s=5.0)

        runtime_settings._ensure_refresh_worker()
        restarted = runtime_settings._refresh_thread
        assert restarted is not None and restarted.is_alive()
        assert runtime_settings.stop_refresh_worker(timeout_s=5.0) is True

    def test_the_loop_condition_is_an_event_not_a_constant(self) -> None:
        """A ratchet the next edit has to answer to."""
        import inspect

        from core.runtime import runtime_settings

        source = inspect.getsource(runtime_settings._refresh_worker)
        assert "while True" not in source
        assert "_refresh_stop.is_set()" in source
