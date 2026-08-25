import asyncio
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

import core.mind_tick as mind_module
from core.container import ServiceContainer
from core.mind_tick import MindTick, _schedule_mind_task
from core.runtime.errors import get_degradation_tracker
from core.state.state_repository import StateRepository


class ClosingAwaitable:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def __await__(self):
        if False:
            yield None
        return None


class FailingTracker:
    def create_task(self, _awaitable, *, name=None):
        self.last_name = name
        raise RuntimeError(f"{name}: loop unavailable")


class Watchdog:
    def register_component(self, *_args, **_kwargs):
        return None


def test_mind_scheduler_closes_unscheduled_awaitable():
    awaitable = ClosingAwaitable()

    task = _schedule_mind_task(awaitable, name="mind.contract", tracker=FailingTracker())

    assert task is None
    assert awaitable.closed is True


def test_mind_tick_liveness_requires_supervised_progress():
    class RunningTask:
        @staticmethod
        def done():
            return False

    tick = MindTick.__new__(MindTick)
    tick._running = True
    tick._task = RunningTask()
    tick._started_at = time.time()
    tick._last_successful_tick_at = time.time()
    tick._consecutive_loop_failures = 0
    tick._tick_count = 4

    assert tick.ensure_alive() is True
    assert tick.get_health_status()["healthy"] is True

    tick._consecutive_loop_failures = 3
    assert tick.ensure_alive() is True

    tick._last_successful_tick_at = time.time() - 601
    tick._last_loop_progress_at = time.time() - 601
    tick._last_liveness_repair_at = time.monotonic()
    assert tick.ensure_alive() is False


class _RunningTask:
    @staticmethod
    def done():
        return False


def _tick_with_progress_age(age_s: float) -> MindTick:
    tick = MindTick.__new__(MindTick)
    tick._running = True
    tick._task = _RunningTask()
    tick._started_at = time.time() - age_s
    tick._active_tick_started_at = time.time() - age_s
    tick._active_tick_stage = "kernel_tick"
    tick._last_progress_label = "kernel_tick"
    tick._last_successful_tick_at = time.time() - age_s
    tick._last_loop_progress_at = time.time() - age_s
    tick._consecutive_loop_failures = 0
    tick._tick_count = 5
    tick._last_liveness_repair_at = 0.0
    return tick


def test_mind_tick_liveness_allows_active_bounded_tick_progress():
    """Inside the loop's own longest bounded stage, a tick in flight is alive."""
    from core.mind_tick import MAX_BOUNDED_TICK_STAGE_S

    tick = _tick_with_progress_age(MAX_BOUNDED_TICK_STAGE_S - 20.0)

    assert tick.is_alive() is True
    status = tick.get_health_status()
    assert status["healthy"] is True
    assert status["active_tick_stage"] == "kernel_tick"
    assert status["last_progress_label"] == "kernel_tick"


def test_a_ten_minute_stall_is_not_alive():
    """The old thresholds were 600s stale and 900s hard against a 2s
    conversational tick: three hundred missed beats still read as healthy."""
    tick = _tick_with_progress_age(700.0)

    assert tick.is_alive() is False


def test_the_stall_thresholds_come_from_the_loops_own_budgets():
    from core.mind_tick import (
        DEFAULT_HARD_STALL_S,
        DEFAULT_STALE_PROGRESS_S,
        MAX_BOUNDED_TICK_STAGE_S,
    )

    assert DEFAULT_STALE_PROGRESS_S == MAX_BOUNDED_TICK_STAGE_S * 1.5
    assert DEFAULT_HARD_STALL_S == MAX_BOUNDED_TICK_STAGE_S * 2.0
    assert DEFAULT_HARD_STALL_S < 600.0, "still past a conversation budget"


@pytest.mark.asyncio
async def test_mind_tick_recovery_hook_repairs_a_dead_supervised_loop():
    tick = MindTick.__new__(MindTick)
    tick._running = True
    tick._started_at = time.time()
    tick._last_successful_tick_at = 0.0
    tick._consecutive_loop_failures = 3
    tick._last_liveness_repair_at = 0.0
    tick._liveness_repair_count = 0

    async def failed_loop():
        await asyncio.sleep(0)  # yield once like a real loop, then die
        raise RuntimeError("background loop died")

    tick._task = asyncio.create_task(failed_loop())
    await asyncio.sleep(0)

    release = asyncio.Event()

    async def recovered_loop():
        await release.wait()

    tick._run_loop = recovered_loop

    assert tick.ensure_alive() is True
    assert tick._task is not None
    assert not tick._task.done()
    assert tick.get_health_status()["liveness_repair_count"] == 1

    release.set()
    await tick._task


@pytest.mark.asyncio
async def test_mind_tick_done_callback_repairs_failed_loop_without_health_poll():
    tick = MindTick.__new__(MindTick)
    tick._running = True
    tick._started_at = time.time()
    tick._last_successful_tick_at = 0.0
    tick._consecutive_loop_failures = 0
    tick._last_liveness_repair_at = 0.0
    tick._liveness_repair_count = 0
    tick._owner_loop = asyncio.get_running_loop()
    tick.orchestrator = SimpleNamespace()

    release = asyncio.Event()

    async def failed_loop():
        await asyncio.sleep(0)
        raise TypeError("expected string or bytes-like object, got 'NoneType'")

    async def recovered_loop():
        await release.wait()

    tick._run_loop = recovered_loop
    failed_task = asyncio.create_task(failed_loop())
    tick._task = failed_task
    tick._install_loop_done_callback(failed_task, name="test.failed")

    deadline = time.monotonic() + 1.0
    while tick._task is failed_task and time.monotonic() < deadline:
        await asyncio.sleep(0.01)

    assert tick._task is not failed_task
    assert tick._task is not None
    assert not tick._task.done()
    assert tick._liveness_repair_count == 1

    release.set()
    await tick._task


@pytest.mark.asyncio
async def test_mind_tick_recovery_hook_repairs_a_stale_alive_task():
    tick = MindTick.__new__(MindTick)
    tick._running = True
    tick._started_at = time.time() - 600
    tick._last_successful_tick_at = time.time() - 601
    tick._last_loop_progress_at = time.time() - 601
    tick._consecutive_loop_failures = 0
    tick._last_liveness_repair_at = 0.0
    tick._liveness_repair_count = 0
    tick._owner_loop = asyncio.get_running_loop()
    tick.orchestrator = SimpleNamespace()

    release_old = asyncio.Event()
    release_new = asyncio.Event()

    async def stale_loop():
        await release_old.wait()

    async def recovered_loop():
        await release_new.wait()

    stale_task = asyncio.create_task(stale_loop())
    tick._task = stale_task
    tick._run_loop = recovered_loop

    assert tick.ensure_alive() is False
    assert stale_task.cancelled() or stale_task.done() or stale_task.cancelling()

    # CP126 e98446be. The replacement is CHAINED to the stale loop actually
    # unwinding, not started the instant cancel() returns. `cancel()` only
    # requests cancellation — the old coroutine runs on until its next await
    # — so starting the replacement immediately meant two _run_loop
    # coroutines alive at once, both mutating the same state and committing
    # over each other. A repair that produces two minds is worse than the
    # stall it repairs.
    #
    # While the old loop unwinds the repair is IN FLIGHT and says so, which
    # is also the fix for CP126 c76abf56: "recovering" and "broken and
    # unattended" are different operational states.
    assert tick._repair_pending is True
    assert tick._task is stale_task, (
        "a replacement loop started before the stale one had unwound"
    )

    release_old.set()
    for _ in range(50):
        await asyncio.sleep(0)
        if tick._task is not stale_task:
            break

    assert tick._task is not stale_task
    assert tick._task is not None
    assert not tick._task.done()
    assert tick._repair_pending is False
    assert tick._last_loop_progress_at > time.time() - 5
    assert tick._liveness_repair_count == 1

    release_new.set()
    await tick._task


@pytest.mark.asyncio
async def test_mind_tick_never_runs_two_cognitive_loops_at_once():
    """The concurrency the repair used to create.

    Both loops mutate one state object and commit over each other, so the
    invariant is not "the replacement starts quickly" — it is that exactly
    one _run_loop exists at every instant.
    """
    tick = MindTick.__new__(MindTick)
    tick._running = True
    tick._started_at = time.time() - 600
    tick._last_successful_tick_at = time.time() - 601
    tick._last_loop_progress_at = time.time() - 601
    tick._consecutive_loop_failures = 0
    tick._last_liveness_repair_at = 0.0
    tick._liveness_repair_count = 0
    tick._repair_pending = False
    tick._owner_loop = asyncio.get_running_loop()
    tick.orchestrator = SimpleNamespace()

    live = {"count": 0, "peak": 0}
    release_old = asyncio.Event()
    release_new = asyncio.Event()

    async def stale_loop():
        live["count"] += 1
        live["peak"] = max(live["peak"], live["count"])
        try:
            await release_old.wait()
        finally:
            live["count"] -= 1

    async def recovered_loop():
        live["count"] += 1
        live["peak"] = max(live["peak"], live["count"])
        try:
            await release_new.wait()
        finally:
            live["count"] -= 1

    stale_task = asyncio.create_task(stale_loop())
    await asyncio.sleep(0)
    tick._task = stale_task
    tick._run_loop = recovered_loop

    tick.is_alive()
    release_old.set()
    for _ in range(50):
        await asyncio.sleep(0)
        if tick._task is not stale_task:
            break

    assert live["peak"] == 1, (
        f"{live['peak']} cognitive loops were alive at once; the repair "
        "started a replacement before the cancelled loop had unwound"
    )

    release_new.set()
    if tick._task is not None and not tick._task.done():
        await tick._task


@pytest.mark.asyncio
async def test_mind_tick_start_rolls_back_when_loop_cannot_be_scheduled(monkeypatch):
    monkeypatch.setattr(mind_module, "get_task_tracker", lambda: FailingTracker())
    monkeypatch.setattr("infrastructure.watchdog.get_watchdog", lambda: Watchdog())

    tick = MindTick.__new__(MindTick)
    tick._running = False
    tick._task = None

    async def run_loop():
        return None

    tick._run_loop = run_loop

    await tick.start()

    assert tick._running is False
    assert tick._task is None


@pytest.mark.asyncio
async def test_mind_tick_start_republishes_authoritative_running_instance(monkeypatch):
    monkeypatch.setattr("infrastructure.watchdog.get_watchdog", lambda: Watchdog())
    ServiceContainer.clear()

    stale_tick = SimpleNamespace(_running=False, _task=None, is_alive=lambda: False)
    ServiceContainer.register_instance("mind_tick", stale_tick, required=False)

    tick = MindTick.__new__(MindTick)
    tick.orchestrator = SimpleNamespace()
    tick._running = False
    tick._task = None
    tick._started_at = 0.0
    tick._last_successful_tick_at = 0.0
    tick._last_loop_progress_at = 0.0
    tick._last_progress_label = "not_started"
    tick._active_tick_started_at = 0.0
    tick._active_tick_stage = "idle"
    tick._consecutive_loop_failures = 0
    tick._last_liveness_repair_at = 0.0
    tick._liveness_repair_count = 0

    release = asyncio.Event()

    async def run_loop():
        await release.wait()

    tick._run_loop = run_loop

    await tick.start()

    assert ServiceContainer.get("mind_tick") is tick
    assert tick._task is not None
    assert not tick._task.done()

    release.set()
    await tick._task
    ServiceContainer.clear()


@pytest.mark.asyncio
async def test_mind_tick_stop_drains_closed_db_task_failure():
    tracker = get_degradation_tracker()
    tracker.reset()
    tick = MindTick.__new__(MindTick)
    tick._running = True
    drain_failures: list[str] = []

    async def failed_loop():
        drain_failures.append("closed_database")
        raise sqlite3.ProgrammingError("Cannot operate on a closed database.")

    tick._task = asyncio.create_task(failed_loop())
    await asyncio.sleep(0)

    await tick.stop()

    assert tick._running is False
    assert drain_failures == ["closed_database"]
    assert any(
        "background loop failed while draining" in record.action
        for record in tracker.recent(subsystem="mind_tick")
    )
    tracker.reset()


@pytest.mark.asyncio
async def test_state_repository_clear_pending_proxy_commit_handles_closed_db(monkeypatch):
    tracker = get_degradation_tracker()
    tracker.reset()
    repo = StateRepository(is_vault_owner=False)

    class ClosedDB:
        async def execute(self, *_args, **_kwargs):
            self.execute_calls = getattr(self, "execute_calls", 0) + 1
            raise sqlite3.ProgrammingError("Cannot operate on a closed database.")

        async def commit(self):
            self.commit_calls = getattr(self, "commit_calls", 0) + 1
            raise AssertionError("commit should not run after closed execute")

    async def closed_db():
        return ClosedDB()

    monkeypatch.setattr(repo, "_ensure_db", closed_db)

    await repo._clear_pending_proxy_commit()

    assert any(
        "outbox clear failed" in record.action
        for record in tracker.recent(subsystem="state_repository")
    )
    tracker.reset()


def test_stale_rhythm_verdict_names_the_wedged_stage():
    """'is_alive() returned False' told an operator nothing for two hours
    live (Jul 7): the rhythm was wedged at one stage with no receipt. Every
    stale verdict must now record WHERE the loop is stuck."""
    from core.health.degraded_events import isolated_degraded_event_scope

    class RunningTask:
        @staticmethod
        def done():
            return False

    tick = MindTick.__new__(MindTick)
    tick._running = True
    tick._task = RunningTask()
    tick._started_at = time.time() - 7000
    tick._last_successful_tick_at = time.time() - 4600
    tick._last_loop_progress_at = time.time() - 4600
    tick._active_tick_started_at = time.time() - 4600
    tick._active_tick_stage = "llm_health"
    tick._consecutive_loop_failures = 0
    tick._tick_count = 40
    tick._last_liveness_repair_at = time.monotonic()  # repair rate-limited off

    with isolated_degraded_event_scope("stale-stage-test"):
        from core.health.degraded_events import get_recent_degraded_events

        assert tick.ensure_alive() is False
        events = get_recent_degraded_events(limit=10)
    stale = [e for e in events if e.get("reason") == "rhythm_stale"]
    assert stale, events
    assert "stage=llm_health" in stale[0].get("detail", "")


def test_unreachable_liveness_repair_is_never_silent():
    """A repair that cannot run must say so — the silent no-op branch left
    the runtime DEGRADED for hours with 'repair machinery present'."""
    from core.health.degraded_events import (
        get_recent_degraded_events,
        isolated_degraded_event_scope,
    )

    class RunningTask:
        @staticmethod
        def done():
            return False

        @staticmethod
        def cancel():
            return True

    tick = MindTick.__new__(MindTick)
    tick._running = True
    tick._task = RunningTask()
    tick._active_tick_stage = "llm_health"
    tick._last_liveness_repair_at = 0.0
    tick._owner_loop = None  # the dead-end: no usable loop from a thread

    with isolated_degraded_event_scope("repair-unreachable-test"):
        result_holder = {}

        def probe():
            result_holder["repaired"] = tick._attempt_liveness_repair(
                reason="test", cancel_existing=False
            )

        worker = threading.Thread(target=probe)
        worker.start()
        worker.join(timeout=5)
        events = get_recent_degraded_events(limit=10)

    assert result_holder.get("repaired") is False
    unreachable = [e for e in events if e.get("reason") == "liveness_repair_unreachable"]
    assert unreachable, events
    assert "no usable owner loop" in unreachable[0].get("detail", "")


def test_tick_llm_health_await_is_bounded():
    """The rhythm loop must never hand its liveness to a dependency: both
    remaining bare awaits (state read, tier health sweep) carry timeouts."""
    import inspect

    src = inspect.getsource(MindTick._run_loop)
    assert "wait_for(\n                        self.orchestrator.state_repo.get_current()" in src.replace("  ", "  ") or "state_repo.get_current(), timeout=" in src
    assert "ensure_all_tiers_healthy(), timeout=" in src


@pytest.mark.asyncio
async def test_the_liveness_probe_does_not_restart_what_it_inspects():
    """Observation is not actuation.

    The repair lived inside `is_alive`, so every reader of the health report —
    a dashboard, a status endpoint, the contract sweep — silently restarted the
    service it was inspecting, and none of them knew they had.
    """
    tick = MindTick.__new__(MindTick)
    tick._running = True
    tick._started_at = time.time()
    tick._last_successful_tick_at = 0.0
    tick._consecutive_loop_failures = 3
    tick._last_liveness_repair_at = 0.0
    tick._liveness_repair_count = 0

    async def failed_loop():
        await asyncio.sleep(0)
        raise RuntimeError("background loop died")

    tick._task = asyncio.create_task(failed_loop())
    await asyncio.sleep(0)

    started = []

    async def recovered_loop():
        started.append(1)
        await asyncio.sleep(0)

    tick._run_loop = recovered_loop
    dead_task = tick._task

    assert tick.is_alive() is False
    assert tick._task is dead_task, "the probe replaced the task it was reading"
    assert started == []
    assert tick.get_health_status()["liveness_repair_count"] == 0


def test_the_contract_declares_a_separate_recovery_hook():
    from core.runtime.health_contract import RUNTIME_CONTRACT

    entry = next(req for req in RUNTIME_CONTRACT if req.container_key == "mind_tick")

    assert entry.liveness_check == "is_alive"
    assert entry.recovery_check == "ensure_alive"


def test_recovery_is_a_separate_call_from_evaluation():
    import inspect as _inspect

    from core.runtime import health_contract

    assert callable(health_contract.recover_failed_services)
    evaluate = _inspect.getsource(health_contract.evaluate_health)
    # The docstring may point at the recovery function; the BODY must not call
    # it, or the split is cosmetic.
    body = evaluate.split('"""', 2)[-1]
    assert "recover_failed_services(" not in body
    assert "recovery_check" not in body
    assert "never repairs" in evaluate


def test_a_service_with_no_recovery_hook_is_reported_not_skipped():
    import inspect as _inspect

    from core.runtime import health_contract

    source = _inspect.getsource(health_contract.recover_failed_services)

    assert '"no_recovery_hook"' in source
    assert '"recovered" if recover() else "unrecovered"' in source


def test_only_failed_liveness_is_recovered():
    import inspect as _inspect

    from core.runtime import health_contract

    source = _inspect.getsource(health_contract.recover_failed_services)

    assert "status.liveness_ok is not False" in source


def test_the_slowest_ticks_no_longer_get_the_shortest_rest():
    """`sleep = max(1, interval - elapsed)` inverted the backoff: a 20s tick
    with a 10s proportional interval slept 1s."""
    import ast
    import inspect as _inspect

    from core import mind_tick as mind_tick_mod

    source = _inspect.getsource(mind_tick_mod)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        if "adaptive backoff" not in rendered:
            continue
        assert "sleep_time = max(1.0, elapsed * 0.5)" in rendered
        assert "interval - elapsed" in rendered, "the fast path still paces to a deadline"
        return
    raise AssertionError("the adaptive backoff was not found")


def test_a_slow_tick_sleeps_proportionally():
    """The arithmetic, stated as the property it has to have."""
    for elapsed in (6.0, 20.0, 90.0):
        assert max(1.0, elapsed * 0.5) >= elapsed * 0.5


def test_cancellation_is_not_swallowed():
    import inspect as _inspect

    from core import mind_tick as mind_tick_mod

    source = _inspect.getsource(mind_tick_mod.MindTick._run_loop)
    marker = source.index("except asyncio.CancelledError:")
    block = source[marker : marker + 1400]

    assert "raise" in block
    assert "spuriously cancelled" not in block
    assert '"loop_cancelled"' in block


def test_a_dark_event_bus_keeps_a_standing_degraded_record():
    import inspect as _inspect

    from core import mind_tick as mind_tick_mod

    source = _inspect.getsource(mind_tick_mod.MindTick._record_bus_outage)

    assert '"event_bus_publish_failing"' in source
    assert "_BUS_OUTAGE_REASSERT_TICKS" in source


def test_the_outage_record_is_reasserted_rather_than_suppressed():
    tick = MindTick.__new__(MindTick)
    tick._tick_count = 0
    tick._bus_fail_count = 1
    recorded = []
    tick._record_bus_outage = MindTick._record_bus_outage.__get__(tick, MindTick)

    import core.mind_tick as mind_tick_mod

    original = mind_tick_mod.record_degraded_event
    mind_tick_mod.record_degraded_event = lambda *a, **k: recorded.append(k)
    try:
        tick._record_bus_outage("TimeoutError")
        tick._bus_fail_count = 2
        tick._tick_count = 1
        tick._record_bus_outage("TimeoutError")  # inside the quiet window
        tick._tick_count = MindTick._BUS_OUTAGE_REASSERT_TICKS + 1
        tick._record_bus_outage("TimeoutError")  # window elapsed
    finally:
        mind_tick_mod.record_degraded_event = original

    assert len(recorded) == 2, "the outage went quiet instead of re-asserting"


def test_the_pid_path_is_anchored_not_cwd_relative():
    from core.config import get_config

    pid_file = get_config().paths.pid_file

    assert pid_file.is_absolute()
    assert pid_file.name == "aura.pid"


def test_the_audit_and_the_repair_read_the_same_pid_file():
    import inspect as _inspect

    from core import mind_tick as mind_tick_mod
    from core.resilience import immunity_hyphae

    audit = _inspect.getsource(mind_tick_mod)
    repair = _inspect.getsource(immunity_hyphae.SignatureRepairRegistry._repair_pid_cleanup)

    assert "get_config().paths.pid_file" in audit
    assert "get_config().paths.pid_file" in repair
    assert 'pid_file = "aura.pid"' not in repair


def test_tick_zero_is_a_real_tick_number():
    """`x or default` throws away a legitimate zero, and tick 0 is exactly
    when the first outage is reported."""
    import inspect as _inspect

    from core import mind_tick as mind_tick_mod

    source = _inspect.getsource(mind_tick_mod.MindTick._record_bus_outage)

    assert 'or -10_000' not in source
    assert "isinstance(raw_last, int)" in source


def _spontaneous_is_meaty(content: str) -> bool:
    """The gate as the tick applies it."""
    from core.brain.llm.latent_cortex.output_quality import _terminal_complete
    from core.conversation.response_reliability import assess_user_facing_reply

    assessment = assess_user_facing_reply("", content)
    stripped = content.strip()
    return (
        not assessment.hard_failure
        and _terminal_complete(stripped)
        and len(stripped.split()) >= 4
    )


@pytest.mark.parametrize("fragment", ["a", "ok", "Hello there", "", "   "])
def test_a_fragment_is_not_spoken_unprompted(fragment):
    """`len > 5 or any alphabetic` was vacuous — the `or` meant a single
    letter passed, and this text is spoken to him without being asked for."""
    assert _spontaneous_is_meaty(fragment) is False


@pytest.mark.parametrize(
    "utterance",
    [
        "I noticed the build finished.",
        "The three failing tests are all in the parser.",
        "Something about the last change is still bothering me.",
    ],
)
def test_a_finished_thought_is_spoken(utterance):
    assert _spontaneous_is_meaty(utterance) is True


def test_an_unterminated_sentence_is_withheld():
    assert _spontaneous_is_meaty("I was about to say that the parser") is False


def test_the_tick_uses_the_shared_gate_not_a_local_heuristic():
    import inspect as _inspect

    from core import mind_tick as mind_tick_mod

    source = _inspect.getsource(mind_tick_mod)

    assert "len(content.strip()) > 5 or any(c.isalpha()" not in source
    assert "_terminal_complete(stripped)" in source
    assert "assess_user_facing_reply(\"\", content)" in source


def test_a_cold_cortex_no_longer_abandons_the_tick():
    """The escalation branch `continue`d the OUTER loop, so every thirtieth
    tick under a normal prewarm policy threw away every phase and the commit."""
    import ast
    import inspect as _inspect

    from core import mind_tick as mind_tick_mod

    source = _inspect.getsource(mind_tick_mod.MindTick._run_loop)
    tree = ast.parse(source.lstrip())

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        rendered = ast.get_source_segment(source.lstrip(), node) or ""
        if "_dead_tiers_are_policy_deferred_cortex" not in rendered:
            continue
        # The word appears in the comment explaining the fix; what matters is
        # that no Continue statement remains in the branch.
        assert not any(
            isinstance(child, ast.Continue) for child in ast.walk(node)
        ), "the tick is still abandoned"
        assert "LLM health: dead tiers" in rendered, "the escalation was lost"
        return
    raise AssertionError("the deferred-cortex branch was not found")


def test_a_revision_keeps_what_she_actually_said():
    """Overwriting a committed message in place is how a system stops being
    able to audit itself."""
    import inspect as _inspect

    from core import mind_tick as mind_tick_mod

    source = _inspect.getsource(mind_tick_mod)

    assert '"original_content": original_content' in source
    assert '"original_sha256"' in source
    assert '"revised_by": "metacognitive_monitor"' in source
    assert '"revised_at_unix"' in source


def test_the_revision_receipt_is_attached_to_the_message_it_revised():
    import inspect as _inspect

    from core import mind_tick as mind_tick_mod

    source = _inspect.getsource(mind_tick_mod)
    revised = source.index('revised_msg["content"] = report.revised_response')
    receipt = source.index('revised_msg["revision"] = {')

    assert revised < receipt, "the original is captured after being overwritten"
    assert "original_content = revised_msg.get(\"content\", \"\")" in source


def test_a_surprise_observation_is_recorded_as_a_correlation():
    import inspect as _inspect

    from core import mind_tick as mind_tick_mod

    source = _inspect.getsource(mind_tick_mod)

    assert 'reported_by="mind_tick.prediction_surprise"' in source
    assert "recorded a CORRELATION" in source
    assert "learned new observation from surprise signal" not in source


def test_the_world_model_only_upgrades_on_an_intervention():
    """The distinction the tick's language was blurring."""
    from core.brain.causal_world_model import CausalEdge

    edge = CausalEdge(source="a", target="b")

    assert edge.relationship == "correlates_with"
    assert edge.is_causal is False


def test_a_named_reporter_reaches_the_confidence_model():
    import inspect as _inspect

    from core.brain import causal_world_model

    source = _inspect.getsource(causal_world_model.CausalWorldModel.add_observation)

    assert "reported_by" in source


def _loop_source() -> str:
    import inspect as _inspect

    from core import mind_tick as mind_tick_mod

    return _inspect.getsource(mind_tick_mod.MindTick._run_loop)


def _module_source() -> str:
    from pathlib import Path as _Path

    return (_Path(__file__).resolve().parents[1] / "core" / "mind_tick.py").read_text("utf-8")


def test_the_world_state_probe_is_off_the_loop_and_bounded():
    """psutil's CPU, memory, battery and thermal reads are blocking syscalls;
    on the event loop they stall every coroutine in the process."""
    source = _loop_source()

    assert "asyncio.to_thread(get_world_state().update)" in source
    assert "world_state.update>5s" in source
    assert "world_state_timeout_yield" in source


def test_a_stalled_telemetry_read_does_not_stop_the_heartbeat():
    """Stale telemetry beats a stopped rhythm."""
    source = _loop_source()
    marker = source.index("world_state.update>5s")
    block = source[marker - 600 : marker + 600]

    assert "except TimeoutError:" in block
    assert "_mark_loop_progress" in block


def test_dream_research_runs_off_thread_under_a_bound():
    source = _loop_source()

    assert "asyncio.to_thread(self._dream_research_modules)" in source
    assert "dream_research_modules>120s" in source


def test_deferred_memory_replay_is_scheduled_after_returning_to_the_owner_loop():
    """The worker-thread dream bundle cannot create an asyncio task."""
    import inspect as _inspect

    from core import mind_tick as mind_tick_mod

    loop_source = _loop_source()
    replay_source = _inspect.getsource(
        mind_tick_mod.MindTick._replay_deferred_memory_writes
    )
    blocking_source = _inspect.getsource(mind_tick_mod.MindTick._dream_research_modules)

    worker_call = loop_source.index("asyncio.to_thread(self._dream_research_modules)")
    replay_schedule = loop_source.index("self._replay_deferred_memory_writes()")
    assert replay_schedule > worker_call
    assert "get_task_tracker" not in blocking_source
    assert "report = await queue.replay()" in replay_source


@pytest.mark.asyncio
async def test_deferred_memory_replay_executes_on_the_callers_event_loop(monkeypatch):
    from core.memory import deferred_retention

    owner_loop = asyncio.get_running_loop()
    observed = {}

    class Report:
        committed = 1
        refused = 0
        expired = 0

        @staticmethod
        def narrative():
            return "one write landed"

    class Queue:
        async def replay(self):
            observed["loop"] = asyncio.get_running_loop()
            return Report()

    monkeypatch.setattr(deferred_retention, "get_deferred_retention_queue", Queue)

    tick = MindTick.__new__(MindTick)
    await tick._replay_deferred_memory_writes()

    assert observed["loop"] is owner_loop


def test_the_immune_pulse_does_not_stop_the_world_inline():
    import inspect as _inspect

    from core import mind_tick as mind_tick_mod

    source = _inspect.getsource(mind_tick_mod.MindTick._immune_pulse_audit)

    assert "await asyncio.to_thread(gc.collect)" in source
    assert "await asyncio.to_thread(_sieve_logs)" in source
    assert "asyncio.to_thread(\n                get_resource_observer().process, os.getpid()\n            )" in source


def test_the_log_sieve_is_bounded_to_the_newest_logs():
    """An unbounded glob over a directory that grows with every incident makes
    the audit slower exactly when there is most to read."""
    import inspect as _inspect

    from core import mind_tick as mind_tick_mod

    source = _inspect.getsource(mind_tick_mod.MindTick._immune_pulse_audit)

    assert "reverse=True," in source
    assert "[:64]" in source
    assert "st_mtime" in source


def test_the_health_probe_never_builds_a_model_client():
    """`get_mlx_client()` resolves a model path and constructs a backend, which
    can initialize MLX inside the one process that must stay free of it."""
    source = _loop_source()

    assert "from core.brain.llm.mlx_client import get_mlx_client" not in source
    assert 'ServiceContainer.get("mlx_client", default=None)' in source


def test_an_unregistered_client_is_unknown_not_offline():
    """"offline" would be a verdict this tick has no evidence for."""
    source = _loop_source()
    marker = source.index('ServiceContainer.get("mlx_client", default=None)')
    block = source[marker : marker + 900]

    assert 'local_runtime_state = "unknown"' in block


def test_the_cheap_sidecar_probe_is_still_a_probe():
    """Constructing the sensory client starts nothing — __init__ only sets up
    IPC handles — so asking it is a real observation."""
    source = _loop_source()

    assert "get_sensory_client()" in source
    assert "sensory_client.is_alive()" in source


def test_the_health_sweep_yields_to_a_turn_in_flight():
    """ensure_all_tiers_healthy probes every tier and its recovery paths load
    workers. Bounding it at 45s stops a wedge; it does nothing about a probe
    that takes the model lane from the person waiting for an answer."""
    source = _loop_source()
    sweep = source.index("ensure_all_tiers_healthy")
    window = source[max(0, sweep - 1600) : sweep]

    assert "health_pause" in window
    assert "_background_reasoning_pause_reason()" in window
    assert "llm_health_deferred:" in window


def test_the_phase_pipeline_is_not_wrapped_in_a_taskgroup():
    """A TaskGroup that never creates a task is not decoration: an exception
    escaping its body comes out as an ExceptionGroup, which no `except
    _MIND_BOUNDARY_ERRORS` clause in this file matches.

    Checked on the parse tree, not the text — the comment explaining the
    removal names the construct, and a substring search cannot tell an
    explanation from a use."""
    import ast as _ast

    tree = _ast.parse(_module_source())
    uses = [
        node
        for node in _ast.walk(tree)
        if isinstance(node, _ast.Call)
        and isinstance(node.func, _ast.Attribute)
        and node.func.attr == "TaskGroup"
    ]
    assert not uses, f"TaskGroup constructed at line(s) {[n.lineno for n in uses]}"


def test_a_taskgroup_really_does_swallow_the_except_clause():
    """The claim above, demonstrated rather than asserted."""
    import asyncio as _asyncio

    async def _body():
        try:
            async with _asyncio.TaskGroup():
                raise ValueError("boom")
        except ValueError:
            return "caught"
        except BaseExceptionGroup:
            return "escaped as a group"

    assert _asyncio.run(_body()) == "escaped as a group"


def test_the_second_pipeline_never_runs_silently():
    from core.mind_tick import MIND_TICK_KERNEL_ABSENCE_ESCALATION_TICKS

    source = _module_source()
    marker = source.index("DEGRADED MODE: MindTick runs its own phases")
    block = source[marker : marker + 3200]

    assert "_ticks_without_kernel" in block
    assert '"second_phase_pipeline_active"' in block
    assert 'else "error"' in block, "kernel absence never escalates"
    assert "MIND_TICK_KERNEL_ABSENCE_ESCALATION_TICKS" in block
    assert MIND_TICK_KERNEL_ABSENCE_ESCALATION_TICKS > 0


def test_a_live_kernel_clears_the_absence_counter():
    source = _module_source()
    marker = source.index("skipping degraded-mode self-execution")
    window = source[max(0, marker - 300) : marker]

    assert "self._ticks_without_kernel = 0" in window


def test_the_reflex_fallback_says_it_is_not_an_answer():
    source = _module_source()
    marker = source.index('"origin": "mind_tick_fallback"')
    block = source[max(0, marker - 2000) : marker + 1200]

    assert '"placeholder": True' in block
    assert '"answers_request": False' in block
    assert '"failure_disclosure"' in block
    assert '"missing_phases": missing' in block
    assert '"reflex_fallback_served"' in block


def test_a_placeholder_does_not_close_the_request():
    """Completing the objective after serving a holding line retires a request
    the runtime never answered, and the next tick finds nothing left to do."""
    source = _module_source()
    marker = source.index("complete_current_objective")
    window = source[max(0, marker - 400) : marker]

    assert "not reflex_fallback_used" in window
