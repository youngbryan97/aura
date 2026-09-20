"""A boot step that is still completing its parts is working, however slowly.

LIVE 2026-09-16, load 34 on 18 cores, boot offset +286s: the live-mind
activation materialized all eight organs in 17s. Its wall budget was 15s. The
boot raised TimeoutError with the work done and the desktop process exited.

The bound is now on progress. The budget is the longest one part may take to
complete; a host that makes every part slow never trips it, and a part that
never completes still does.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from core.runtime.progress_bound import await_while_it_progresses

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.asyncio
async def test_a_step_that_keeps_completing_parts_outlives_the_budget():
    steps: list[int] = []

    async def slow_but_working():
        for i in range(6):
            await asyncio.sleep(0.15)
            steps.append(i)
        return "done"

    # Six parts at 0.15s is 0.9s of wall against a 0.4s budget.
    result = await await_while_it_progresses(
        slow_but_working(), progress=lambda: len(steps), stall_s=0.4, name="six parts"
    )
    assert result == "done"
    assert steps == list(range(6))


@pytest.mark.asyncio
async def test_a_step_whose_parts_stop_completing_is_a_wedge():
    steps: list[int] = []

    async def wedges_after_two():
        for i in range(2):
            await asyncio.sleep(0.05)
            steps.append(i)
        await asyncio.sleep(60)
        return "never"

    with pytest.raises(TimeoutError, match=r"wedge made no progress for .*last progress: 2"):
        await await_while_it_progresses(
            wedges_after_two(), progress=lambda: len(steps), stall_s=0.3, name="wedge"
        )
    assert steps == [0, 1]


def test_the_live_mind_activation_is_bounded_by_its_progress():
    source = (ROOT / "aura_main.py").read_text(encoding="utf-8")
    block = source[source.index("activate_live_mind_runtime") :]
    block = block[: block.index("_mark_runtime_boot_phase(\"live_mind_activation\")")]
    assert "await_while_it_progresses(" in block
    assert "progress=get_live_mind_runtime().materialized" in block
    assert not re.search(r"wait_for\(\s*asyncio\.to_thread\(activate_live_mind_runtime\)", block)

    from core.runtime.live_mind_runtime import LiveMindRuntime

    runtime = LiveMindRuntime()
    assert runtime.materialized() == ""


@pytest.mark.asyncio
async def test_a_stage_working_through_synchronous_steps_outlives_the_budget():
    """The kernel stage: organ after organ, synchronous on the loop thread,
    106s on a loaded host against a 15s wall budget."""
    import time

    from core.runtime.progress_bound import await_while_the_task_moves

    organs: list[str] = []

    async def kernel_like():
        for organ in ("llm", "vision", "memory", "affect", "will", "mesh"):
            time.sleep(0.12)  # synchronous work on the loop thread
            organs.append(organ)
            await asyncio.sleep(0)
        return "kernel ready"

    # 0.72s of blocking work against a 0.3s stall budget.
    result = await await_while_the_task_moves(kernel_like(), stall_s=0.3, name="kernel")
    assert result == "kernel ready"
    assert len(organs) == 6


@pytest.mark.asyncio
async def test_a_stage_awaiting_something_that_never_resolves_is_a_wedge():
    from core.runtime.progress_bound import await_while_the_task_moves

    never = asyncio.get_running_loop().create_future()

    async def wedged():
        await asyncio.sleep(0.05)
        await never
        return "never"

    with pytest.raises(TimeoutError, match=r"boot stage x sat on one await"):
        await await_while_the_task_moves(wedged(), stall_s=0.3, name="boot stage x")


def test_the_boot_stages_are_bounded_by_their_motion():
    source = (ROOT / "core/ops/resilient_boot.py").read_text(encoding="utf-8")
    assert "await_while_the_task_moves(\n                        stage_fn(), stall_s=timeout" in source
    assert "asyncio.wait_for(stage_fn()" not in source


def test_the_provenance_git_queries_are_bounded_by_their_work():
    """2026-09-16: a boot died in its provenance snapshot when
    ``git symbolic-ref --short HEAD`` took more than 3.0s of wall on a host
    where another agent's git held a core. The query costs milliseconds of
    CPU; the bound is on that."""
    source = (ROOT / "core/runtime/launch_provenance.py").read_text(encoding="utf-8")
    git = source[source.index("def _run_git(") : source.index("def _status_paths(")]
    assert "run_until_its_work_is_done(" in git
    assert "cpu_budget_s=" in git
    assert "timeout=3.0" not in git


@pytest.mark.host_observation  # real children, read through the host observer
def test_the_gateway_runner_stops_a_wedged_child_and_a_busy_one():
    import os
    import sys
    import time

    from core.runtime.subprocess_gateway import get_subprocess_gateway

    gateway = get_subprocess_gateway()
    started = time.monotonic()
    wedged = gateway.run_until_its_work_is_done(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        cpu_budget_s=1.0,
        cwd=os.getcwd(),
        read_only=True,
        source="test.progress_bound.wedged",
        accelerator_capability="none",
        watch_period_s=0.2,
    )
    assert wedged.returncode == 124 and wedged.stderr.startswith("wedged:")
    assert time.monotonic() - started < 30.0

    done = gateway.run_until_its_work_is_done(
        [sys.executable, "-c", "print('ok')"],
        cpu_budget_s=5.0,
        cwd=os.getcwd(),
        read_only=True,
        source="test.progress_bound.done",
        accelerator_capability="none",
        watch_period_s=0.2,
    )
    assert done.returncode == 0 and done.stdout.strip() == "ok"


# ── a probe on a thread is bounded by the thread's own work ─────────────────


def _burn_cpu(seconds: float) -> str:
    import threading
    import time

    from core.runtime.thread_cpu import thread_cpu_seconds

    ident = threading.get_ident()
    start = thread_cpu_seconds(ident)
    if start is None:  # a platform with no per-thread clock: burn wall instead
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            sum(range(1000))
        return "done"
    while thread_cpu_seconds(ident) - start < seconds:
        sum(range(1000))
    return "done"


@pytest.mark.asyncio
async def test_a_probe_still_using_the_cpu_outlives_its_stall_budget():
    """2026-09-16: control-plane probes and Skynet health probes failed by the
    dozen on a host loaded past 30, while still running their lines."""
    from core.runtime.progress_bound import run_on_a_thread_while_it_works

    assert await run_on_a_thread_while_it_works(_burn_cpu, 0.5, stall_s=0.2, name="busy") == "done"


@pytest.mark.asyncio
async def test_a_probe_that_stopped_working_is_a_stall():
    import time

    from core.runtime.progress_bound import run_on_a_thread_while_it_works

    with pytest.raises(TimeoutError, match="idle made no progress"):
        await run_on_a_thread_while_it_works(time.sleep, 1.0, stall_s=0.2, name="idle")


@pytest.mark.asyncio
async def test_a_scheduled_task_that_keeps_moving_outlives_its_budget():
    from core.scheduler import Lifecycle, Scheduler, TaskSpec

    async def _reconcile_many_bindings() -> None:
        for _ in range(8):
            await asyncio.sleep(0.05)

    scheduler = Scheduler()
    spec = TaskSpec(name="reconcile", coro=_reconcile_many_bindings, tick_interval=5.0, timeout_s=0.2, critical=True)
    await scheduler.register(spec)
    await scheduler._run_task(spec)

    assert scheduler.get_health()["task_details"][spec.name]["status"] == "ok"
    assert scheduler.state is not Lifecycle.RECOVERING


def test_every_probe_on_a_thread_is_bounded_by_its_work():
    """The whole class, not one site: each of these read a status or a
    snapshot on a worker thread under a fixed wall budget."""
    bounded = {
        "core/scheduler.py": "await_while_the_task_moves(",
        "core/runtime/control_plane.py": "run_on_a_thread_while_it_works(callback",
        "core/fictional/skynet.py": "run_on_a_thread_while_it_works(",
        "core/fictional/mist.py": "run_on_a_thread_while_it_works(",
        "core/autonomic/allostasis.py": "run_on_a_thread_while_it_works(",
        "core/mind_tick.py": "run_on_a_thread_while_it_works(",
        "core/brain/llm_health_router.py": "run_on_a_thread_while_it_works(",
        "core/kernel/organs.py": "run_on_a_thread_while_it_works(",
    }
    from pathlib import Path as _Path

    for rel, call in bounded.items():
        # the module and every module lifted out of it (mind_tick's loop
        # steps live in mind_tick_loop_steps now)
        own = ROOT / rel
        family = [own, *sorted(_Path(own.parent).glob(f"{own.stem}_*.py"))]
        source = "\n".join(path.read_text(encoding="utf-8") for path in family)
        assert call in source, rel
    for rel in ("core/fictional/skynet.py", "core/runtime/control_plane.py", "core/kernel/organs.py"):
        source = (ROOT / rel).read_text(encoding="utf-8")
        assert not re.search(r"wait_for\(\s*asyncio\.to_thread", source), rel


# ── a save a coroutine waits for runs off the loop, inline once the lane is gone


@pytest.mark.asyncio
async def test_off_the_loop_runs_the_call_on_a_worker_and_returns_its_result():
    import threading

    from core.runtime.executors import off_the_loop

    loop_thread = threading.current_thread().name
    where = await off_the_loop(lambda: threading.current_thread().name)
    assert where != loop_thread
    assert await off_the_loop(lambda a, b=1: a + b, 2, b=3) == 5


@pytest.mark.asyncio
async def test_off_the_loop_runs_inline_once_the_executor_is_gone(monkeypatch):
    """The last saves of a shutdown must still happen when the loop's
    executor has already been shut down."""
    import asyncio

    from core.runtime import executors

    async def gone(*args, **kwargs):
        raise RuntimeError("cannot schedule new futures after shutdown")

    monkeypatch.setattr(executors.asyncio, "to_thread", gone)
    ran: list[str] = []
    assert await executors.off_the_loop(lambda: ran.append("saved") or "ok") == "ok"
    assert ran == ["saved"]

    async def broken(*args, **kwargs):
        raise RuntimeError("something else")

    monkeypatch.setattr(executors.asyncio, "to_thread", broken)
    with pytest.raises(RuntimeError, match="something else"):
        await executors.off_the_loop(lambda: "never")


def test_the_named_shutdown_saves_are_off_the_loop():
    """The loop report of 2026-09-19 named these from stop() coroutines and
    the shutdown itself; each is awaited off the loop now."""
    sites = {
        "core/consciousness/mhaf_field.py": "await off_the_loop(self._save)",
        "core/adaptation/epistemic_humility.py": "await off_the_loop(self._save)",
        "core/introspection/insight_journal.py": "await off_the_loop(self._save)",
        "core/epistemics/inquiry_engine.py": "await off_the_loop(self._save)",
        "core/evolution/evolution_orchestrator.py": "await off_the_loop(self._save)",
        "core/orchestrator/handlers/shutdown.py": "await off_the_loop(orch._save_state, \"shutdown\")",
        "core/learning/selfplay_flywheel.py": "await off_the_loop(self._save_state, state)",
        "core/brain/cognitive/integrity_check.py": "await off_the_loop(self._write_audit_log, report)",
        "core/orchestrator/mixins/boot/boot_cognitive.py": "await off_the_loop(get_live_learner)",
    }
    for rel, call in sites.items():
        assert call in (ROOT / rel).read_text(encoding="utf-8"), rel


def test_a_degradation_receipt_is_written_behind_the_loop() -> None:
    """record_degradation(receipt_required=True) writes a durable receipt — an
    atomic write, an fsync and a chain append. From an exception handler in
    async code that ran on the loop (LIVE 2026-09-20, named by the loop report
    from the code generator). On the loop it is queued behind it."""
    import asyncio

    import core.runtime.errors as errors

    handed: list[str] = []

    def fake_behind(key: str, fn):
        handed.append(key)
        return False

    async def on_the_loop() -> None:
        with pytest.MonkeyPatch.context() as mp:
            import core.runtime.executors as executors

            mp.setattr(executors, "behind_the_loop", fake_behind)
            errors.record_degradation(
                "a_subsystem", RuntimeError("x"), action="tested", receipt_required=True
            )

    asyncio.run(on_the_loop())
    assert handed and handed[0].startswith("degradation_receipt:a_subsystem:")
