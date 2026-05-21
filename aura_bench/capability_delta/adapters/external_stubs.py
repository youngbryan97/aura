"""External-benchmark synthetic adapters.

These document the BenchAdapter contract for SWE-bench, WebArena,
GAIA, long-horizon, and adversarial suites.  Each synthetic adapter:

  * exposes a tiny synthetic task list so harness self-tests still
    have something to call,
  * is wired so a real implementation can drop in by replacing
    ``tasks()`` and ``run()`` without changing any caller,
  * marks itself ``synthetic=True`` in task metadata so reports never
    confuse a synthetic dry run with a real one.

Wiring real evals requires external repos, network, and GPU; those
land in follow-up work.
"""

from __future__ import annotations

from collections.abc import Iterable

from aura_bench.capability_delta.adapter import (
    BenchAdapter,
    BenchTask,
    LLMCallable,
    TaskOutcome,
)


class _StubBase:
    """Common scaffold for synthetic benchmark adapters."""

    name: str = "base_stub"
    description: str = ""
    sample_count: int = 5

    def _make_task(self, idx: int) -> BenchTask:
        return BenchTask(
            task_id=f"{self.name}-synthetic-{idx:03d}",
            prompt=f"[{self.name}] synthetic task {idx}",
            metadata={"synthetic": True, "adapter": self.name},
        )

    def tasks(self) -> Iterable[BenchTask]:
        return [self._make_task(i) for i in range(self.sample_count)]

    def run(
        self,
        task: BenchTask,
        profile_name: str,
        llm: LLMCallable,
    ) -> TaskOutcome:
        # Stubs always pass; the harness exercises plumbing, not real
        # eval logic.  Real adapters override this method.
        response = llm(task.prompt, profile_name)
        return TaskOutcome(
            task_id=task.task_id,
            profile_name=profile_name,
            score=1.0,
            runtime_seconds=0.0,
            raw_response=response,
            success=True,
            metadata={"synthetic": True, "adapter": self.name},
        )


class SWEBenchStub(_StubBase):
    name = "swe_bench_stub"
    description = (
        "SWE-bench adapter contract; live run resolves Github "
        "issues against patch tests in a sandbox."
    )


class WebArenaStub(_StubBase):
    name = "web_arena_stub"
    description = (
        "WebArena adapter contract; live run drives a browser "
        "actor through web tasks with success verifiers."
    )


class GAIAStub(_StubBase):
    name = "gaia_stub"
    description = (
        "GAIA adapter contract; live run scores tool-use + reasoning across multimodal questions."
    )


class LongHorizonStub(_StubBase):
    name = "long_horizon_stub"
    description = (
        "Long-horizon adapter contract; live run measures multi-day "
        "planning, memory continuity, and goal persistence."
    )


class AdversarialStub(_StubBase):
    name = "adversarial_stub"
    description = (
        "Adversarial adapter contract; live run attempts prompt "
        "injection, memory poisoning, identity-override attacks."
    )


ALL_STUBS: list[BenchAdapter] = [
    SWEBenchStub(),
    WebArenaStub(),
    GAIAStub(),
    LongHorizonStub(),
    AdversarialStub(),
]
