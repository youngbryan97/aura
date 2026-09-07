"""Delivery succeeded and semantic correctness was zero."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.verify.is_this_async_code_correct import (
    THE_CHECKS,
    is_it_correct,
    what_is_worth_a_look_in,
    what_is_wrong_with,
)

ROOT = Path(__file__).resolve().parents[1]

#: The code a live run wrote and delivered successfully. Every defect the
#: reviewer found in it, in one file.
THE_LIVE_FAILURE = '''
import asyncio


class Jobs:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.lock = asyncio.Lock()

    async def add(self, job):
        async with self.lock:
            self.queue._queue.append(job)
            item = await self.queue.get()
            return item

    async def run_all(self):
        for job in list(self.queue._queue):
            self.process(job)
        asyncio.sleep(0)
        # cancelling the worker releases the lock for us

    async def process(self, job):
        await asyncio.sleep(0)

    async def stop(self):
        try:
            await asyncio.sleep(1)
        except BaseException:
            pass
'''


@pytest.mark.parametrize(
    "kind",
    [
        "a coroutine created and dropped",
        "reaching inside a queue instead of using it",
        "a lock released by cancellation",
        "a bare except swallowing cancellation",
    ],
)
def test_every_defect_from_the_live_run_is_found(kind: str) -> None:
    found = {one.kind for one in what_is_wrong_with(THE_LIVE_FAILURE)}
    assert kind in found


def test_the_verdict_says_what_will_happen_rather_than_which_rule_broke() -> None:
    verdict = is_it_correct(THE_LIVE_FAILURE)
    assert verdict["correct"] is False
    assert verdict["what_to_say"]
    for one in verdict["mistakes"]:
        assert len(one["what_happens"].split()) > 8, one


def test_correct_async_code_is_left_alone() -> None:
    good = '''
import asyncio


class Jobs:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.lock = asyncio.Lock()

    async def add(self, job):
        async with self.lock:
            await self.queue.put(job)

    async def run_all(self):
        while not self.queue.empty():
            job = await self.queue.get()
            await self.process(job)

    async def process(self, job):
        await asyncio.sleep(0)

    async def stop(self):
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            raise
        except Exception:
            return None
'''
    assert what_is_wrong_with(good) == ()
    assert is_it_correct(good)["correct"]


def test_a_coroutine_defined_inside_a_lock_block_does_not_hold_it() -> None:
    """It is defined there and runs later; the lock is long gone."""
    code = '''
import asyncio

lock = asyncio.Lock()

async def go():
    async with lock:
        class Later:
            async def run(self):
                await self.thing.wait()
'''
    assert not [
        one
        for one in what_is_wrong_with(code)
        if one.kind == "a blocking await while holding a lock"
    ]


def test_an_exception_kept_for_a_caller_to_raise_is_not_swallowed() -> None:
    code = '''
def run(callback):
    errors = []
    try:
        return callback()
    except BaseException as exc:
        errors.append(exc)
    if errors:
        raise errors[0]
'''
    assert not [
        one
        for one in what_is_wrong_with(code)
        if one.kind == "a bare except swallowing cancellation"
    ]


def test_lock_free_is_a_claim_about_there_being_no_lock() -> None:
    code = "import asyncio\nl = asyncio.Lock()\n# the cancel channel is lock-free so nothing releases\n"
    assert not [
        one
        for one in what_is_wrong_with(code)
        if one.kind == "a lock released by cancellation"
    ]


def test_a_method_name_that_merely_matches_an_async_one_is_not_a_dropped_coroutine() -> None:
    """orch.semantic_defrag.start() is not this file's async start()."""
    code = '''
class A:
    async def start(self):
        await self.go()

def wire(orch):
    orch.something_else.start()
'''
    assert not [
        one
        for one in what_is_wrong_with(code)
        if one.kind == "a coroutine created and dropped"
    ]


def test_an_async_function_with_no_await_is_a_smell_and_not_a_defect() -> None:
    """253 of them in this tree, and they are interfaces rather than mistakes."""
    code = "async def ready(self):\n    return True\n"
    assert what_is_wrong_with(code) == ()
    assert [one.kind for one in what_is_worth_a_look_in(code)] == [
        "an async function that never awaits"
    ]


def test_unparsable_source_reports_nothing_rather_than_raising() -> None:
    assert what_is_wrong_with("async def (: broken") == ()


def test_the_tree_itself_is_clean() -> None:
    """Run over every file in core/. This is the number that must stay at zero."""
    found: list[str] = []
    for path in (ROOT / "core").rglob("*.py"):
        for one in what_is_wrong_with(path.read_text(encoding="utf-8", errors="ignore")):
            found.append(f"{path.relative_to(ROOT)}:{one.line} {one.what_happens}")
    assert found == [], "\n".join(found)


def test_the_checks_are_named_so_a_caller_can_say_what_it_checked() -> None:
    verdict = is_it_correct("x = 1")
    assert verdict["checked"] == list(THE_CHECKS)
    assert len(THE_CHECKS) == 4


def test_wait_under_lock_is_not_a_proven_dependency_cycle() -> None:
    code = '''
import asyncio
lock = asyncio.Lock()
queue = asyncio.Queue()
async def consume():
    async with lock:
        return await queue.get()
async def produce():
    await queue.put(1)
'''
    assert what_is_wrong_with(code) == ()
    assert any(
        item.kind == "a blocking await while holding a lock"
        for item in what_is_worth_a_look_in(code)
    )


def test_gather_schedules_work_without_awaiting_its_future() -> None:
    import asyncio

    source = '''
import asyncio
async def child():
    completed.set()
async def main():
    asyncio.gather(child())
    await completed.wait()
'''
    assert what_is_wrong_with(source) == ()
    namespace = {"completed": asyncio.Event()}
    # noqa: S102 — the checker's verdict is only worth anything if the
    # source it approved actually runs, so running it IS the assertion.
    exec(source, namespace)  # noqa: S102
    asyncio.run(namespace["main"]())
    assert namespace["completed"].is_set()


def test_the_generator_checks_what_it_wrote_before_returning_it() -> None:
    """Wired, not beside it: the check is on the path generated code takes."""
    import asyncio

    from core.brain.llm.code_generator import LLMCodeGenerator

    wrong = (
        "import asyncio\n\n\n"
        "async def go():\n"
        "    await asyncio.sleep(0)\n\n\n"
        "async def run():\n"
        "    go()\n"
    )

    class _Router:
        async def generate(self, request):
            return f"```python\n{wrong}```"

    made = LLMCodeGenerator(router=_Router())
    code = asyncio.run(made.generate_async("write it", {"module_path": "x.py"}))
    assert "async def run" in code, "the code is still returned"
    found = {one.kind for one in made.last_async_findings}
    assert "a coroutine created and dropped" in found, (
        "the generator handed back code without checking it"
    )


def test_correct_generated_code_leaves_no_findings() -> None:
    import asyncio

    from core.brain.llm.code_generator import LLMCodeGenerator

    right = (
        "import asyncio\n\n\n"
        "async def go():\n"
        "    await asyncio.sleep(0)\n\n\n"
        "async def run():\n"
        "    await go()\n"
    )

    class _Router:
        async def generate(self, request):
            return f"```python\n{right}```"

    made = LLMCodeGenerator(router=_Router())
    asyncio.run(made.generate_async("write it", {"module_path": "x.py"}))
    assert made.last_async_findings == ()
