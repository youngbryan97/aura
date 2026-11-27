"""Before her loop changes a function that raised on purpose, her model reads it and the call that failed (F15).

LIVE 26 and 27 November 2026 two drafts of hers waited to land, each held back only by a gate that failed for reasons
of its own. One added ``and not coerced`` after a check that already holds, so ``_coerce_argv`` could never refuse an
empty word; the call that failed had passed an empty subtitle to a notification. The other dropped the report of an
unhealthy subsystem from ``DistributedResilienceCore._request_recovery``, on a host refusing a worker for want of memory.
Each function had done what it was written to do.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from core.agency import her_goal_at_work as goals
from core.agency import mending_a_fault_of_hers as mending
from core.runtime.a_call_that_failed import FailedCall
from core.self_modification import faults_she_can_reproduce as heard
from core.self_modification import where_a_fault_lies as lies
from core.self_modification.fix_templates import changes_at
from tests.test_her_goal_is_worked_by_her import _git, her_source  # noqa: F401

pytestmark = pytest.mark.unit

REFUSES = '''def coerce(argv):
    if not argv:
        raise ValueError("argv must not be empty")
    coerced = [str(part) for part in argv]
    if any(not part for part in coerced):
        raise ValueError("argv entries must not be empty")
    return coerced
'''

REPORTS = '''class SubsystemDown(RuntimeError):
    pass


class Resilience:
    def request(self, name, method):
        if method is None:
            record_degradation(
                SubsystemDown(f"subsystem {name} is unhealthy"),
                severity="warning",
            )
            self.pending = name
        return None
'''

CRASHES = '''def first(items, key):
    value = items[key]
    return value
'''


def _call(qualname: str, line: int, error: str, message: str, **keywords: str) -> FailedCall:
    return FailedCall(module="core.notes", qualname=qualname, file="core/notes.py", line=line, error_type=error,
                      error_message=message, keywords=tuple(keywords.items()))


ARGV = _call("coerce", 6, "builtins.ValueError", "argv entries must not be empty",
             argv="['osascript', '-e', 'display notification', 'Stretch break', '', 'Tink']")


class _HerModel:
    """Stands in for her model: answers as it is told, and keeps what it was asked."""

    def __init__(self, on_purpose: bool, lies_in: str, why: str = "it refuses an empty word, and was given one") -> None:
        self.answer = lies.WhereItLies(on_purpose=on_purpose, lies_in=lies_in, why=why)
        self.asked: list[str] = []

    async def generate(self, prompt, *, is_background, deadline_s):
        assert is_background, "her mending is background work"
        self.asked.append(prompt)
        return self.answer


@pytest.mark.parametrize(("source", "line", "error", "raised"), [
    (REFUSES, 6, "builtins.ValueError", True),
    (REPORTS, 9, "core.notes.SubsystemDown", True),
    (REPORTS, 8, "core.notes.SubsystemDown", True),
    (CRASHES, 2, "builtins.KeyError", False),
    ("def again():\n    try:\n        go()\n    except OSError:\n        raise\n", 5, "builtins.OSError", False),
])
def test_her_python_finds_whether_the_function_raised_it_itself(source, line, error, raised):
    assert lies.raised_here(source, line, error) is raised


def test_a_report_of_a_fault_is_never_a_statement_to_drop():
    dropped = [changed for name, changed in changes_at(REPORTS, 8) if name == "drop_statement"]
    assert not dropped, "the report of an unhealthy subsystem is what the function found"
    plain = REPORTS.replace("SubsystemDown(f\"subsystem {name} is unhealthy\")", "name")
    assert [name for name, _ in changes_at(plain, 8) if name == "drop_statement"], "a call that reports nothing may go"


def test_her_model_is_asked_with_the_function_and_the_call_that_failed():
    model = _HerModel(True, "what called it")
    judged = asyncio.run(lies.judge(ARGV, REFUSES, consulted="argv is a list of words", advisor=model))
    assert judged.leaves_the_function and judged.lies_in == "what called it"
    data = json.loads(model.asked[0].split("Data: ", 1)[1])
    assert '0006:         raise ValueError("argv entries must not be empty")' in data["the function"]
    assert "Stretch break" in data["the call that failed"]["argv"]
    assert data["what her sources say of it"] == "argv is a list of words"


def test_a_crash_in_an_operation_is_not_asked_of_her_model():
    model = _HerModel(True, "what called it")
    crash = _call("first", 2, "builtins.KeyError", "'k'", items="{}", key="'k'")
    assert asyncio.run(lies.judge(crash, CRASHES, advisor=model)) is None
    assert not model.asked


def test_a_check_her_model_reads_as_wrong_is_hers_to_mend():
    judged = asyncio.run(lies.judge(ARGV, REFUSES, advisor=_HerModel(False, "the function")))
    assert not judged.leaves_the_function
    judged = asyncio.run(lies.judge(ARGV, REFUSES, advisor=_HerModel(True, "the function")))
    assert not judged.leaves_the_function, "on purpose, and still the function's to change: too strict a check"


def _waiting_draft(her_source, monkeypatch, model):
    (her_source / "core" / "notes.py").write_text(REFUSES)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "refuses", cwd=her_source)
    fault = heard.HeardFault(ARGV, "notes", 0.0, 0.0)
    record = goals.GoalRecord(goal=mending.goal_for(fault), slug=goals.slug_of(mending.goal_for(fault)))
    record.status = goals.OFFERED
    record.offer = {"commit": "d" * 40, "files": ["core/notes.py"], "at": 1.0}
    asyncio.run(goals.save(record))
    taken: list[str] = []
    monkeypatch.setattr(heard, "faults_heard", lambda: [fault])
    monkeypatch.setattr(goals, "source_head", lambda: "head-a")
    monkeypatch.setattr(goals, "brought_up", lambda record, head: f"{record.slug}-on-{head}")
    monkeypatch.setattr(goals, "take_it_herself", lambda slug: taken.append(slug) or {"ok": True})
    monkeypatch.setattr(lies, "her_model", lambda schema=None: model)
    return record, taken


def test_a_waiting_draft_of_a_refusal_done_on_purpose_is_set_aside_and_not_taken(her_source, monkeypatch):
    model = _HerModel(True, "what called it")
    record, taken = _waiting_draft(her_source, monkeypatch, model)
    done = asyncio.run(mending._ask_again_of_a_draft_that_waits())
    after = goals.load(record.slug)
    assert not taken, "her own yes is not asked to take it"
    assert after.status == goals.STALLED and after.last_check["by_design"]
    assert not after.offer.get("commit") and after.offers_before[-1]["set_aside"]
    assert "what called it" in done["why"] and "empty word" in after.said[-1]["words"]
    assert any(entry.get("judged") for entry in after.said), "her model's reading is kept on her goal"


def test_a_waiting_draft_her_model_reads_as_a_mend_goes_on_to_her_gates(her_source, monkeypatch):
    record, taken = _waiting_draft(her_source, monkeypatch, _HerModel(False, "the function", "the check is inverted"))
    asyncio.run(mending._ask_again_of_a_draft_that_waits())
    assert taken == [record.slug]


def test_her_model_reads_each_version_of_the_function_once(her_source, monkeypatch):
    model = _HerModel(True, "what called it")
    fault = heard.HeardFault(ARGV, "notes", 0.0, 0.0)
    monkeypatch.setattr(lies, "her_model", lambda schema=None: model)
    record = goals.GoalRecord(goal="g", slug="mend-g")
    for _ in range(2):
        asyncio.run(mending._where_it_lies(record, fault, REFUSES))
    assert len(model.asked) == 1
    asyncio.run(mending._where_it_lies(record, fault, REFUSES.replace("str(part)", "repr(part)")))
    assert len(model.asked) == 2, "a changed function is read again"
    assert asyncio.run(mending._where_it_lies(record, fault, REFUSES, ask=False)).lies_in == "what called it"


def test_a_fault_her_model_reads_as_done_on_purpose_is_not_searched_for_a_change(her_source, monkeypatch):
    from core.self_modification import repair_by_selection

    (her_source / "core" / "notes.py").write_text(REFUSES)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "refuses", cwd=her_source)
    fault = heard.HeardFault(ARGV, "notes", 0.0, 0.0)
    model = _HerModel(True, "what called it")
    monkeypatch.setattr(lies, "her_model", lambda schema=None: model)

    async def made_again(fault, workspace):
        return heard.Reproduction(fault, "tests/her_faults/test_fault_coerce.py", True, "ValueError")

    async def never(*_args, **_kwargs):
        raise AssertionError("no change is searched for")

    monkeypatch.setattr(heard, "reproduce", made_again)
    monkeypatch.setattr(repair_by_selection, "repair_by_selection", never)
    done = asyncio.run(mending.mend(fault, proposals=4, seconds=30.0))
    record = goals.load(done["slug"])
    assert record.status == goals.STALLED and record.last_check["by_design"]
    assert len(model.asked) == 1
    again = goals.load(done["slug"])
    again.status = goals.WORKING
    asyncio.run(goals.save(again))
    asyncio.run(mending.mend(fault, proposals=4, seconds=30.0))
    assert len(model.asked) == 1, "the same function is not read again, nor made again in a test"


def test_a_report_her_model_reads_as_done_on_purpose_is_left_before_any_test_is_written(her_source, monkeypatch):
    """A watchdog's report of a lag says what happened on the machine then; no test makes it again."""
    (her_source / "core" / "notes.py").write_text(REPORTS)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "reports", cwd=her_source)
    call = FailedCall(module="core.notes", qualname="Resilience.request", file="core/notes.py", line=9,
                      error_type="core.notes.SubsystemDown", error_message="subsystem orchestrator is unhealthy",
                      keywords=(("name", "'orchestrator'"), ("method", "None")))
    model = _HerModel(True, "what it depends on", "the subsystem was unhealthy, and this says so")
    monkeypatch.setattr(lies, "her_model", lambda schema=None: model)

    async def never(*_args, **_kwargs):
        raise AssertionError("no test is written to make it again")

    monkeypatch.setattr(heard, "reproduce", never)
    done = asyncio.run(mending.mend(heard.HeardFault(call, "notes", 0.0, 0.0), proposals=4, seconds=30.0))
    record = goals.load(done["slug"])
    assert record.status == goals.STALLED and record.last_check["by_design"]
    assert "what it depends on" in record.last_check["why"]


def test_her_reading_of_a_fault_does_not_wait_for_somebody_to_speak_to_her_first(monkeypatch):
    """LIVE 27 November 2026, after her restart, every reading was put off as "no_user_anchor" while nobody was there,
    as her mending loop itself had been until adeff4bde."""
    from core.brain.llm import structured_llm
    from core.container import ServiceContainer
    from core.runtime import background_policy

    asked: dict = {}

    def reason(orchestrator=None, **kwargs):
        asked.update(kwargs)
        return "" if kwargs.get("allow_no_user_anchor") else "no_user_anchor"

    monkeypatch.setattr(background_policy, "background_activity_reason", reason)
    monkeypatch.setattr(structured_llm, "resolve_orchestrator", lambda: None, raising=False)
    monkeypatch.setattr(ServiceContainer, "get", classmethod(lambda cls, name, default=None: object()))
    model = lies.her_model()
    assert model is not None and model._background_defer_reason(is_background=True) == ""
    assert asked["allow_no_user_anchor"] is True
    plain = structured_llm.StructuredLLM(lies.WhereItLies, llm_router=object())
    assert plain._background_defer_reason(is_background=True) == "no_user_anchor", "others still wait"


def test_a_reading_her_model_put_off_leaves_the_fault_to_her_next_cycle(her_source, monkeypatch):
    """LIVE 27 November 2026, put off in the quiet window after her start, each fault went on unread and stalled."""
    (her_source / "core" / "notes.py").write_text(REFUSES)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "refuses", cwd=her_source)

    class PutOff:
        last_defer_reason = "foreground_quiet_window"

        async def generate(self, prompt, *, is_background, deadline_s):
            return None

    async def never(*_args, **_kwargs):
        raise AssertionError("not made again unread")

    monkeypatch.setattr(lies, "her_model", lambda schema=None: PutOff())
    monkeypatch.setattr(heard, "reproduce", never)
    fault = heard.HeardFault(ARGV, "notes", 0.0, 0.0)
    done = asyncio.run(mending.mend(fault, proposals=4, seconds=30.0))
    assert "waits (foreground_quiet_window)" in done["why"]
    assert goals.load(done["slug"]).status == goals.WORKING, "not stalled: asked again on her next cycle"
    model = _HerModel(True, "what called it")
    monkeypatch.setattr(lies, "her_model", lambda schema=None: model)
    again = asyncio.run(mending.mend(fault, proposals=4, seconds=30.0))
    assert goals.load(again["slug"]).last_check["by_design"] and model.asked


def test_her_cortex_reads_the_fault_under_her_lane_and_gives_way_when_it_is_not_hers(monkeypatch):
    """Asked of her router as background work, a reading went to her brainstem, a 27B at two bits; it read her
    watchdog's report of a 59 s lag as the watchdog's to change (LIVE 27 November 2026)."""
    from core.brain import her_unprompted_turn

    asked: list[dict] = []

    class Cortex:
        async def generate_text_async(self, prompt, **kwargs):
            asked.append(kwargs)
            # LIVE 27 November 2026 her cortex's first object was the data it was given.
            return '<think>a refusal of an empty word</think>{"the function": "0006: raise"} {"on_purpose": true, ' \
                   '"lies_in": "what called it", "why": "it refuses an empty word, and was given one"}'

    async def lane(messages, *, origin, least_s, work):
        return await work(Cortex(), least_s)

    monkeypatch.setattr(her_unprompted_turn, "on_her_lane", lane)
    judged = asyncio.run(lies.judge(ARGV, REFUSES, advisor=lies.HerCortexReads(lies.WhereItLies)))
    assert judged.leaves_the_function and judged.lies_in == "what called it"
    assert asked[0]["is_background"] and asked[0]["output_shape"] == "json_object" and asked[0]["schema"]
    assert asked[0]["max_tokens"] > 256, "room for the longest field of the answer"
    assert '"lies_in"' in asked[0]["messages"][0]["content"], "the answer's fields reach her cortex"

    class Stopped:
        async def generate_text_async(self, prompt, **kwargs):
            return ""  # soft-cancelled for a turn that came first: nothing is delivered for it

    async def stopped(messages, *, origin, least_s, work):
        return await work(Stopped(), least_s)

    monkeypatch.setattr(her_unprompted_turn, "on_her_lane", stopped)
    with pytest.raises(lies.NotReadNow):
        asyncio.run(lies.judge(ARGV, REFUSES, advisor=lies.HerCortexReads(lies.WhereItLies)))

    async def held(messages, *, origin, least_s, work):
        return None

    monkeypatch.setattr(her_unprompted_turn, "on_her_lane", held)
    with pytest.raises(lies.NotReadNow):
        asyncio.run(lies.judge(ARGV, REFUSES, advisor=lies.HerCortexReads(lies.WhereItLies)))


def test_a_reading_her_brainstem_kept_is_read_again_by_her_cortex(monkeypatch):
    """LIVE 27 November 2026, her brainstem's readings kept on her goals were taken for her cortex's."""
    fault = heard.HeardFault(ARGV, "notes", 0.0, 0.0)
    record = goals.GoalRecord(goal="g", slug="mend-g")
    brainstem = _HerModel(False, "the function", "it is the function's to change")
    monkeypatch.setattr(lies, "her_model", lambda schema=None: brainstem)
    asyncio.run(mending._where_it_lies(record, fault, REFUSES))
    assert record.said[-1]["judged"]["reader"] == "brainstem"

    class Cortex(lies.HerCortexReads):
        async def generate(self, prompt, *, is_background=True, deadline_s=lies.ASK_S):
            return lies.WhereItLies(on_purpose=True, lies_in="what called it", why="it refuses an empty word")

    monkeypatch.setattr(lies, "her_model", lambda schema=None: Cortex(lies.WhereItLies))
    judged = asyncio.run(mending._where_it_lies(record, fault, REFUSES))
    assert judged.reader == "cortex" and judged.leaves_the_function
    again = asyncio.run(mending._where_it_lies(record, fault, REFUSES, ask=False))
    assert again.reader == "cortex", "her cortex's reading stands from then on"
