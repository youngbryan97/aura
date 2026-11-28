"""A goal of hers that needs no change to her source ends, saying what came of it (F16).

LIVE 27 November 2026 "Running a self-integrity scan" had been worked for 352 passes and 1,054 calls. Her model said
there was nothing left to verify. A goal ended only by offering a draft, so a goal that asked for a check went on being
worked each time her source moved: 266 of its steps that day were left unwritten and 289 calls refused as off its step.
"""
from __future__ import annotations

import asyncio

import pytest

from core.agency import her_goal_at_work as work
from core.brain.llm.her_steps import HerSteps
from core.skills import her_draft
from tests.test_her_goal_is_worked_by_her import _HerModelStandIn, her_source  # noqa: F401

pytestmark = pytest.mark.unit

SCAN = "Running a self-integrity scan"


def _pass(her_source, calls):
    return asyncio.run(work.work_a_while(SCAN, router=_HerModelStandIn([calls])))


def test_a_goal_that_needed_no_change_is_done_with_what_came_of_it(her_source):
    record = _pass(her_source, [("read_source", {"path": "core/notes.py"}),
                                ("nothing_to_change", {"what_came_of_it": "note_count reads as it should."})])
    assert record.status == work.DONE
    assert record.came_of_it["words"] == "note_count reads as it should."
    again = asyncio.run(work.work_a_while(SCAN, router=_HerModelStandIn([[("read_source", {"path": "core/notes.py"})]])))
    assert again.status == work.DONE and len(again.steps) == len(record.steps), "a done goal is not worked again"


def test_a_goal_whose_workspace_holds_a_change_cannot_end_without_it(her_source):
    record = _pass(her_source, [("edit_draft", {"path": "core/notes.py", "old_text": "return 0", "new_text": "return 1"}),
                                ("nothing_to_change", {"what_came_of_it": "Nothing to do."})])
    assert record.status == work.WORKING
    refused = [step for step in record.steps if step["tool"] == "nothing_to_change"]
    assert refused and "holds a change" in refused[0]["result"]


def test_her_task_engine_counts_a_goal_ended_without_a_change_as_done(her_source, monkeypatch):
    monkeypatch.setattr(work, "_done_otherwise", lambda goal: asyncio.sleep(0, result=None))
    _pass(her_source, [("read_source", {"path": "core/notes.py"})])

    async def ended(goal, *, router=None):
        record = work.load(work.slug_of(goal))
        record.status, record.came_of_it = work.DONE, {"words": "Clean.", "at": 1.0}
        await work.save(record)
        return record

    monkeypatch.setattr(work, "work_a_while", ended)
    result = asyncio.run(work.work_as_task(SCAN, "plan", "trace"))
    assert result.succeeded and "What came of it: Clean." in result.summary


def test_once_a_step_that_ends_her_way_came_back_done_no_step_is_left():
    tools = {name: {} for name in work.HER_WAY}
    steps = HerSteps.for_loop(tools, SCAN, order=work.HER_WAY, needs=work.HER_WAY_NEEDS, ends=work.HER_WAY_ENDS,
                              already=[("read_source", True), ("nothing_to_change", True)])
    assert steps.next() is None
    going = HerSteps.for_loop(tools, SCAN, order=work.HER_WAY, needs=work.HER_WAY_NEEDS, ends=work.HER_WAY_ENDS,
                              already=[("read_source", True), ("nothing_to_change", False)])
    assert going.next() == "what_went_wrong"


def test_her_way_ends_without_a_change_only_after_she_has_read_and_looked_up():
    way = list(work.HER_WAY)
    assert way.index("search_the_web") < way.index("nothing_to_change") < way.index("edit_draft")
    assert "nothing_to_change" in {skill.name for skill in her_draft.HER_TOOLS}


def test_a_pass_opens_with_her_goal_read_from_her_record_and_not_by_a_skill_run_outside_her_engine(her_source, monkeypatch):
    """LIVE 27 November 2026, 14:52, the opening read ran my_goal's skill outside her capability engine and was refused
    as ungoverned, so every pass opened with the refusal instead of her goal."""
    async def refused(self, params, context=None):
        raise AssertionError("not run as a skill here")

    monkeypatch.setattr(her_draft.MyGoalSkill, "execute", refused)
    record = work.GoalRecord(goal=SCAN, slug=work.slug_of(SCAN), passes=3)
    opening = asyncio.run(work.her_goal_as_she_reads_it(record))
    assert opening[0] == {"role": "user", "content": ""}
    assert '"goal": "Running a self-integrity scan"' in opening[2]["content"] and '"passes_so_far": 3' in opening[2]["content"]


def test_a_stalled_goal_is_woken_only_by_what_could_change_how_it_goes(her_source):
    """LIVE 27 November 2026 "Running a self-integrity scan", stalled, was woken by every commit to her source and
    each time spent two passes thinking 2,221 tokens and writing nothing."""
    from tests.test_her_goal_is_worked_by_her import _git

    def commit(path, text):
        (her_source / path).parent.mkdir(parents=True, exist_ok=True)
        (her_source / path).write_text(text)
        _git("-c", "user.name=t", "-c", "user.email=t@t", "add", path, cwd=her_source)
        _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", path, cwd=her_source)
        return _git("rev-parse", "HEAD", cwd=her_source).strip()

    record = work.GoalRecord(goal=SCAN, slug=work.slug_of(SCAN), status=work.STALLED,
                             stalled_on=_git("rev-parse", "HEAD", cwd=her_source).strip(),
                             steps=[{"tool": "edit_draft", "args": {"path": "core/notes.py"}}])
    assert not work.what_it_stalled_on_has_moved(record, commit("core/elsewhere.py", "x = 1\n"))
    assert work.what_it_stalled_on_has_moved(record, commit("core/notes.py", "def note_count():\n    return 1\n"))
    record.stalled_on = _git("rev-parse", "HEAD", cwd=her_source).strip()
    assert work.what_it_stalled_on_has_moved(record, commit("core/skills/her_draft.py", "# her tools\n"))


def test_a_goal_her_own_time_finds_nothing_of_hers_to_do_is_not_worked_in_her_source(her_source, monkeypatch):
    """LIVE 27 November 2026 somebody's words kept as a goal of hers, read as asking her to resume, were worked in her
    source three times, each pass saying sorry to nobody."""
    from types import SimpleNamespace

    from core.container import ServiceContainer

    async def nothing_does_it(goal):
        return None

    own = SimpleNamespace(act_of=nothing_does_it)
    real_get = ServiceContainer.get
    monkeypatch.setattr(ServiceContainer, "get",
                        classmethod(lambda cls, name, default=None: own if name == "her_own_time" else real_get(name, default)))

    async def never(goal, *, router=None):
        raise AssertionError("not worked in her source")

    monkeypatch.setattr(work, "work_a_while", never)
    result = asyncio.run(work.work_as_task("Stop interrupting me. I have asked three times now.", "plan", "trace"))
    assert not result.succeeded and "not worked there" in result.summary


def test_a_draft_doubted_behind_a_change_on_probation_waits_without_asking_its_gates_again(her_source, monkeypatch):
    """LIVE 27 November 2026 one draft was asked of every gate four times, half an hour each, behind a change on
    probation for 34 hours, and waited each time."""
    standing = work.GoalRecord(goal="A change on probation", slug="on-probation")
    standing.status, standing.offer = work.MERGED, {"on_probation": True, "commit": "c" * 40}
    asyncio.run(work.save(standing))
    draft = work.GoalRecord(goal="A doubted draft", slug="doubted-draft")
    draft.status, draft.offer = work.OFFERED, {"commit": "d" * 40, "files": ["core/notes.py"], "waited_behind": "on-probation"}
    asyncio.run(work.save(draft))
    asked: list[str] = []
    monkeypatch.setattr(work, "the_gates_her_yes_asks", lambda record: asked.append(record.slug) or {})
    monkeypatch.setattr(work, "source_head", lambda: "")
    waited = work.take_it_herself("doubted-draft")
    assert not waited["ok"] and "on probation" in waited["why"] and not asked
    standing.offer = {**standing.offer, "probation_over": 1.0}
    asyncio.run(work.save(standing))
    work.take_it_herself("doubted-draft")
    assert asked == ["doubted-draft"], "asked again once that probation is over"
