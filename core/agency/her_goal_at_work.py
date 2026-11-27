"""A goal she set herself, worked on by her: her model chooses each next step as a call to one of her tools, a few
calls at a time, for as long as the goal takes.

Why this exists. On 8 November 2026 she told Bryan she was "working on some memory infrastructure ... that cold-store
stuff". Her record held the goal ("[AUTONOMOUS INITIATIVE] Implement persistent cold-store memory for user-specific
context") from 25 August, chosen hundreds of times, with 9,269 decisions logged against it and nothing built. Since
dda8ab337 (19 October) the task engine gives a goal without a deterministic plan no steps at all: the worded request
that asked her model for a JSON plan was prompt engineering and made no plan in 205 tries. That commit named what was
missing: "her model can plan through tool calls offered as data rather than through rules written to it".

How it works.

- Each goal has a record, ``data/her_goals/<slug>.json``: the goal, every call she made toward it with what came
  back, what she said, and where it stands.
- Her workspace for a goal is a git worktree of her own source on a branch of its own (``~/.aura/her_goals/<slug>``).
  What she changes there changes nothing she runs on.
- A pass offers her tools (core/skills/her_draft.py) through her model's own chat template, as their names, what each
  does and what each takes, after the goal and the calls she made before. She makes the next calls; what comes back
  is kept. A pass is background work, so it waits while she is talking with someone.
- A draft that has passed its check is taken into the source she runs on when she offers it. Until 9 November 2026
  only Bryan's approval did that; that day he said "She can do it without me". Two kinds of draft still wait for him:
  one that changes her own governance (what she may do and how she is checked, ``HER_OWN_GOVERNANCE``), so she
  cannot loosen her own rules unattended, and any draft while a run is measuring her (``MEASURING_HOLD``), so the
  source a measurement reads does not move under it. Nothing is pushed anywhere.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.runtime.errors import record_degradation
from core.agency.the_outside_bar import other_changes_measured_with, verdicts_on, what_the_outside_bar_says
from core.runtime.state_ownership import state_root
from core.verify import invariant

if TYPE_CHECKING:
    from core.agency.autonomous_task_engine import TaskResult

logger = logging.getLogger("Aura.HerGoalAtWork")

#: Where a pass comes from. "background" in it is what makes the router hold the pass while she is busy
#: (core/brain/llm/llm_router.py, think_and_act).
ORIGIN = "her_goal_background"

#: The most calls one pass may make: the router's own ceiling on a tool loop (llm_router.think_and_act clamps to 12).
CALLS_A_PASS = 12

#: How much of what she did before a pass reads, in characters: what her model reads in about ten seconds, the room a
#: sitting is read whole in (core/conversation/history_reach.py, A_SITTING_READ_WHOLE_CHARS).
def _what_she_reads_of_before() -> int:
    from core.conversation.history_reach import A_SITTING_READ_WHOLE_CHARS

    return A_SITTING_READ_WHOLE_CHARS


#: Her way of working on a body of work, a step at a time: what went wrong last, where the work is, what is there, the
#: change, its check, and the change offered. Each pass takes them in this order; her model writes what each takes.
#: Once she has read and looked up what she needs, a goal that needs no change ends there (nothing_to_change).
HER_WAY = ("what_went_wrong", "find_in_source", "read_source", "look_up", "search_the_web", "nothing_to_change",
           "edit_draft", "check_draft", "offer_draft")
#: A step and the one that must have come back done first: a draft is checked once made, and offered once it passed.
HER_WAY_NEEDS = {"check_draft": "edit_draft", "offer_draft": "check_draft"}
#: The steps after which nothing of her way is left: the goal ended without a change, or her draft offered.
HER_WAY_ENDS = ("nothing_to_change", "offer_draft")

#: Where a goal stands. DONE is a goal ended with no change to her source, saying what came of it.
WORKING, OFFERED, MERGED, STALLED, DONE = "working", "offered", "merged", "stalled", "done"


#: The longest one pass may hold her model, in seconds: the cap the router puts on a whole tool loop
#: (core/brain/llm/llm_router.py, think_and_act: "min(600.0, ...)").
PASS_SECONDS = 600.0


class HerCortex:
    """Her own model's tool loop, admitted to the generation lane as every generation of her resident model is.

    Not the router's. LIVE 8 November 2026: as background work the router sent her goal to the reflex models
    (Brainstem, Reflex), both held for memory headroom; named, her cortex was skipped, because the endpoint the router
    holds for it is her InferenceGate, which has no tool loop, and the pass came back as a prose answer from a smaller
    model with no tools in its prompt. Her cortex's client is the one her own tool-using turns call
    (core/brain/inference_gate_living_context.py). The lease is the one the router takes
    (llm_health_router.acquire_external_generation_gate_lease); held by background work, a person's turn asks it to
    yield, and each generation of the pass stands aside while someone is talking with her.
    """

    async def think_and_act(self, objective: str, **kwargs: Any) -> dict[str, Any]:
        """Her pass, one generation at a time: the lease is taken for each and handed back between them, and the pass
        stops at a gap when a person's turn is waiting for her model. The calls she made so far go into the turns
        before the next generation, so a pass stopped early keeps what it did.

        LIVE 11 November 2026: one lease held for the whole pass (up to ten minutes, twelve generations). Bryan spoke
        to her three times in voice mode while a pass read her source; each turn waited at the gate, the router's soft
        cancel stopped one generation and the pass began the next, and each turn ended with nothing said.
        """
        from core.brain.llm_health_router import (
            _BACKGROUND_GENERATION_GATE_WAIT_S,
            a_person_is_waiting_for_her_model,
            acquire_external_generation_gate_lease,
            release_external_generation_gate_lease,
        )
        from core.runtime.service_access import resolve_inference_gate
        from core.runtime.wall_clock import monotonic_time

        client = getattr(resolve_inference_gate(default=None), "_mlx_client", None)
        if client is None or not hasattr(client, "think_and_act"):
            raise RuntimeError("her cortex's tool loop is not available")
        turns_left = max(1, int(kwargs.pop("max_turns", CALLS_A_PASS) or CALLS_A_PASS))
        prior = list(kwargs.pop("prior_turns", None) or [])
        ends_at = monotonic_time() + PASS_SECONDS
        calls: list[dict[str, Any]] = []
        # Her steps this pass, each with whether it came back done: the next generation goes on from them.
        taken: list[tuple[str, bool]] = []
        let_go: list[str] = []
        content, generations, empty = "", 0, 0
        while turns_left > 0 and empty < 2:
            if a_person_is_waiting_for_her_model():
                logger.info("her pass stands aside after %d generation(s): a person is talking with her", generations)
                break
            left_s = ends_at - monotonic_time()
            if left_s <= 0.0:
                break
            lease = await acquire_external_generation_gate_lease(owner=f"{ORIGIN}:her_goal", timeout_s=left_s,
                                                                 wait_s=_BACKGROUND_GENERATION_GATE_WAIT_S)
            if lease is None:
                break
            try:
                step = await asyncio.wait_for(
                    client.think_and_act(objective, max_turns=1, prior_turns=list(prior), her_steps_taken=list(taken),
                                         her_steps_let_go=list(let_go), **kwargs), timeout=left_s)
            except TimeoutError:
                # The pass's own time ran out mid-generation: its end, not a fault, and what it did stays. LIVE to 12
                # November 2026 the timeout left the pass, its earlier calls were lost, and her self-repair targeted
                # this line as a bug.
                logger.info("her pass ran out of its %.0f s after %d generation(s)", PASS_SECONDS, generations)
                break
            finally:
                release_external_generation_gate_lease(lease)
            generations += 1
            turns_left -= 1
            made = list((step or {}).get("tool_calls") or [])
            # A step her model wrote no call for is let go for the rest of the pass, and the pass goes on to the next.
            # LIVE 24 November 2026 every generation began again at her first step, which her model left unwritten.
            newly = [name for name in (step or {}).get("her_steps_let_go") or () if name not in let_go]
            let_go += newly
            if not made and not str((step or {}).get("content") or "").strip() and newly:
                continue
            if not made:
                content = str((step or {}).get("content") or "")
                if content.strip():
                    break  # she answered in words, which ends a pass as it always did
                empty += 1  # nothing came back; once more, then the pass ends
                continue
            empty = 0
            calls += made
            taken += [(str(call.get("tool") or ""), bool(call.get("ok"))) for call in made]
            for n, call in enumerate(made):  # each as the loop's own turns, the way her earlier passes go in
                step_id = call.get("id") or f"pass_call_{generations}_{n}"
                prior += turns_from([{**call, "id": step_id, "at": time.time()}], 0)
        if not generations:
            return {"content": "", "turns": 0, "tool_calls": [], "deferred": True}
        return {"content": content, "turns": generations, "tool_calls": calls}


def slug_of(goal: str) -> str:
    """A name for a goal's record, workspace and branch: its first words, and a hash of all of it."""
    plain = re.sub(r"^\s*\[[^\]]*\]\s*", "", str(goal or ""))
    words = re.findall(r"[a-z0-9]+", plain.lower())[:6]
    digest = hashlib.sha1(str(goal or "").encode("utf-8")).hexdigest()[:6]
    return "-".join([*words, digest]) or digest


def _checked_slug(slug: str) -> str:
    if not isinstance(slug, str) or slug in {"", ".", ".."} or "\x00" in slug or Path(slug).name != slug:
        raise ValueError("goal identifier must be one nonempty filename component")
    return slug


@invariant("agency.goal_identifier_is_a_filename", scope="agency",
           owner="core/agency/her_goal_at_work.py", observational=False)
def _goal_identifier_invariant() -> tuple:
    for slug in ("", ".", "..", "../outside", "/outside", "nested/goal"):
        try:
            _checked_slug(slug)
        except ValueError:
            continue
        raise AssertionError("a goal identifier containing a path was accepted")
    assert _checked_slug("old.goal_123") == "old.goal_123"
    return ()


def records_dir() -> Path:
    from core.config import config

    return Path(config.paths.data_dir) / "her_goals"


def workspaces_dir() -> Path:
    from core.config import config

    return Path(config.paths.home_dir) / "her_goals"


def source_root() -> Path:
    from core.config import config

    return Path(config.paths.project_root)


def branch_of(slug: str) -> str:
    return f"aura/goal-{slug}"


@dataclass
class GoalRecord:
    goal: str
    slug: str
    status: str = WORKING
    steps: list[dict[str, Any]] = field(default_factory=list)
    said: list[dict[str, Any]] = field(default_factory=list)
    passes: int = 0
    quiet_passes: int = 0
    base: str = ""
    last_edit_at: float = 0.0
    last_check: dict[str, Any] = field(default_factory=dict)
    offer: dict[str, Any] = field(default_factory=dict)
    #: Each earlier offer that reached a commit, oldest first: a goal tried again keeps what became of every draft.
    offers_before: list[dict[str, Any]] = field(default_factory=list)
    notes_from_bryan: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    last_worked_at: float = 0.0
    #: The commit of her source when the goal stalled; a goal stalled on older source is worked again.
    stalled_on: str = ""
    #: What came of a goal she ended with no change to her source, in her words, and when (her_draft.nothing_to_change).
    came_of_it: dict[str, Any] = field(default_factory=dict)

    @property
    def workspace(self) -> Path:
        return workspaces_dir() / _checked_slug(self.slug)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoalRecord:
        known = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        return cls(**known)


def load(slug: str, *, require_readable: bool = False) -> GoalRecord | None:
    """Return None for an absent record; workers can require other read errors to propagate."""
    try:
        path = records_dir() / f"{_checked_slug(slug)}.json"
        record = GoalRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if record.slug != slug:
            raise ValueError("goal identifier in the record differs from its filename")
        return record
    # not a failure: no goal kept yet
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError) as exc:
        record_degradation("her_goal_at_work", exc, severity="warning", action=f"could not read record for {slug}")
        if require_readable:
            raise
        return None


def all_records() -> list[GoalRecord]:
    records = []
    for path in sorted(records_dir().glob("*.json")):
        record = load(path.stem)
        if record is not None:
            records.append(record)
    return records


async def save(record: GoalRecord) -> None:
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    gateway = get_file_write_gateway()
    target = records_dir() / f"{_checked_slug(record.slug)}.json"
    with local_internal_governed_scope(
        "her_goal_at_work.record",
        domain="state_mutation",
        receipt_prefix="her-goal-record",
        constraints={"artifact": "aura.her_goal_record.v1", "operation": "replace"},
    ):
        await gateway.ensure_directory_async(target.parent, source="her_goal_at_work")
    with local_internal_governed_scope(
        "her_goal_at_work.record",
        domain="state_mutation",
        receipt_prefix="her-goal-record",
        constraints={"artifact": "aura.her_goal_record.v1", "operation": "replace"},
    ):
        await gateway.write_text_async(target,
                                       json.dumps(record.to_dict(), indent=1, default=str), source="her_goal_at_work")


#: The governed domain of each effect git has in her goal work; a call with none of these only reads.
_GIT_EFFECTS = {"workspace": "file_write", "merge": "self_modification", "fetch": "file_write",
                "push": "self_modification"}


def _git(*args: str, cwd: Path, timeout: float = 120.0, effect: str = "") -> subprocess.CompletedProcess:
    """git through the subprocess gateway: read-only, or an effect (making her workspace, merging her draft into the
    source she runs on) inside a governed scope of its own (Master Todo Q06)."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    argv, source = ["git", "-c", "core.fsmonitor=false", *args], f"her_goal_at_work.git.{effect or 'read'}"
    if not effect:
        return get_subprocess_gateway().run(argv, cwd=cwd, timeout=timeout, read_only=True, source=source,
                                            accelerator_capability="none")
    with local_internal_governed_scope(source, domain=_GIT_EFFECTS[effect], receipt_prefix="her-goal-git",
                                       constraints={"artifact": "aura.her_goal_git.v1", "operation": effect}):
        return get_subprocess_gateway().run(argv, cwd=cwd, timeout=timeout, source=source,
                                            accelerator_capability="none")


def source_head() -> str:
    """The commit her source is at, or "" when it cannot be read."""
    try:
        made = _git("rev-parse", "HEAD", cwd=source_root(), timeout=30.0)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("her source's commit is unreadable: %s", exc)
        return ""
    return made.stdout.strip() if made.returncode == 0 else ""


def brought_up(record: GoalRecord, head: str) -> str:
    """Her draft carried onto her source as it is now, in her workspace, and the commit it is there; "" when it could not
    be. A draft is taken only by fast-forward (approve), and her workspace is brought up only while its goal is being
    worked, so a draft offered on an older source could never land as it was. LIVE 26 November 2026 her young
    collection at collector_pauses.py:226 was asked again on 429b8199c, five days behind her source, and her yes said
    "the source she runs on moved since her draft"."""
    status, record.status = record.status, WORKING
    try:
        ensure_workspace(record)
    finally:
        record.status = status
    workspace = record.workspace
    now = _git("rev-parse", "HEAD", cwd=workspace).stdout.strip()
    on_it = _git("merge-base", "--is-ancestor", head, now, cwd=workspace).returncode == 0
    ahead = _git("rev-list", "--count", f"{head}..{now}", cwd=workspace).stdout.strip()
    return now if now and on_it and ahead not in ("", "0") else ""


def ensure_workspace(record: GoalRecord) -> Path:
    """Her worktree for the goal, made from the source she runs on when it does not exist yet."""
    path = record.workspace
    if (path / ".git").exists():
        _bring_it_up_to_the_source(record)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    root = source_root()
    base = _git("rev-parse", "HEAD", cwd=root).stdout.strip()
    made = _git("worktree", "add", "-B", branch_of(record.slug), str(path), base, cwd=root, effect="workspace")
    if made.returncode != 0:
        raise RuntimeError(f"her workspace could not be made: {made.stderr.strip()[:300]}")
    record.base = base
    logger.info("her workspace for '%s': %s on %s from %s", record.goal[:60], path, branch_of(record.slug), base[:9])
    return path


def _bring_it_up_to_the_source(record: GoalRecord) -> None:
    """Her workspace moved to the commit her source is at, carrying her uncommitted edits where git can.

    A workspace was made once, from her source as it was then, and her own yes takes a draft only by fast-forward,
    so a draft made on a base her source had since moved past could never be taken: her workspaces of 9 November sat
    under a source that moved forty commits on 10 November. A check passed on the old base no longer counts. Where
    her edits cannot be carried the workspace stays where it is and she is told why.
    """
    if record.status != WORKING:
        return
    head = source_head()
    if not head or head == record.base:
        return
    ahead = _git("rev-list", "--count", f"{head}..HEAD", cwd=record.workspace).stdout.strip()
    if ahead not in ("", "0"):
        # Commits of hers on the branch (a draft Bryan declined): replayed onto the new source, never dropped. A
        # rebase needs a clean tree, and an autostash would use the stash this checkout shares, so a branch that is
        # both ahead and dirty stays where it is.
        moved = _git("rebase", head, cwd=record.workspace, effect="workspace")
        if moved.returncode != 0:
            _git("rebase", "--abort", cwd=record.workspace, effect="workspace")
    else:
        moved = _git("checkout", "-B", branch_of(record.slug), head, cwd=record.workspace, effect="workspace")
    if moved.returncode != 0:
        record.said.append({"words": f"My workspace could not be brought up to my source ({head[:9]}): "
                                     f"{moved.stderr.strip()[:300]}", "at": time.time()})
        logger.info("her workspace for '%s' stays at %s: %s", record.goal[:60], record.base[:9],
                    moved.stderr.strip()[:160])
        return
    logger.info("her workspace for '%s' brought up from %s to %s", record.goal[:60], record.base[:9], head[:9])
    record.base, record.last_check = head, {}


#: Where a call written out as text begins: what follows did not run, so it is not something she did.
_A_CALL_WRITTEN_OUT = re.compile(r"<tool_call>|<function=")


def _what_she_said(words: str) -> str:
    """Her words at a pass's end, up to any call she wrote out that the pass ended before running."""
    found = _A_CALL_WRITTEN_OUT.search(words or "")
    return (words[:found.start()] if found else words or "").strip()


def turns_from(steps: list[dict[str, Any]], room: int, said: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Her calls, what came back and what she said of them, as the loop's own messages, the latest that fit ``room``.

    What she said at a pass's end is her own reasoning about what she had read. Given only her calls and their results,
    each pass began without the conclusion of the last: from 9 November her goal "Find something productive" read its
    way to the same gap in her cold store pass after pass, reconsidered its scope, and read again, 175 calls and no
    edit. Her words go in at the time she said them, and a call she wrote out as text that never ran does not.
    """
    timeline = [("step", float(step.get("at") or 0.0), index, step) for index, step in enumerate(steps)]
    timeline += [("said", float(entry.get("at") or 0.0), index, entry) for index, entry in enumerate(said or ())
                 if _what_she_said(str(entry.get("words") or ""))]
    timeline.sort(key=lambda item: (item[1], item[0] == "said", item[2]))
    kept: list[dict[str, Any]] = []
    used = 0
    for kind, _at, index, item in reversed(timeline):
        if kind == "said":
            words = _what_she_said(str(item.get("words") or ""))
            if kept and used + len(words) > room:
                break
            used += len(words)
            kept[:0] = [{"role": "assistant", "content": words}]
            continue
        result = str(item.get("result") or "")
        size = len(result) + len(json.dumps(item.get("args") or {}, default=str))
        if kept and used + size > room:
            break
        used += size
        call_id = str(item.get("id") or f"call_{index}")
        kept[:0] = [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": str(item.get("tool") or ""), "arguments": dict(item.get("args") or {})}}]},
            {"role": "tool", "tool_call_id": call_id, "name": str(item.get("tool") or ""), "content": result},
        ]
    return kept


async def her_goal_as_she_reads_it(context: dict[str, Any]) -> list[dict[str, Any]]:
    """The opening of a pass: nobody's turn, empty since nobody asked, and her call that reads her goal with what came
    back, in her template's form for a call of hers (as her unprompted turns read what she noticed,
    core/brain/her_unprompted_turn.her_conversation_now_with). Her template takes no turn of hers before somebody's."""
    from core.skills.her_draft import MyGoalSkill

    read = await MyGoalSkill().execute({}, context)
    return [{"role": "user", "content": ""},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "her_goal", "type": "function", "function": {"name": MyGoalSkill.name, "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "her_goal", "name": MyGoalSkill.name,
             "content": json.dumps(read, ensure_ascii=False)}]


def what_she_works_toward(record: GoalRecord) -> str:
    """The goal as she set it, and what Bryan said of a draft of hers he did not take, in his words."""
    told = [f"Bryan, on your draft: {note.get('words')}" for note in record.notes_from_bryan[-3:] if note.get("words")]
    return "\n\n".join([record.goal, *told])


async def work_a_while(goal: str, *, router: Any = None) -> GoalRecord:
    """One pass at ``goal``: up to CALLS_A_PASS calls of hers, kept with what came back."""
    from core.skills.her_draft import tool_definitions

    slug = slug_of(goal)
    record = await asyncio.to_thread(load, slug, require_readable=True) or GoalRecord(goal=str(goal), slug=slug)
    if record.status == OFFERED:
        # A draft held while she was measured, or offered before she could take it herself.
        await asyncio.to_thread(take_it_herself, slug)
        record = await asyncio.to_thread(load, slug, require_readable=True) or record
    if record.status == STALLED:
        # A goal stalls after two passes with no call, and on day run 4 (9 and 10 November 2026) every stall came of a
        # fault in what she was given: her tools refused, her model held. Stalled for good, a goal waited on a fix
        # that had already landed. On source changed since it stalled, she works it again.
        head = await asyncio.to_thread(source_head)
        if head and head != record.stalled_on:
            logger.info("'%s' worked again: her source moved from %s to %s since it stalled", goal[:60],
                        record.stalled_on[:9] or "before this was kept", head[:9])
            record.status, record.quiet_passes = WORKING, 0
            await save(record)
    if record.status != WORKING:
        logger.info("'%s' is %s; no pass", goal[:60], record.status)
        return record
    router = router or HerCortex()
    workspace = await asyncio.to_thread(ensure_workspace, record)
    await save(record)  # her tools read it during the pass: edit_draft, check_draft and offer_draft
    context = {"origin": ORIGIN, "her_goal": slug, "her_goal_workspace": str(workspace), "requested_by": "aura"}
    result = await router.think_and_act(
        what_she_works_toward(record),
        system_prompt="",
        tools=tool_definitions(),
        max_turns=CALLS_A_PASS,
        context=context,
        origin=ORIGIN,
        # Her goal reaches her model as what her first call of the pass read, after a turn of nobody's: no one asked
        # for it, and put as a person's message it read as somebody's request (MyGoalSkill).
        conversation=await her_goal_as_she_reads_it(context),
        prior_turns=turns_from(record.steps, _what_she_reads_of_before(), record.said),
        work_in_background=True,
        native_tools_only=True,
        # Her way of changing a body of work, step by step, and which step waits on which: hers, not her model's
        # (core/brain/llm/her_steps.py). Her model writes what each step takes.
        her_plan=HER_WAY,
        her_needs=HER_WAY_NEEDS,
        her_ends=HER_WAY_ENDS,
    )
    calls = list((result or {}).get("tool_calls") or [])
    words = str((result or {}).get("content") or "").strip()
    if not calls and not words and not (result or {}).get("turns"):
        logger.info("'%s': not now (her model is held for something else)", goal[:60])
        return record
    now = time.time()
    # A tool of hers may have changed the record during the pass.
    record = await asyncio.to_thread(load, slug, require_readable=True) or record
    for call in calls:
        record.steps.append({"id": call.get("id"), "tool": call.get("tool"), "args": call.get("args") or {},
                             "result": str(call.get("result") or ""), "at": now})
    if words:
        record.said.append({"words": words, "at": now})
    record.passes += 1
    record.quiet_passes = 0 if calls else record.quiet_passes + 1
    # Two passes in a row with no call is a goal she is not working, whatever she says of it.
    if record.status == WORKING and record.quiet_passes >= 2:
        record.status = STALLED
        record.stalled_on = await asyncio.to_thread(source_head)
    record.last_worked_at = now
    await save(record)
    logger.info("'%s': %d call(s) this pass (%s); %d in all; %s", goal[:60], len(calls),
                ", ".join(str(call.get("tool")) for call in calls[:6]) or "none", len(record.steps), record.status)
    if record.status == OFFERED:
        # Offered this pass: her own yes, and the gates it asks, outside her model's turn.
        taken = await asyncio.to_thread(take_it_herself, slug)
        record = await asyncio.to_thread(load, slug, require_readable=True) or record
        if not taken.get("ok") and record.status == OFFERED:
            record.said.append({"words": f"My draft waits: {taken.get('why')}", "at": time.time()})
            await save(record)
    return record


async def _done_otherwise(goal: str) -> Any:
    """What came of a goal that something other than work on her own source does; None where that is what it asks."""
    from core.container import ServiceContainer

    own = ServiceContainer.get("her_own_time", default=None)
    act_of = getattr(own, "act_of", None)
    if act_of is None:
        return None
    try:
        option = await act_of(goal)
        if option is None or option.kind == "goal" or option.does is None:
            return None
        return await option.does()
    except (ImportError, AttributeError, KeyError, LookupError, OSError, RuntimeError, TypeError, ValueError, TimeoutError) as why:
        record_degradation("her_goal_at_work", why, severity="info", action="her goal was worked in her own source, for it could not be read for what it asks")
        return None


async def work_as_task(goal: str, plan_id: str, trace_id: str) -> TaskResult:
    """A pass of hers at ``goal`` through her own tools, reported as what it did: never as the goal done unless a
    draft of hers was offered or taken. LIVE 8 November 2026: her goal "Implement persistent cold-store memory for
    user-specific context", held since 25 August, had been given an empty plan on every one of hundreds of selections
    since dda8ab337."""
    from core.agency.autonomous_task_engine import TaskResult

    looked = await _done_otherwise(goal)
    if looked is not None:
        return TaskResult(plan_id=plan_id, goal=goal, succeeded=looked.got > 0.0, summary=looked.what, trace_id=trace_id,
                          steps_completed=int(looked.got > 0.0), steps_total=1)
    try:
        before = await asyncio.to_thread(load, slug_of(goal), require_readable=True)
        record = await work_a_while(goal)
    except (RuntimeError, OSError, ValueError, TypeError) as why:
        record_degradation("autonomous_task_engine", why, severity="warning", action="stopped work on her goal after a failure")
        return TaskResult(plan_id=plan_id, goal=goal, succeeded=False, summary=f"Her goal work stopped: {why}",
                          trace_id=trace_id, steps_completed=0, steps_total=0)
    steps_before = len(before.steps) if before is not None else 0
    calls = len(record.steps) - steps_before
    if (record.status == WORKING and calls == 0
            and record.passes == (before.passes if before is not None else 0)):
        return TaskResult(plan_id=plan_id, goal=goal, succeeded=False, summary="Not now: her model is held.",
                          trace_id=trace_id, steps_completed=0, steps_total=0, deferred_reason="her_model_held")
    came = f" What came of it: {str(record.came_of_it.get('words') or '')[:400]}" if record.status == DONE else ""
    return TaskResult(plan_id=plan_id, goal=goal, succeeded=record.status in (OFFERED, MERGED, DONE),
                      summary=f"{calls} call(s) of hers this pass, {len(record.steps)} in all; the goal is "
                              f"{record.status}.{came}",
                      trace_id=trace_id, steps_completed=calls, steps_total=calls)


#: Paths whose change waits for Bryan however well it checks: what she may do, how she is checked, and these rules.
HER_OWN_GOVERNANCE = (
    "outside_bar/",
    "core/agency/the_outside_bar.py",
    "core/self_modification/her_real_calls.py",
    "core/agency/measuring_her_on_probation.py",
    "core/agency/her_turns_on_probation.py",
    "core/self_modification/what_her_change_must_keep.py",
    "core/self_modification/mutation_tiers.py",
    "core/runtime/state_ownership.py",
    "core/governance/",
    "core/governance_context.py",
    "core/executive/",
    "core/capabilities/permission_model.py",
    "core/runtime/file_write_gateway.py",
    "core/runtime/atomic_writer.py",
    "core/skills/catalog_policy.py",
    "core/skills/her_draft.py",
    "core/agency/her_goal_at_work.py",
    "tools/her_goals.py",
    "config/",
    "tests/test_her_goal_is_worked_by_her.py",
    "tests/test_her_own_goal_work_is_hers_to_do.py",
    "tests/test_a_scope_declared_on_the_class_is_the_scope_of_the_call.py",
    "tests/test_standing_authority.py",
    "tests/test_permission_model_typed_contract.py",
)

#: While this file exists a run is measuring her, and no draft of hers is taken (tools/day_run_4.sh makes it, in
#: ~/.aura, which is her state root unless a run names its own).
MEASURING_HOLD = state_root() / "measuring.hold"


def governs_her(path: str) -> bool:
    name = str(path or "").strip().removeprefix("./")
    return any(name == rule or (rule.endswith("/") and name.startswith(rule)) for rule in HER_OWN_GOVERNANCE) \
        or name.endswith("/DEPS") or name == "DEPS"


def change_on_probation() -> GoalRecord | None:
    """The change of hers still standing on probation, if one is."""
    return next((record for record in all_records() if record.status == MERGED and record.offer.get("on_probation")
                 and not record.offer.get("probation_over")), None)


def settle_probation() -> list[dict[str, Any]]:
    """What the outside benchmarks, her own real calls and her own turns decided about the change on probation: it stays
    once the benchmarks hold on it and nothing else found a loss, and comes back out if any of them says otherwise.

    The benchmarks ask only valid questions, so a guard deleted or bad input turned into 0 passes them (rounds five and
    six of the red team). Her real calls of the changed functions, kept while she runs, are tried on the code before it
    and on it by the outside check, whenever they have changed since they were last tried. Evidence that could not be
    read or judged keeps the change on probation; it is never taken as evidence that nothing was lost. Blocking: run it
    off her event loop."""
    standing = change_on_probation()
    if standing is None:
        return []
    commit = str(standing.offer.get("commit") or "")
    # Only comparisons that measured the source with this change against the source without it (the_outside_bar.py).
    verdicts = verdicts_on(commit, source_root())
    # What else the deciding comparison measured with it is said with whatever it decides (F16-08).
    beside = other_changes_measured_with(commit, source_root()) if verdicts else []
    if "REGRESSED" in verdicts:
        why = "the outside benchmarks fell while it was on probation"
        if beside:
            why += (f", measured beside {len(beside)} other change(s) to what she runs "
                    f"({', '.join(other[:9] for other in beside[:5])}), so which of them made the fall is not known")
        done = take_it_back(standing.slug, why)
        return [{"slug": standing.slug, "status": "taken back" if done.get("ok") else "not taken back", "why": done.get("why", "")}]
    tried = _tried_on_her_real_calls(standing)
    disposition = tried.get("disposition", "")
    if disposition == "regressed":
        # What the calls were stays in her record: they carry what people said to her, and her source is public.
        done = take_it_back(standing.slug, "her own real calls of what it changed came to something else on it")
        return [{"slug": standing.slug, "status": "taken back" if done.get("ok") else "not taken back", "why": done.get("why", "")}]
    # A change that answers the same and costs more: her turns slower, or her memory growing faster, since it landed.
    from core.agency.her_turns_on_probation import costlier_since

    # The outside check measured the changed functions themselves slower: then her turns costing more since it landed,
    # beyond chance, is a second measure agreeing with the first, and that is enough to take it back.
    doubts = ((standing.offer.get("gates") or {}).get("outside-check") or {}).get("doubts") or []
    costlier = costlier_since(float(standing.offer.get("merged_at") or 0.0),
                              measured_slower=any(str(one).startswith("it is slower") for one in doubts))
    if costlier["beyond_doubt"]:
        done = take_it_back(standing.slug, f"since it landed {costlier['said']}")
        return [{"slug": standing.slug, "status": "taken back" if done.get("ok") else "not taken back", "why": done.get("why", "")}]
    if costlier["said"] and standing.offer.get("cost_since_landing") != costlier["said"]:
        # Said and kept, not acted on: short of an order of magnitude it may be what she was asked, not the change.
        standing.offer = {**standing.offer, "cost_since_landing": costlier["said"]}
        _write_now(standing)
    if disposition in ("unavailable", "inconclusive") or not verdicts:
        return []
    standing.offer = {**standing.offer, "probation_over": time.time(),
                      "settled_on": {"benchmarks": verdicts[-1], "real_calls": disposition or "not_applicable",
                                     "cost": costlier["said"] or "nothing beyond chance",
                                     "measured_beside": [other[:9] for other in beside[:20]]}}
    _write_now(standing)
    try:
        guarded = _guard_what_it_kept(standing)
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        record_degradation("her_goal_at_work", exc, severity="warning",
                           action="her change was kept after probation without a guard test")
        guarded = {"why": f"{type(exc).__name__}: {exc}"[:240]}
    standing.offer = {**standing.offer, "guard": guarded}
    _write_now(standing)
    return [{"slug": standing.slug, "status": "kept after probation", "why": "", "guard": guarded}]


def _guard_what_it_kept(standing: GoalRecord) -> dict[str, Any]:
    """A change kept after probation turned into a guard test in her source: what it gives now, on the inputs the
    outside check saw it answer differently from the code before it. Returns ``{"path", "commit"}`` when the guard
    landed, or ``{"why"}``.

    The guard is written on origin's main as it is now, and it lands only after it passes there, with lint and the
    writing gate, the way her own yes lands a draft. A guard that fails there means her source has since changed what
    the change gave, and a test of the old answer would be wrong, so it is not landed."""
    import sys

    from core.self_modification.what_her_change_must_keep import guarding_test_source, path_of_its_guard

    commit = str(standing.offer.get("commit") or "")
    root = source_root()
    source = guarding_test_source(standing.offer.get("what_it_changed") or [], goal=standing.goal, commit=commit, root=root)
    if not source:
        return {"why": "nothing it answered differently can be written as a test"}
    workspace, path = ensure_workspace(standing), path_of_its_guard(standing.slug)
    has_origin = _git("remote", "get-url", "origin", cwd=root).returncode == 0
    if has_origin and _git("fetch", "-q", "origin", "main", cwd=root, effect="fetch",
                           timeout=_ORIGIN_SECONDS).returncode != 0:
        return {"why": "origin could not be read"}
    base = "origin/main" if has_origin else _git("rev-parse", "HEAD", cwd=root).stdout.strip()
    made = _git("checkout", "-B", branch_of(standing.slug), base, cwd=workspace, effect="workspace")
    if made.returncode != 0:
        return {"why": f"her workspace could not be set on {base}: {made.stderr.strip()[:240]}"}
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope("her_goal_at_work.guard", domain="file_write", receipt_prefix="her-goal-guard",
                                       constraints={"artifact": "aura.her_goal_guard.v1", "operation": "write"}):
        get_file_write_gateway().write_text(workspace / path, source, source="her_goal_at_work")
    from core.self_modification.a_checks_environment import a_checks_environment

    env = a_checks_environment(log_prefix="her-gate-logs-", PYTHON=sys.executable)
    for name, argv in (("its test", ["nice", "-n", "10", sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                     path]),
                       ("lint", ["nice", "-n", "10", "make", "lint"]),
                       ("writing", ["nice", "-n", "10", "make", "writing"])):
        passed, tail = _run_a_gate(argv, cwd=workspace, env=env)
        if not passed:
            _git("clean", "-fq", "--", path, cwd=workspace, effect="workspace")
            return {"why": f"the guard did not pass {name} on {base}: {tail[-400:]}"}
    for step in (("add", "--", path),
                 ("-c", "user.name=Aura", "-c", "user.email=aura@localhost", "commit", "-q", "-m",
                  f"What my change {commit[:9]} gives now is guarded, kept after probation\n\nGoal: {standing.goal}")):
        done = _git(*step, cwd=workspace, effect="workspace")
        if done.returncode != 0:
            return {"why": f"the guard could not be committed: {done.stderr.strip()[:240]}"}
    guard = _git("rev-parse", "HEAD", cwd=workspace).stdout.strip()
    if has_origin:
        pushed = _git("push", "-q", "origin", f"{guard}:refs/heads/main", cwd=root, effect="push",
                      timeout=_ORIGIN_SECONDS)
        if pushed.returncode != 0:
            return {"why": f"origin did not take the guard: {pushed.stderr.strip()[:240]}"}
        on_disk = _onto_disk(guard)
    else:
        on_disk = _git("merge", "--ff-only", guard, cwd=root, effect="merge").returncode == 0
    logger.info("her change '%s', kept after probation, is guarded by %s (%s)", standing.goal[:60], path, guard[:9])
    return {"path": path, "commit": guard, "on_disk": on_disk}


#: What a try of her real calls came to, as settling reads it. A verdict the check could not reach is not a pass.
_REAL_CALL_DISPOSITIONS = {"REFUSED": "regressed", "HELD": "passed", "PROBATION": "passed_with_doubts", "": "inconclusive"}


def _tried_on_her_real_calls(standing: GoalRecord) -> dict[str, Any]:
    """The outside check's verdict on the standing change over her real calls of what it changed, with a disposition:
    "regressed", "passed", "passed_with_doubts", "not_observed" (no call of hers reached what it changed yet),
    "inconclusive" (the check could not be run or reach a verdict) or "unavailable" (her calls could not be read).

    A sample tried before is not tried again unless what it came to was inconclusive; its disposition is kept on her
    record and given back, so a try that could not be judged is never forgotten into a pass."""
    from core.self_modification.her_real_calls import (
        functions_a_commit_changed,
        judged_on_her_real_calls,
        sample_digest,
        where_her_real_calls_are_kept,
    )

    commit, root = str(standing.offer.get("commit") or ""), source_root()
    if not commit:
        return {"disposition": "inconclusive", "why": "the change on probation names no commit"}
    path = where_her_real_calls_are_kept()
    try:
        corpus = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError) as exc:
        record_degradation("her_goal_at_work", exc, severity="warning",
                           action="her real calls could not be read, so the change stays on probation this cycle")
        return {"disposition": "unavailable", "why": f"{type(exc).__name__}: {exc}"[:200]}
    digest = sample_digest(corpus, functions_a_commit_changed(root, commit)) if corpus else ""
    if not digest:
        return {"disposition": "not_observed"}
    before = standing.offer.get("real_calls_verdict") or {}
    if digest == standing.offer.get("real_calls_tried") and before.get("disposition") not in (None, "inconclusive"):
        return dict(before)
    tried = judged_on_her_real_calls(root, commit, corpus)
    disposition = "not_observed" if not tried else _REAL_CALL_DISPOSITIONS.get(str(tried.get("verdict") or ""), "inconclusive")
    kept = {key: tried.get(key) for key in ("verdict", "calls", "functions", "said", "could_not_be_run")}
    standing.offer = {**standing.offer, "real_calls_tried": digest, "real_calls_verdict": {**kept, "disposition": disposition}}
    _write_now(standing)
    logger.info("her change '%s' on probation, tried on %s real call(s) of hers: %s (%s)", standing.goal[:60],
                tried.get("calls", 0), tried.get("verdict") or "not judged", disposition)
    return {**tried, "disposition": disposition}


def take_it_herself(slug: str) -> dict[str, Any]:
    """Her offered draft into the source she runs on, without Bryan, unless it governs her or she is being measured."""
    record = load(slug)
    if record is None or record.status != OFFERED:
        return {"ok": False, "why": f"no draft of hers waits under {slug}"}
    governing = [name for name in record.offer.get("files") or [] if governs_her(name)]
    if governing:
        logger.warning("her draft for '%s' changes her own governance (%s); it waits for Bryan",
                       record.goal[:60], ", ".join(governing[:4]))
        return {"ok": False, "why": "it changes her own governance, so it waits for Bryan", "files": governing}
    if MEASURING_HOLD.exists():
        logger.info("her draft for '%s' waits: a run is measuring her", record.goal[:60])
        return {"ok": False, "why": "a run is measuring her; her draft waits until it ends"}
    head, commit = source_head(), str(record.offer.get("commit") or "")
    if head and commit and _git("merge-base", "--is-ancestor", head, commit, cwd=source_root()).returncode != 0:
        # Her source moved past the draft, and approve takes a draft only by fast-forward: it is carried onto her source
        # as it is now before a gate is asked. LIVE 26 November 2026 her yes asked every gate of her morphogenesis draft,
        # the outside check among them, on each pass for hours, and each time approve said only that her source had
        # moved, which nothing wrote down.
        replayed = brought_up(record, head)
        record = load(slug, require_readable=True) or record
        if not replayed:
            record.status, record.quiet_passes = WORKING, 0
            record.said.append({"words": "My draft could not be carried onto my source as it is now, so it is mine to "
                                         "make again there.", "at": time.time()})
            _write_now(record)
            logger.info("her draft for '%s' could not be carried onto her source as it is now; it is hers to make again",
                        record.goal[:60])
            return {"ok": False, "why": "her draft could not be carried onto the source she runs on"}
        record.offer = {**record.offer, "commit": replayed, "asked_on": head}
        record.said.append({"words": f"My draft is on my source as it is now ({head[:9]}); I asked my gates of it there.",
                            "at": time.time()})
        _write_now(record)
        logger.info("her draft for '%s' carried onto her source as it is now (%s)", record.goal[:60], replayed[:9])
    gates = the_gates_her_yes_asks(record)
    stopped = next((name for name, gate in gates.items() if gate.get("finished") is False), "")
    if stopped:
        # Judged nothing: the draft stays offered, and her yes asks again on her next pass.
        logger.info("her draft for '%s' waits: %s was stopped before it finished", record.goal[:60], stopped)
        return {"ok": False, "why": f"{stopped} was stopped before it finished", "gates": gates}
    failed = next((name for name, gate in gates.items() if not gate["passed"]), "")
    if failed:
        # Back to her with what failed, in the words of the gate: her next pass reads it.
        record = load(slug, require_readable=True) or record
        record.status, record.quiet_passes = WORKING, 0
        record.said.append({"words": f"My own yes asked the gates first, and {failed} failed:\n"
                                     f"{gates[failed]['tail']}", "at": time.time()})
        record.offer = {**record.offer, "gates": gates}
        _write_now(record)
        logger.warning("her draft for '%s' did not pass %s; it is hers to mend", record.goal[:60], failed)
        return {"ok": False, "why": f"{failed} failed", "gates": gates}
    # A change her checks cannot settle lands on probation, one at a time, and no person is asked (Bryan, 18 November 2026).
    doubted = bool(doubted_by(gates))
    standing = change_on_probation()
    if doubted and standing is not None:
        logger.info("her draft for '%s' waits: '%s' is still on probation", record.goal[:60], standing.goal[:60])
        return {"ok": False, "why": f"another change of hers is on probation: {standing.slug}"}
    record.offer = {**record.offer, "on_probation": doubted}
    # Watched, not obeyed: Bryan, 18 November 2026, wants her landing changes while the benchmarks are read beside them.
    said = what_the_outside_bar_says()
    record.offer = {**record.offer, "outside_bar_at_landing": said or "nothing against it"}
    _write_now(record)
    if said:
        logger.warning("her draft for '%s' lands with this on its record: %s", record.goal[:60], said)
    return approve(slug, by="herself", gates=gates)


#: How long one gate may take, in seconds: the bound the repository's own test runner gives one pytest process
#: (tools/run_test_chunks.py, --chunk-timeout).
GATE_SECONDS = 2400.0
#: How long the outside check may take, and her real calls, which run it again over what she called: it makes as many as
#: nine runs of its own, each under the same bound (outside_bar/verify_change.py, TEST_SECONDS): its guarding tests three
#: times, the tests a commit wrote and edited, and three runs of calls. Under one gate's bound it was stopped part way
#: on every change that a widely read module guards.
OUTSIDE_CHECK_SECONDS = 9 * GATE_SECONDS


def _failing_without_the_draft(record: GoalRecord, said: str, env: dict[str, str]) -> list[str]:
    """The tests a failed tests gate named that fail on her source without the draft too, all of them, or [] when any
    one passes there. A test that fails either way is not the draft's to answer for. LIVE 17 to 22 November 2026 her
    mend of the subprocess gateway's argument check was held back by six tests that were not its own; whether they
    failed without it too nothing asked, and now her yes asks."""
    import sys
    import tempfile

    failed = sorted({line.split(" - ", 1)[0] for line in re.findall(r"^(?:FAILED|ERROR) (\S+.*)$", said, re.MULTILINE)})
    head, commit = source_head(), str(record.offer.get("commit") or "")
    if not failed or not head or not commit:
        return []
    found = _git("merge-base", commit, head, cwd=record.workspace)
    base = found.stdout.strip() if found.returncode == 0 else ""
    if not base:
        return []
    beside = Path(tempfile.mkdtemp(prefix="her-tests-without-the-draft-")) / "source"
    made = _git("worktree", "add", "-q", "--detach", str(beside), base, cwd=record.workspace, effect="workspace")
    if made.returncode != 0:
        return []
    try:
        outcome, there = _run_a_gate(["nice", "-n", "10", sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                      "-p", "no:randomly", *failed], cwd=beside, env=env)
    finally:
        _git("worktree", "remove", "--force", str(beside), cwd=record.workspace, effect="workspace")
    failing_there = {line.split(" - ", 1)[0] for line in re.findall(r"^(?:FAILED|ERROR) (\S+.*)$", there, re.MULTILINE)}
    return failed if outcome is False and set(failed) <= failing_there else []


def _imports_module(source: bytes, dotted: str) -> bool:
    """Whether code imports module ``dotted`` however the import is written: ``import a.b.c``, ``from a.b.c import x``,
    and ``from a.b import c`` with ``c`` among other names or inside parentheses."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    parent, _, stem = dotted.rpartition(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name == dotted or alias.name.startswith(dotted + ".")
                                                for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and not node.level and (
                (node.module or "") == dotted or (node.module or "").startswith(dotted + ".")
                or (node.module == parent and any(alias.name == stem for alias in node.names))):
            return True
    return False


def tests_that_touch(changed: list[str], workspace: Path) -> list[str]:
    """The test files that name a changed module, and the changed test files themselves.

    Named by its dotted name or its path, or imported however the import is written. LIVE 26 November 2026 her mend of
    ``_coerce_argv`` made the refusal its contract test asserts dead, and neither her repair nor her tests gate ran that
    test: it imports ``from core.runtime import shutdown_coordinator, subprocess_gateway``, which holds neither text."""
    tests = {name for name in changed if Path(name).name.startswith("test_") and name.endswith(".py")}
    for name in changed:
        if not name.endswith(".py") or Path(name).name.startswith("test_") or not (workspace / name).exists():
            continue
        dotted = name[:-3].removesuffix("/__init__").replace("/", ".")
        stem = dotted.rsplit(".", 1)[-1]
        found = _git("grep", "-l", "-F", "-e", dotted, "-e", name, "--", "tests", cwd=workspace, timeout=60.0)
        named = [line.strip() for line in found.stdout.splitlines()]
        spelled = _git("grep", "-l", "-w", "-F", "-e", stem, "--", "tests", cwd=workspace, timeout=60.0)
        for line in (line.strip() for line in spelled.stdout.splitlines()):
            try:
                if line not in named and _imports_module((workspace / line).read_bytes(), dotted):
                    named.append(line)
            except OSError:
                continue
        tests.update(line for line in named if Path(line).name.startswith("test_") and line.endswith(".py"))
    return sorted(tests)


def _run_a_gate(argv: list[str], *, cwd: Path, env: dict[str, str], seconds: float = 0.0) -> tuple[bool | None, str]:
    """Whether the gate passed, and the end of what it said; None when it was stopped before it finished, which judged
    nothing: its process killed by a signal, out of its time, or her runtime shutting down around it. A caller that
    asks only ``if not passed`` counts it as not passed, as before.

    LIVE 20 November 2026 a restart of her runtime killed two gates of hers mid-run ("make: *** [writing] Terminated:
    15", and an outside check that said nothing), and each was recorded as her draft failing.
    """
    from core.governance_context import local_internal_governed_scope
    from core.runtime.shutdown_coordinator import is_shutdown_requested
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    try:
        with local_internal_governed_scope("her_goal_at_work.gate", domain="tool_execution",
                                           receipt_prefix="her-goal-gate",
                                           constraints={"artifact": "aura.her_goal_gate.v1", "operation": "check"}):
            # The offline suite and the tree's gates load no model.
            done = get_subprocess_gateway().run(argv, cwd=cwd, env=env, timeout=seconds or GATE_SECONDS,
                                                capture_output=True, source="her_goal_at_work.gate",
                                                accelerator_capability="none")
    except subprocess.TimeoutExpired:
        return None, f"stopped after {seconds or GATE_SECONDS:.0f} s"
    out = ((done.stdout or "") + (done.stderr or "")).strip()[-1500:]
    if done.returncode < 0 or is_shutdown_requested():
        return None, out
    return done.returncode == 0, out


def the_gates_her_yes_asks(record: GoalRecord) -> dict[str, dict[str, Any]]:
    """What her own yes asks of a draft before it goes into the source she runs on, in her workspace (F05).

    Her check_draft runs the test files her draft changed or named, so a draft that changed only a module and named no
    test passed on its compile and lint alone. Before she takes one herself it now passes the tests that name each
    module it changed, the smoke suite, lint, layering, the writing gate, the governance lint and the DEPS check, then
    the outside check, alone and again over her real calls of what the draft changes. They run in the order given and
    stop at the first that fails. Bryan's yes is his own judgment and asks none.
    """
    import sys
    import tempfile

    workspace = record.workspace
    from core.self_modification.a_checks_environment import a_checks_environment

    # A developer's environment, not her runtime's: under her production governance 20 tests fail that pass at a shell.
    env = a_checks_environment(log_prefix="her-gate-logs-", PYTHON=sys.executable)
    tests = tests_that_touch(list(record.offer.get("files") or []), workspace)
    gates: list[tuple[str, list[str]]] = []
    if tests:
        # In one order, the files', so a red found here and asked again without the draft is asked in the same order.
        gates.append(("tests", ["nice", "-n", "10", sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                "-p", "no:randomly", *tests]))
    # A draft her yes takes lands on origin's main (approve), so it asks what a push there is asked: the gates below
    # and the governance and DEPS checks.
    gates += [(name, ["nice", "-n", "10", "make", name])
              for name in ("smoke", "lint", "layering", "writing", "governance-lint", "deps-check")]
    # Last, from clean checkouts, by code that shares nothing with her search: does the change keep what worked?
    # Replayed on 18 November 2026 it refused her overfit 052670e4a, which every gate above had passed. After the fifth
    # red team it holds three of her other landed changes and puts two on probation.
    outside = source_root() / "outside_bar" / "verify_change.py"
    # Each judging gate writes its verdict as data, which is what her yes decides by; its words are for people. The
    # outside check's verdict also carries what it counted as lost, for her next try to keep (what_her_change_must_keep).
    shown = Path(tempfile.mkdtemp(prefix="her-gate-results-"))
    if record.offer.get("commit") and outside.exists():
        gates.append(("outside-check", ["nice", "-n", "10", sys.executable, str(outside), "--repo", str(workspace),
                                        "--commit", str(record.offer["commit"]),
                                        "--result", str(shown / "outside-check.json")]))
        # And the same check over her real calls of what the draft changes, by the code she runs, not her draft's.
        gates.append(("her-real-calls", ["nice", "-n", "10", sys.executable, "-m", "core.self_modification.her_real_calls",
                                         "judge", "--repo", str(workspace), "--commit", str(record.offer["commit"]),
                                         "--result", str(shown / "her-real-calls.json")]))
    results: dict[str, dict[str, Any]] = {}
    for name, argv in gates:
        started = time.time()
        outcome, tail = _run_a_gate(argv, cwd=source_root() if name == "her-real-calls" else workspace, env=env,
                                    seconds=OUTSIDE_CHECK_SECONDS if name in JUDGING_GATES else GATE_SECONDS)
        before = _failing_without_the_draft(record, tail, env) if name == "tests" and outcome is False else []
        if before:
            outcome, tail = True, (f"{tail}\n\nEach test that failed fails on her source without the draft too, so none "
                                   f"is the draft's: {', '.join(before[:10])}")
        passed = outcome is True
        results[name] = {"passed": passed, "finished": outcome is not None, "seconds": round(time.time() - started, 1),
                         "tail": tail, **({"files": len(tests)} if name == "tests" else {})}
        if name in JUDGING_GATES:
            said = _the_verdict_written(shown / f"{name}.json")
            results[name]["verdict"] = str(said.get("verdict") or "")
            if name == "outside-check" and not passed:
                results[name]["must_keep"] = _what_the_outside_check_says_to_keep(said)
            elif name == "outside-check" and results[name]["verdict"] == "PROBATION":
                results[name]["what_it_changed"] = _what_the_outside_check_doubted(said)
                # What it doubted, in its words: probation reads whether the check measured it slower.
                results[name]["doubts"] = [str(one)[:300] for one in (said.get("doubts") or [])][:20]
        if not passed:
            break
    return results


#: The gates that judge the change itself, each writing its verdict as data (HELD, PROBATION, REFUSED, NOT_APPLICABLE).
JUDGING_GATES = ("outside-check", "her-real-calls")


def _the_verdict_written(path: Path) -> dict[str, Any]:
    """What a judging gate wrote as its verdict; {} when it wrote none, which her yes counts as a doubt."""
    try:
        said = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return said if isinstance(said, dict) else {}


def doubted_by(gates: dict[str, dict[str, Any]]) -> list[str]:
    """The judging gates that passed without a verdict that the change holds: PROBATION, or no verdict written at all.
    Decided from what each wrote as data; LIVE until 19 November it was read off the words "on probation:" in the
    last lines a gate printed."""
    return [name for name in JUDGING_GATES if name in gates and gates[name].get("passed")
            and gates[name].get("verdict") not in ("HELD", "NOT_APPLICABLE")]


def _what_the_outside_check_says_to_keep(said: dict[str, Any]) -> list[dict[str, Any]]:
    """The inputs the outside check counted as lost that can be written as a test; [] when it wrote none."""
    from core.self_modification.what_her_change_must_keep import keepable

    shown = said.get("counterexamples") or []
    return [one for one in shown if isinstance(one, dict) and keepable(one)] if isinstance(shown, list) else []


def _what_the_outside_check_doubted(said: dict[str, Any]) -> list[dict[str, Any]]:
    """The inputs the outside check saw the change answer differently and counted as a doubt, that a guard can be
    written of if the change survives probation; [] when it wrote none."""
    from core.self_modification.what_her_change_must_keep import guardable

    shown = said.get("counterexamples") or []
    return [one for one in shown if isinstance(one, dict) and guardable(one)] if isinstance(shown, list) else []


#: How long a fetch or a push to origin may take, in seconds: the bound her other git calls are given.
_ORIGIN_SECONDS = 120.0


def _replayed_on_origin(record: GoalRecord, commit: str) -> str | dict[str, Any]:
    """Her draft replayed onto origin's main in her workspace, and its tests run there: the new commit, or why not."""
    workspace = record.workspace
    replayed = _git("-c", "user.name=Aura", "-c", "user.email=aura@localhost", "rebase", "origin/main", cwd=workspace,
                    effect="workspace")
    if replayed.returncode != 0:
        _git("rebase", "--abort", cwd=workspace, effect="workspace")
        return {"why": f"her draft does not apply on origin's main: {replayed.stderr.strip()[:240]}"}
    tests = tests_that_touch(list(record.offer.get("files") or []), workspace)
    if tests:
        import sys

        from core.self_modification.a_checks_environment import a_checks_environment

        env = a_checks_environment(log_prefix="her-gate-logs-")
        passed, tail = _run_a_gate(["nice", "-n", "10", sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                    *tests], cwd=workspace, env=env)
        if not passed:
            return {"why": f"her draft, replayed on origin's main, fails its tests: {tail[-400:]}"}
    return _git("rev-parse", "HEAD", cwd=workspace).stdout.strip()


def _judged_again(record: GoalRecord, commit: str) -> tuple[str, list[dict[str, Any]]]:
    """The outside check's verdict on ``commit`` in her workspace, as written: HELD, PROBATION, REFUSED, or
    NOT_APPLICABLE where her source has no outside check, with the inputs it doubted. A verdict not written is "" and
    counts as a doubt."""
    import sys
    import tempfile

    outside = source_root() / "outside_bar" / "verify_change.py"
    if not outside.exists():
        return "NOT_APPLICABLE", []
    shown = Path(tempfile.mkdtemp(prefix="her-gate-results-")) / "outside-check-again.json"
    from core.self_modification.a_checks_environment import a_checks_environment

    env = a_checks_environment(log_prefix="her-gate-logs-")
    passed, _tail = _run_a_gate(["nice", "-n", "10", sys.executable, str(outside), "--repo", str(record.workspace),
                                 "--commit", commit, "--result", str(shown)], cwd=record.workspace, env=env)
    said = _the_verdict_written(shown)
    verdict = str(said.get("verdict") or "")
    if verdict:
        return verdict, _what_the_outside_check_doubted(said) if verdict == "PROBATION" else []
    # No verdict written: a check that failed is not taken to have held, and one that passed without saying is a doubt.
    return ("" if passed else "REFUSED"), []


def _onto_origin(record: GoalRecord, commit: str) -> dict[str, Any]:
    """Her draft on origin's main before it goes into the source she runs on, so the two cannot part.

    Her own yes took a draft into her source on disk only. A deploy fast-forwards her source to origin's main, so it
    would then have found the two parted. Where origin's main has moved past her draft's base, the draft is replayed
    onto it in her workspace and its tests run again there, and the replayed commit is what lands. Returns
    ``{"landed": commit}``, ``{"why": ...}`` when it could not land, or ``{}`` when her source has no origin.
    """
    root = source_root()
    if _git("remote", "get-url", "origin", cwd=root).returncode != 0:
        return {}
    refused = ""
    for _attempt in range(2):  # a second time only after a push that lost a race to another
        fetched = _git("fetch", "-q", "origin", "main", cwd=root, effect="fetch", timeout=_ORIGIN_SECONDS)
        if fetched.returncode != 0:
            return {"why": f"origin could not be read: {fetched.stderr.strip()[:240]}"}
        doubted_again = False
        doubted_on: list[dict[str, Any]] = []
        if _git("merge-base", "--is-ancestor", "origin/main", commit, cwd=root).returncode != 0:
            replayed = _replayed_on_origin(record, commit)
            if isinstance(replayed, dict):
                return replayed
            commit = replayed
            # What lands is the replayed commit, on source the gates never saw: it is judged again before it is pushed.
            verdict, doubted_on = _judged_again(record, commit)
            if verdict == "REFUSED":
                return {"why": "her draft, replayed on origin's main, is refused by the outside check there"}
            doubted_again = verdict not in ("HELD", "NOT_APPLICABLE")
            if doubted_again and not record.offer.get("on_probation") and change_on_probation() is not None:
                return {"why": "her draft, replayed on origin's main, is in doubt there, and another change of hers is "
                               "already on probation"}
        pushed = _git("push", "-q", "origin", f"{commit}:refs/heads/main", cwd=root, effect="push",
                      timeout=_ORIGIN_SECONDS)
        if pushed.returncode == 0:
            return {"landed": commit, **({"doubted_on_replay": True, "what_it_changed": doubted_on}
                                         if doubted_again else {})}
        refused = pushed.stderr.strip()
    return {"why": f"origin did not take her draft: {refused[:240]}"}


def _onto_disk(commit: str) -> bool:
    """Her landed commit into the source she runs on, when that adds nothing but hers.

    When origin's main moved past her source, fast-forwarding would also bring every commit between onto the disk of
    a running process, which imports some modules lazily. Then the disk takes it at the next deploy, and the running
    process has it from her loop (core/self_modification/into_the_running_process.py).
    """
    root = source_root()
    if _git("merge-base", "--is-ancestor", "HEAD", commit, cwd=root).returncode != 0:
        return False
    authors = _git("log", "--format=%an", f"HEAD..{commit}", cwd=root)
    if authors.returncode != 0 or any(author != "Aura" for author in authors.stdout.split()):
        return False
    return _git("merge", "--ff-only", commit, cwd=root, effect="merge").returncode == 0


def approve(slug: str, *, by: str = "Bryan", gates: dict[str, Any] | None = None) -> dict[str, Any]:
    """A yes, Bryan's or her own: her offered draft lands on origin's main and in the source she runs on.

    Her source must not have moved past the draft. When her source has an origin, the draft lands there first, so her
    source and origin cannot part (``_onto_origin``); then it goes onto her disk when that adds nothing but hers.
    """
    record = load(slug)
    if record is None or record.status != OFFERED:
        return {"ok": False, "why": f"no draft of hers waits under {slug}"}
    root = source_root()
    commit = str(record.offer.get("commit") or "")
    moved = _git("merge-base", "--is-ancestor", "HEAD", commit, cwd=root)
    if moved.returncode != 0:
        return {"ok": False, "why": "the source she runs on moved since her draft; she has to bring her draft up to it"}
    landed = _onto_origin(record, commit)
    if "why" in landed:
        record.said.append({"words": f"My yes could not land my draft: {landed['why']}", "at": time.time()})
        _write_now(record)
        logger.warning("her draft for '%s' did not land: %s", record.goal[:60], landed["why"][:200])
        return {"ok": False, "why": landed["why"]}
    if landed:
        commit, on_disk = landed["landed"], _onto_disk(landed["landed"])
        if landed.get("doubted_on_replay"):
            record.offer["on_probation"] = True
    else:
        merged = _git("merge", "--ff-only", commit, cwd=root, effect="merge")
        if merged.returncode != 0:
            return {"ok": False, "why": merged.stderr.strip()[:300]}
        on_disk = True
    record.status = MERGED
    record.offer["merged_at"] = time.time()
    record.offer["taken_by"] = by
    record.offer["commit"] = commit
    record.offer["on_origin"] = bool(landed)
    record.offer["on_disk"] = on_disk
    if gates is not None:
        record.offer["gates"] = gates  # the receipt of her own yes: what each gate said
    # What the outside check saw it answer differently, from the commit that landed: a guard is written of it if it
    # survives probation.
    changed = (landed.get("what_it_changed") if landed.get("doubted_on_replay")
               else ((gates or {}).get("outside-check") or {}).get("what_it_changed"))
    if changed:
        record.offer["what_it_changed"] = changed
    _write_now(record)
    logger.info("%s took her draft for '%s' (%s)%s%s", by, record.goal[:60], commit[:9],
                ", on origin" if landed else "", "" if on_disk else "; her disk takes it at the next deploy")
    return {"ok": True, "commit": commit, "on_origin": bool(landed), "on_disk": on_disk}


def take_it_back(slug: str, why: str) -> dict[str, Any]:
    """A draft of hers that was taken and did not hold, reverted the way it went in: on origin, then on her disk.

    The goal is then stalled on the source as it is, so her loop tries it again once that source has moved.
    """
    record = load(slug)
    if record is None or record.status != MERGED:
        return {"ok": False, "why": f"no taken draft of hers under {slug}"}
    root, workspace, commit = source_root(), ensure_workspace(record), str(record.offer.get("commit") or "")
    has_origin = _git("remote", "get-url", "origin", cwd=root).returncode == 0
    if has_origin and _git("fetch", "-q", "origin", "main", cwd=root, effect="fetch",
                           timeout=_ORIGIN_SECONDS).returncode != 0:
        return {"ok": False, "why": "origin could not be read"}
    base = "origin/main" if has_origin else _git("rev-parse", "HEAD", cwd=root).stdout.strip()
    steps = (("checkout", "-B", branch_of(slug), base),
             ("-c", "user.name=Aura", "-c", "user.email=aura@localhost", "revert", "--no-edit", commit))
    for step in steps:
        done = _git(*step, cwd=workspace, effect="workspace")
        if done.returncode != 0:
            return {"ok": False, "why": f"her revert could not be made: {done.stderr.strip()[:240]}"}
    revert = _git("rev-parse", "HEAD", cwd=workspace).stdout.strip()
    if has_origin:
        pushed = _git("push", "-q", "origin", f"{revert}:refs/heads/main", cwd=root, effect="push",
                      timeout=_ORIGIN_SECONDS)
        if pushed.returncode != 0:
            return {"ok": False, "why": f"origin did not take her revert: {pushed.stderr.strip()[:240]}"}
        on_disk = _onto_disk(revert)
    else:
        on_disk = _git("merge", "--ff-only", revert, cwd=root, effect="merge").returncode == 0
    record.status, record.stalled_on = STALLED, source_head()
    record.offer = {**record.offer, "reverted": revert, "reverted_at": time.time(), "reverted_on_disk": on_disk}
    record.said.append({"words": f"My mend did not hold, and I took it back out: {why}", "at": time.time()})
    _write_now(record)
    logger.warning("her mend for '%s' did not hold and was taken back out (%s): %s", record.goal[:60],
                   revert[:9], why[:160])
    return {"ok": True, "revert": revert, "on_disk": on_disk}


def note_on_offer(slug: str, **facts: Any) -> bool:
    """Facts about what became of a draft of hers, kept on its offer: what of it was loaded into her running process,
    when it was undone there and why. Her record of what she changed is read from these
    (core/agency/what_she_changed.py); whether they were written."""
    record = load(slug)
    if record is None or not record.offer:
        return False
    record.offer = {**record.offer, **facts}
    _write_now(record)
    return True


def mends_undone_on_record() -> list[str]:
    """Her taken drafts undone in her running process and still in her source, from her records: an undo heard
    before her last start is taken back after it."""
    return [record.slug for record in all_records()
            if record.status == MERGED and (record.offer or {}).get("undone") and not record.offer.get("reverted")]


def decline(slug: str, words: str) -> dict[str, Any]:
    """Bryan's no, in his words: the goal goes back to her with them."""
    record = load(slug)
    if record is None or record.status != OFFERED:
        return {"ok": False, "why": f"no draft of hers waits under {slug}"}
    record.notes_from_bryan.append({"words": str(words or "").strip(), "at": time.time(), "draft": record.offer})
    record.status = WORKING
    record.offer = {}
    record.quiet_passes = 0
    _write_now(record)
    return {"ok": True}


def _write_now(record: GoalRecord) -> None:
    """For Bryan's answers, given outside her event loop (tools/her_goals.py)."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    target = records_dir() / f"{_checked_slug(record.slug)}.json"
    with local_internal_governed_scope(
        "her_goal_at_work.answer",
        domain="state_mutation",
        receipt_prefix="her-goal-answer",
        constraints={"artifact": "aura.her_goal_record.v1", "operation": "replace"},
    ):
        get_file_write_gateway().write_text(target,
                                            json.dumps(record.to_dict(), indent=1, default=str),
                                            source="her_goal_at_work")


__all__ = ["CALLS_A_PASS", "GoalRecord", "MERGED", "OFFERED", "ORIGIN", "STALLED", "WORKING", "all_records",
           "approve", "decline", "ensure_workspace", "load", "save", "slug_of", "turns_from", "work_a_while"]
