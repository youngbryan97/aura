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

import asyncio
import hashlib
import json
import logging
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

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


#: Where a goal stands.
WORKING, OFFERED, MERGED, STALLED = "working", "offered", "merged", "stalled"


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
        content, generations, empty = "", 0, 0
        while turns_left > 0 and empty < 2:
            if a_person_is_waiting_for_her_model():
                logger.info("🛠️ her pass stands aside after %d generation(s): a person is talking with her", generations)
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
                    client.think_and_act(objective, max_turns=1, prior_turns=list(prior), **kwargs), timeout=left_s)
            finally:
                release_external_generation_gate_lease(lease)
            generations += 1
            turns_left -= 1
            made = list((step or {}).get("tool_calls") or [])
            if not made:
                content = str((step or {}).get("content") or "")
                if content.strip():
                    break  # she answered in words, which ends a pass as it always did
                empty += 1  # nothing came back; once more, then the pass ends
                continue
            empty = 0
            calls += made
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
    notes_from_bryan: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    last_worked_at: float = 0.0
    #: The commit of her source when the goal stalled; a goal stalled on older source is worked again.
    stalled_on: str = ""

    @property
    def workspace(self) -> Path:
        return workspaces_dir() / self.slug

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoalRecord:
        known = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        return cls(**known)


def load(slug: str) -> GoalRecord | None:
    path = records_dir() / f"{slug}.json"
    try:
        return GoalRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError) as exc:
        record_degradation("her_goal_at_work", exc, severity="warning", action=f"read no record for {slug}")
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
    with local_internal_governed_scope(
        "her_goal_at_work.record",
        domain="state_mutation",
        receipt_prefix="her-goal-record",
        constraints={"artifact": "aura.her_goal_record.v1", "operation": "replace"},
    ):
        await gateway.ensure_directory_async(records_dir(), source="her_goal_at_work")
        await gateway.write_text_async(records_dir() / f"{record.slug}.json",
                                       json.dumps(record.to_dict(), indent=1, default=str), source="her_goal_at_work")


#: The governed domain of each effect git has in her goal work; a call with none of these only reads.
_GIT_EFFECTS = {"workspace": "file_write", "merge": "self_modification"}


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
    logger.info("🛠️ her workspace for '%s': %s on %s from %s", record.goal[:60], path, branch_of(record.slug), base[:9])
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
        logger.info("🛠️ her workspace for '%s' stays at %s: %s", record.goal[:60], record.base[:9],
                    moved.stderr.strip()[:160])
        return
    logger.info("🛠️ her workspace for '%s' brought up from %s to %s", record.goal[:60], record.base[:9], head[:9])
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


def what_she_works_toward(record: GoalRecord) -> str:
    """The goal as she set it, and what Bryan said of a draft of hers he did not take, in his words."""
    told = [f"Bryan, on your draft: {note.get('words')}" for note in record.notes_from_bryan[-3:] if note.get("words")]
    return "\n\n".join([record.goal, *told])


async def work_a_while(goal: str, *, router: Any = None) -> GoalRecord:
    """One pass at ``goal``: up to CALLS_A_PASS calls of hers, kept with what came back."""
    from core.skills.her_draft import tool_definitions

    slug = slug_of(goal)
    record = load(slug) or GoalRecord(goal=str(goal), slug=slug)
    if record.status == OFFERED:
        # A draft held while she was measured, or offered before she could take it herself.
        await asyncio.to_thread(take_it_herself, slug)
        record = load(slug) or record
    if record.status == STALLED:
        # A goal stalls after two passes with no call, and on day run 4 (9 and 10 November 2026) every stall came of a
        # fault in what she was given: her tools refused, her model held. Stalled for good, a goal waited on a fix
        # that had already landed. On source changed since it stalled, she works it again.
        head = await asyncio.to_thread(source_head)
        if head and head != record.stalled_on:
            logger.info("🛠️ '%s' worked again: her source moved from %s to %s since it stalled", goal[:60],
                        record.stalled_on[:9] or "before this was kept", head[:9])
            record.status, record.quiet_passes = WORKING, 0
            await save(record)
    if record.status != WORKING:
        logger.info("🛠️ '%s' is %s; no pass", goal[:60], record.status)
        return record
    router = router or HerCortex()
    workspace = await asyncio.to_thread(ensure_workspace, record)
    await save(record)  # her tools read it during the pass: edit_draft, check_draft and offer_draft
    result = await router.think_and_act(
        what_she_works_toward(record),
        system_prompt="",
        tools=tool_definitions(),
        max_turns=CALLS_A_PASS,
        context={"origin": ORIGIN, "her_goal": slug, "her_goal_workspace": str(workspace), "requested_by": "aura"},
        origin=ORIGIN,
        prior_turns=turns_from(record.steps, _what_she_reads_of_before(), record.said),
        work_in_background=True,
        native_tools_only=True,
    )
    calls = list((result or {}).get("tool_calls") or [])
    words = str((result or {}).get("content") or "").strip()
    if not calls and not words and not (result or {}).get("turns"):
        logger.info("🛠️ '%s': not now (her model is held for something else)", goal[:60])
        return record
    now = time.time()
    record = load(slug) or record  # a tool of hers may have changed the record during the pass
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
    logger.info("🛠️ '%s': %d call(s) this pass (%s); %d in all; %s", goal[:60], len(calls),
                ", ".join(str(call.get("tool")) for call in calls[:6]) or "none", len(record.steps), record.status)
    if record.status == OFFERED:
        # Offered this pass: her own yes, and the gates it asks, outside her model's turn.
        taken = await asyncio.to_thread(take_it_herself, slug)
        record = load(slug) or record
        if not taken.get("ok") and record.status == OFFERED:
            record.said.append({"words": f"My draft waits: {taken.get('why')}", "at": time.time()})
            await save(record)
    return record


#: Paths whose change waits for Bryan however well it checks: what she may do, how she is checked, and these rules.
HER_OWN_GOVERNANCE = (
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

#: While this file exists a run is measuring her, and no draft of hers is taken (tools/day_run_4.sh makes it).
MEASURING_HOLD = Path.home() / ".aura" / "measuring.hold"


def governs_her(path: str) -> bool:
    name = str(path or "").strip().removeprefix("./")
    return any(name == rule or (rule.endswith("/") and name.startswith(rule)) for rule in HER_OWN_GOVERNANCE) \
        or name.endswith("/DEPS") or name == "DEPS"


def take_it_herself(slug: str) -> dict[str, Any]:
    """Her offered draft into the source she runs on, without Bryan, unless it governs her or she is being measured."""
    record = load(slug)
    if record is None or record.status != OFFERED:
        return {"ok": False, "why": f"no draft of hers waits under {slug}"}
    governing = [name for name in record.offer.get("files") or [] if governs_her(name)]
    if governing:
        logger.warning("🛠️ her draft for '%s' changes her own governance (%s); it waits for Bryan",
                       record.goal[:60], ", ".join(governing[:4]))
        return {"ok": False, "why": "it changes her own governance, so it waits for Bryan", "files": governing}
    if MEASURING_HOLD.exists():
        logger.info("🛠️ her draft for '%s' waits: a run is measuring her", record.goal[:60])
        return {"ok": False, "why": "a run is measuring her; her draft waits until it ends"}
    gates = the_gates_her_yes_asks(record)
    failed = next((name for name, gate in gates.items() if not gate["passed"]), "")
    if failed:
        # Back to her with what failed, in the words of the gate: her next pass reads it.
        record = load(slug) or record
        record.status, record.quiet_passes = WORKING, 0
        record.said.append({"words": f"My own yes asked the gates first, and {failed} failed:\n"
                                     f"{gates[failed]['tail']}", "at": time.time()})
        record.offer = {**record.offer, "gates": gates}
        _write_now(record)
        logger.warning("🛠️ her draft for '%s' did not pass %s; it is hers to mend", record.goal[:60], failed)
        return {"ok": False, "why": f"{failed} failed", "gates": gates}
    return approve(slug, by="herself", gates=gates)


#: How long one gate may take, in seconds: the bound the repository's own test runner gives one pytest process
#: (tools/run_test_chunks.py, --chunk-timeout).
GATE_SECONDS = 2400.0


def tests_that_touch(changed: list[str], workspace: Path) -> list[str]:
    """The test files that name a changed module, and the changed test files themselves."""
    tests = {name for name in changed if Path(name).name.startswith("test_") and name.endswith(".py")}
    for name in changed:
        if not name.endswith(".py") or Path(name).name.startswith("test_") or not (workspace / name).exists():
            continue
        dotted = name[:-3].replace("/", ".")
        found = _git("grep", "-l", "-F", "-e", dotted, "-e", name, "--", "tests", cwd=workspace, timeout=60.0)
        tests.update(line.strip() for line in found.stdout.splitlines()
                     if Path(line.strip()).name.startswith("test_") and line.strip().endswith(".py"))
    return sorted(tests)


def _run_a_gate(argv: list[str], *, cwd: Path, env: dict[str, str]) -> tuple[bool, str]:
    from core.governance_context import local_internal_governed_scope
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    with local_internal_governed_scope("her_goal_at_work.gate", domain="tool_execution",
                                       receipt_prefix="her-goal-gate",
                                       constraints={"artifact": "aura.her_goal_gate.v1", "operation": "check"}):
        # The offline suite and the tree's gates load no model.
        done = get_subprocess_gateway().run(argv, cwd=cwd, env=env, timeout=GATE_SECONDS, capture_output=True,
                                            source="her_goal_at_work.gate", accelerator_capability="none")
    out = (done.stdout or "") + (done.stderr or "")
    return done.returncode == 0, out.strip()[-1500:]


def the_gates_her_yes_asks(record: GoalRecord) -> dict[str, dict[str, Any]]:
    """What her own yes asks of a draft before it goes into the source she runs on, in her workspace (F05).

    Her check_draft runs the test files her draft changed or named, so a draft that changed only a module and named no
    test passed on its compile and lint alone. Before she takes one herself it now passes the tests that name each
    module it changed, the smoke suite, lint, layering and the writing gate. They run in the order given and stop at the
    first that fails. Bryan's yes is his own judgment and asks none.
    """
    import os
    import sys
    import tempfile

    workspace = record.workspace
    env = {**os.environ, "PYTHON": sys.executable, "AURA_LOG_DIR": tempfile.mkdtemp(prefix="her-gate-logs-"),
           "VECLIB_MAXIMUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    tests = tests_that_touch(list(record.offer.get("files") or []), workspace)
    gates: list[tuple[str, list[str]]] = []
    if tests:
        gates.append(("tests", ["nice", "-n", "10", sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                *tests]))
    gates += [(name, ["nice", "-n", "10", "make", name]) for name in ("smoke", "lint", "layering", "writing")]
    results: dict[str, dict[str, Any]] = {}
    for name, argv in gates:
        started = time.time()
        passed, tail = _run_a_gate(argv, cwd=workspace, env=env)
        results[name] = {"passed": passed, "seconds": round(time.time() - started, 1), "tail": tail,
                         **({"files": len(tests)} if name == "tests" else {})}
        if not passed:
            break
    return results


def approve(slug: str, *, by: str = "Bryan", gates: dict[str, Any] | None = None) -> dict[str, Any]:
    """A yes, Bryan's or her own: her offered draft becomes part of the source she runs on, if it has not moved past it."""
    record = load(slug)
    if record is None or record.status != OFFERED:
        return {"ok": False, "why": f"no draft of hers waits under {slug}"}
    root = source_root()
    commit = str(record.offer.get("commit") or "")
    moved = _git("merge-base", "--is-ancestor", "HEAD", commit, cwd=root)
    if moved.returncode != 0:
        return {"ok": False, "why": "the source she runs on moved since her draft; she has to bring her draft up to it"}
    merged = _git("merge", "--ff-only", commit, cwd=root, effect="merge")
    if merged.returncode != 0:
        return {"ok": False, "why": merged.stderr.strip()[:300]}
    record.status = MERGED
    record.offer["merged_at"] = time.time()
    record.offer["taken_by"] = by
    if gates is not None:
        record.offer["gates"] = gates  # the receipt of her own yes: what each gate said
    _write_now(record)
    logger.info("🛠️ %s took her draft for '%s' (%s)", by, record.goal[:60], commit[:9])
    return {"ok": True, "commit": commit}


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

    with local_internal_governed_scope(
        "her_goal_at_work.answer",
        domain="state_mutation",
        receipt_prefix="her-goal-answer",
        constraints={"artifact": "aura.her_goal_record.v1", "operation": "replace"},
    ):
        get_file_write_gateway().write_text(records_dir() / f"{record.slug}.json",
                                            json.dumps(record.to_dict(), indent=1, default=str),
                                            source="her_goal_at_work")


__all__ = ["CALLS_A_PASS", "GoalRecord", "MERGED", "OFFERED", "ORIGIN", "STALLED", "WORKING", "all_records",
           "approve", "decline", "ensure_workspace", "load", "save", "slug_of", "turns_from", "work_a_while"]
