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
- When she offers a draft the goal waits for Bryan. Only his approval moves her change into the source she runs on:
  her self-modification has always required a person behind it, and his approval is that person.
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
        from core.brain.llm_health_router import (
            _BACKGROUND_GENERATION_GATE_WAIT_S,
            acquire_external_generation_gate_lease,
            release_external_generation_gate_lease,
        )
        from core.runtime.service_access import resolve_inference_gate

        client = getattr(resolve_inference_gate(default=None), "_mlx_client", None)
        if client is None or not hasattr(client, "think_and_act"):
            raise RuntimeError("her cortex's tool loop is not available")
        lease = await acquire_external_generation_gate_lease(owner=f"{ORIGIN}:her_goal", timeout_s=PASS_SECONDS,
                                                             wait_s=_BACKGROUND_GENERATION_GATE_WAIT_S)
        if lease is None:
            return {"content": "", "turns": 0, "tool_calls": [], "deferred": True}
        try:
            return await asyncio.wait_for(client.think_and_act(objective, **kwargs), timeout=PASS_SECONDS)
        finally:
            release_external_generation_gate_lease(lease)


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


def _git(*args: str, cwd: Path, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", "core.fsmonitor=false", *args], cwd=str(cwd), capture_output=True,
                          text=True, timeout=timeout, check=False)


def ensure_workspace(record: GoalRecord) -> Path:
    """Her worktree for the goal, made from the source she runs on when it does not exist yet."""
    path = record.workspace
    if (path / ".git").exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    root = source_root()
    base = _git("rev-parse", "HEAD", cwd=root).stdout.strip()
    made = _git("worktree", "add", "-B", branch_of(record.slug), str(path), base, cwd=root)
    if made.returncode != 0:
        raise RuntimeError(f"her workspace could not be made: {made.stderr.strip()[:300]}")
    record.base = base
    logger.info("🛠️ her workspace for '%s': %s on %s from %s", record.goal[:60], path, branch_of(record.slug), base[:9])
    return path


def turns_from(steps: list[dict[str, Any]], room: int) -> list[dict[str, Any]]:
    """Her calls and what came back, as the loop's own messages, the latest that fit ``room`` characters."""
    kept: list[dict[str, Any]] = []
    used = 0
    for step in reversed(steps):
        result = str(step.get("result") or "")
        size = len(result) + len(json.dumps(step.get("args") or {}, default=str))
        if kept and used + size > room:
            break
        used += size
        call_id = str(step.get("id") or f"call_{len(steps) - len(kept) // 2}")
        kept[:0] = [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": str(step.get("tool") or ""), "arguments": dict(step.get("args") or {})}}]},
            {"role": "tool", "tool_call_id": call_id, "name": str(step.get("tool") or ""), "content": result},
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
        prior_turns=turns_from(record.steps, _what_she_reads_of_before()),
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
    record.last_worked_at = now
    await save(record)
    logger.info("🛠️ '%s': %d call(s) this pass (%s); %d in all; %s", goal[:60], len(calls),
                ", ".join(str(call.get("tool")) for call in calls[:6]) or "none", len(record.steps), record.status)
    return record


def approve(slug: str) -> dict[str, Any]:
    """Bryan's yes: her offered draft becomes part of the source she runs on, if the source has not moved past it."""
    record = load(slug)
    if record is None or record.status != OFFERED:
        return {"ok": False, "why": f"no draft of hers waits under {slug}"}
    root = source_root()
    commit = str(record.offer.get("commit") or "")
    moved = _git("merge-base", "--is-ancestor", "HEAD", commit, cwd=root)
    if moved.returncode != 0:
        return {"ok": False, "why": "the source she runs on moved since her draft; she has to bring her draft up to it"}
    merged = _git("merge", "--ff-only", commit, cwd=root)
    if merged.returncode != 0:
        return {"ok": False, "why": merged.stderr.strip()[:300]}
    record.status = MERGED
    record.offer["merged_at"] = time.time()
    _write_now(record)
    logger.info("🛠️ Bryan took her draft for '%s' (%s)", record.goal[:60], commit[:9])
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
