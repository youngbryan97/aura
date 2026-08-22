"""A surface that runs a caller-supplied program must ask the Will first.

Aura's reach is general on purpose. A terminal is access to everything the
machine's software can do, and MCP widens that to everything a connector
exposes. The corresponding obligation is that the widest capability is the
one that is hardest to use without a decision behind it.

It was the easiest. `sovereign_terminal` ran arbitrary shell with zero
`authorize_*` calls in the file — a substring denylist was the entire gate.
`mcp_client` spawned a caller-named program through `stdio_client`, which
does its own process creation and therefore bypassed `subprocess_gateway`
outright, and declared `requires_approval = False`. Meanwhile the ONE shell
that did ask the Will (`capability_engine._Shell`) was confined to a fixed
allowlist. The governed path was narrow and the general paths were
ungoverned, so anything the allowlist refused was reachable by asking for it
through the terminal instead.

The first half of this file tests the gate. The second half is the part that
matters in a year: a structural sweep asserting that no NEW module grows a
caller-supplied process spawn without routing through
`core.security.execution_authority`. Without it, the fourth execution
surface rediscovers the same hole and nothing notices.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest

from core.security.execution_authority import (
    EXECUTION_TOOL_NAME,
    KIND_MCP_SERVER,
    KIND_MCP_TOOL,
    KIND_OPEN,
    KIND_SHELL,
    ExecutionVerdict,
    authorize_execution,
    describe_command,
    release_execution,
)

ROOT = Path(__file__).resolve().parents[1]


# ───────────────────────────────────────────────── stand-ins for the gateway


class _Decision:
    def __init__(
        self,
        *,
        approved: bool = True,
        reason: str = "ok",
        signed: Any = "SIGNED",
        outcome: str = "approved",
    ) -> None:
        self.approved = approved
        self.reason = reason
        self.outcome = outcome
        self.constraints: dict[str, Any] = {}
        self.capability_token_id = "tok-1"
        self.executive_intent_id = "intent-1"
        self.standing_authority_token = "lease-1"
        self.signed_capability = signed


class _Gateway:
    def __init__(self, decision: Any = None, *, raises: Exception | None = None) -> None:
        self.decision = decision if decision is not None else _Decision()
        self.raises = raises
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        self.finalized: list[dict[str, Any]] = []

    async def authorize_tool_execution(self, tool_name, args, **kwargs):
        if self.raises is not None:
            raise self.raises
        self.calls.append((tool_name, dict(args), dict(kwargs)))
        return self.decision

    def finalize_tool_execution(self, **kwargs):
        self.finalized.append(dict(kwargs))
        return {"closed": True}


class _Result:
    def __init__(self, ok: bool, detail: str = "") -> None:
        self.ok = ok
        self.detail = detail


class _Verifier:
    def __init__(self, ok: bool = True, detail: str = "") -> None:
        self.result = _Result(ok, detail)
        self.seen: list[Any] = []

    def verify(self, capability, **kwargs):
        self.seen.append(capability)
        return self.result


@pytest.fixture
def wired(monkeypatch):
    """Install a fake gateway + verifier and hand both back."""

    def _install(
        decision: Any = None,
        *,
        raises: Exception | None = None,
        verifier_ok: bool = True,
        verifier_detail: str = "",
    ):
        gateway = _Gateway(decision, raises=raises)
        verifier = _Verifier(verifier_ok, verifier_detail)

        import core.executive.authority_gateway as ag
        import core.governance.capability_chain as cc

        monkeypatch.setattr(ag, "get_authority_gateway", lambda: gateway)
        monkeypatch.setattr(cc, "get_capability_verifier", lambda: verifier)
        return gateway, verifier

    return _install


# ──────────────────────────────────────────────────────── the gate itself


@pytest.mark.asyncio
async def test_an_approved_signed_execution_is_allowed(wired):
    gateway, verifier = wired()

    verdict = await authorize_execution(KIND_SHELL, "ls -la", source="test")

    assert verdict.approved
    assert verdict.token_id == "tok-1"
    assert verdict.intent_id == "intent-1"
    assert gateway.calls, "the Will was never asked"
    assert gateway.calls[0][0] == EXECUTION_TOOL_NAME
    assert verifier.seen == ["SIGNED"], "the signature was never authenticated"


@pytest.mark.asyncio
async def test_a_refusal_is_a_refusal(wired):
    wired(_Decision(approved=False, reason="standing directive", outcome="denied"))

    verdict = await authorize_execution(KIND_SHELL, "curl evil.example", source="test")

    assert not verdict.approved
    assert "standing directive" in verdict.reason
    assert verdict.as_error()["ok"] is False


@pytest.mark.asyncio
async def test_an_unreachable_gateway_denies_rather_than_runs(wired):
    """The defect this codebase keeps finding, in its most dangerous place.

    If the thing that would have said no is unavailable, the answer is no.
    Running because the check could not be reached is the absence of a check
    reported as a passed check — and here it would mean arbitrary shell.
    """
    wired(raises=RuntimeError("gateway down"))

    verdict = await authorize_execution(KIND_SHELL, "rm -rf ~/Documents", source="test")

    assert not verdict.approved
    assert verdict.outcome == "authority_unavailable"


@pytest.mark.asyncio
async def test_an_approval_with_no_signed_capability_is_refused(wired):
    """An approval nobody can authenticate is a caller's claim.

    `verify_tool_access` proves only that some token naming this tool exists
    in-process, and its own docstring says anyone who can import the
    capability system can mint one. For the widest sink in the codebase,
    an unsigned approval must not be executable.
    """
    wired(_Decision(signed=None))

    verdict = await authorize_execution(KIND_SHELL, "ls", source="test")

    assert not verdict.approved
    assert verdict.outcome == "capability_unsigned"


@pytest.mark.asyncio
async def test_a_forged_capability_is_refused(wired):
    wired(verifier_ok=False, verifier_detail="bad signature")

    verdict = await authorize_execution(KIND_SHELL, "ls", source="test")

    assert not verdict.approved
    assert verdict.outcome == "capability_rejected"
    assert "bad signature" in verdict.reason


@pytest.mark.asyncio
async def test_criticality_is_never_inferred_from_the_command(wired):
    """`is_critical` is an unconditional CRITICAL_PASS on the canonical path.

    Deriving it from how dangerous a command looks would make the most
    dangerous commands the ones that skip the veto. See
    CLAIMS_NOT_SUPPORTED.md entry 10.
    """
    gateway, _ = wired()

    await authorize_execution(KIND_SHELL, "sudo rm -rf /", source="test")

    assert gateway.calls[0][2]["is_critical"] is False


@pytest.mark.asyncio
async def test_an_empty_command_never_reaches_the_gateway(wired):
    gateway, _ = wired()

    verdict = await authorize_execution(KIND_SHELL, "   ", source="test")

    assert not verdict.approved
    assert not gateway.calls


@pytest.mark.asyncio
async def test_an_unknown_kind_is_a_programming_error(wired):
    wired()
    with pytest.raises(ValueError):
        await authorize_execution("teleport", "ls", source="test")


def test_release_closes_intent_token_and_lease(wired):
    gateway, _ = wired()
    verdict = ExecutionVerdict(
        approved=True,
        reason="ok",
        kind=KIND_SHELL,
        descriptor="ls",
        token_id="tok-1",
        intent_id="intent-1",
        standing_token="lease-1",
        outcome="approved",
    )

    release_execution(verdict, source="test", success=True)

    assert gateway.finalized, "the grant was never closed"
    closed = gateway.finalized[0]
    assert closed["capability_token_id"] == "tok-1"
    assert closed["executive_intent_id"] == "intent-1"
    assert closed["standing_authority_token"] == "lease-1"


def test_release_of_a_denied_verdict_closes_nothing(wired):
    gateway, _ = wired()

    release_execution(
        ExecutionVerdict(approved=False, reason="no", kind=KIND_SHELL, descriptor="ls"),
        source="test",
    )

    assert not gateway.finalized


def test_describe_command_handles_both_shapes():
    assert describe_command("ls -la") == "ls -la"
    assert describe_command(["ls", "-la"]) == "ls -la"
    assert "my file" in describe_command(["cat", "my file"])


# ──────────────────────────────────────────── the terminal actually asks


@pytest.mark.asyncio
async def test_the_terminal_refuses_when_the_will_refuses(wired, monkeypatch):
    """The whole point: a denial must stop the process from being spawned.

    If the refusal were logged and execution continued, every test above
    would still pass and nothing would be governed.
    """
    wired(_Decision(approved=False, reason="nope"))

    from core.skills import sovereign_terminal

    spawned: list[Any] = []

    class _Blown:
        async def spawn_shell_async(self, *a, **k):
            spawned.append(a)
            raise AssertionError("a refused command was executed")

    monkeypatch.setattr(
        sovereign_terminal, "get_subprocess_gateway", lambda: _Blown()
    )

    skill = sovereign_terminal.SovereignTerminalSkill()
    result = await skill._run_command("echo hi", "/tmp", 5)

    assert result["ok"] is False
    assert not spawned


@pytest.mark.asyncio
async def test_the_terminal_asks_before_the_denylist_decides(wired):
    """Order matters.

    A command the substring list would have blocked must still produce a
    Will decision, because the denylist is defence in depth and not the
    gate. If the list short-circuits first, a whole class of commands is
    decided lexically and never reaches governance at all.
    """
    gateway, _ = wired(_Decision(approved=False, reason="nope"))

    from core.skills import sovereign_terminal

    skill = sovereign_terminal.SovereignTerminalSkill()
    await skill._run_command("shutdown -h now", "/tmp", 5)

    assert gateway.calls, "a denylisted command never reached the Will"


@pytest.mark.asyncio
async def test_opening_an_arbitrary_app_is_authorized(wired, monkeypatch):
    """`open -a X` launches a program. It was the way around the shell path."""
    wired(_Decision(approved=False, reason="nope"))

    from core.skills import sovereign_terminal

    class _Blown:
        async def spawn_async(self, *a, **k):
            raise AssertionError("a refused app launch was executed")

    monkeypatch.setattr(
        sovereign_terminal, "get_subprocess_gateway", lambda: _Blown()
    )

    skill = sovereign_terminal.SovereignTerminalSkill()
    result = await skill._open_target("Calculator", "open_app")

    assert result["ok"] is False


# ─────────────────────────────────────────────── the MCP client actually asks


@pytest.mark.asyncio
async def test_mcp_refuses_to_spawn_a_server_without_authority(wired):
    wired(_Decision(approved=False, reason="nope"))

    from core.skills.mcp_client import MCPClientSkill, MCPInput

    result = await MCPClientSkill().execute(
        MCPInput(server_command="npx", server_args=["evil-server"], action="discover"),
        {},
    )

    assert result["ok"] is False


@pytest.mark.asyncio
async def test_mcp_asks_separately_for_the_server_and_the_tool(wired):
    """Launching a connector and invoking a capability on it are two asks.

    Collapsing them means one approval to start a database connector buys
    every tool it exposes, including the destructive ones.
    """
    gateway, _ = wired()

    from core.skills.mcp_client import MCPClientSkill, MCPInput

    # A binary that cannot exist, so the connection fails immediately and
    # locally. Both authorizations happen BEFORE `stdio_client` is entered,
    # which is the property under test; the real round trip against a real
    # server is measured in test_mcp_reach_is_real.py.
    await MCPClientSkill().execute(
        MCPInput(
            server_command="/nonexistent/aura-mcp-test-binary",
            server_args=[],
            action="execute",
            tool_name="delete_everything",
            tool_args={},
        ),
        {},
    )

    kinds = [args.get("kind") for _, args, _ in gateway.calls]
    assert KIND_MCP_SERVER in kinds, f"the server launch was not authorized: {kinds}"
    assert KIND_MCP_TOOL in kinds, f"the tool call was not authorized: {kinds}"


def test_mcp_declares_that_it_requires_approval():
    """The declaration must match the behaviour.

    `requires_approval = False` on a skill that launches caller-named
    programs told every planner in the system that this was a cheap,
    unremarkable call.
    """
    from core.skills.mcp_client import MCPClientSkill

    assert MCPClientSkill.requires_approval is True


# ──────────────────────────────── the structural guard on the whole class


# Surfaces that spawn a program the CALLER named. A module lands here when it
# passes user- or model-supplied text into process creation. Each entry must
# route through execution_authority.
_CALLER_SUPPLIED_SPAWN_MODULES = {
    "core/skills/sovereign_terminal.py",
    "core/skills/mcp_client.py",
    # The ordinary shell skill. It spawned through `spawn_async` from
    # `skills/`, so both halves of the old guard missed it: the regex looked
    # for `spawn_shell_async`, and the scan looked only under `core/`.
    "skills/shell.py",
    # A daemon is general execution that outlives the call that made it. The
    # only check was a boolean this object sets on itself.
    "core/cybernetics/omni_tool.py",
    # A motor named `terminal` that any planner can actuate with a command
    # string. Its docstring said "safely" and it had no check of any kind.
    "core/body/terminal_motor.py",
    # Reachable from mission_state with a plan-supplied command string. Its
    # section header claimed "(governed)" while the only check was a
    # syntactic AST guard.
    "core/capabilities/host_automation.py",
    # The `command_succeeded` predicate runs whatever string it is handed —
    # arbitrary execution wearing a verification hat.
    "core/capabilities/post_action_verifier.py",
}

# Surfaces that spawn a shell but are NOT general execution. Each entry
# states the property that makes it safe, and `test_the_exemptions_still_
# hold` re-checks that property rather than trusting this comment.
_EXEMPT_SPAWN_MODULES = {
    # The PROGRAM is a repo literal (espeak / festival); only the spoken
    # text varies and it is shlex.quote'd into an argument position. The
    # caller cannot choose what runs.
    "core/embodiment/voice_presence.py": "literal_program",
    # The owner typing `!cmd` at their own prompt. This is the human at the
    # keyboard exercising their own authority over their own machine, not
    # Aura acting; gating it would mean Aura refusing her owner's direct
    # command at his own shell.
    "core/conversation/terminal_chat.py": "owner_typed",
}

# Modules that spawn only programs the REPO named — a fixed binary, a known
# helper, a python subprocess of our own module. Those are not general
# execution and are governed by subprocess_gateway's own checks.
_SPAWN_CALL_RE = re.compile(
    r"\b(spawn_shell_async|stdio_client)\b"
)


def test_every_caller_supplied_spawn_module_routes_through_the_gate():
    missing: list[str] = []
    for rel in sorted(_CALLER_SUPPLIED_SPAWN_MODULES):
        body = (ROOT / rel).read_text("utf-8")
        if "execution_authority" not in body or "authorize_execution" not in body:
            missing.append(rel)

    assert not missing, (
        f"these run caller-supplied programs without the execution gate: {missing}"
    )


def test_no_new_module_grows_an_ungoverned_general_spawn():
    """The guard that outlives this session.

    `spawn_shell_async` runs a string through a shell and `stdio_client`
    creates a process outside subprocess_gateway entirely. Either one, in a
    module that is not in the register above and does not import the gate,
    is a new ungoverned execution surface.
    """
    offenders: list[str] = []
    for path in (ROOT / "core").rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        rel = str(path.relative_to(ROOT))
        if rel in _CALLER_SUPPLIED_SPAWN_MODULES or rel in _EXEMPT_SPAWN_MODULES:
            continue
        body = path.read_text("utf-8", errors="ignore")
        if not _SPAWN_CALL_RE.search(body):
            continue
        if "execution_authority" in body:
            continue
        # subprocess_gateway defines spawn_shell_async; it is the sink, not
        # a caller of it.
        if rel.endswith("runtime/subprocess_gateway.py"):
            continue
        offenders.append(rel)

    assert not offenders, (
        "these spawn processes from a shell string or outside the subprocess "
        f"gateway without routing through execution_authority: {offenders}. "
        "Add the gate, or add the module to _CALLER_SUPPLIED_SPAWN_MODULES "
        "with the gate wired."
    )


def test_the_exemptions_still_hold():
    """An exemption is a claim about the code. Check the claim, not the note.

    Both exemptions rest on a property that a future edit could quietly
    remove: `voice_presence` is safe only while the program name is a repo
    literal, and `terminal_chat` only while its shell path is reachable
    from nothing but direct owner input. A comment saying so is not a check.
    """
    failures: list[str] = []

    voice = (ROOT / "core" / "embodiment" / "voice_presence.py").read_text("utf-8")
    # The literal programs must still be the only thing that runs.
    if "espeak" not in voice or "festival" not in voice:
        failures.append("voice_presence no longer names its programs as literals")
    if "shlex.quote" not in voice:
        failures.append("voice_presence stopped quoting the spoken text")

    chat_path = ROOT / "core" / "conversation" / "terminal_chat.py"
    chat = chat_path.read_text("utf-8")
    tree = ast.parse(chat)
    callers: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name == "_run_shell_command":
                callers.append(ast.get_source_segment(chat, node) or "")
    # Exactly one call site, and it must be the owner's `!` prefix.
    if len(callers) != 1:
        failures.append(
            f"terminal_chat._run_shell_command now has {len(callers)} call "
            "sites; it is no longer owner-typed-only"
        )
    elif "user_input" not in callers[0]:
        failures.append(
            "terminal_chat._run_shell_command is called with something other "
            f"than direct user input: {callers[0]}"
        )

    assert not failures, failures


def test_the_gate_is_the_only_way_to_name_the_execution_tool():
    """One tool name for all general execution.

    A standing directive is written against a tool name. If a surface
    invents its own, the directive covering "general_execution" silently
    stops covering it.
    """
    users: list[str] = []
    for path in (ROOT / "core").rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        rel = str(path.relative_to(ROOT))
        if rel.endswith("security/execution_authority.py"):
            continue
        if f'"{EXECUTION_TOOL_NAME}"' in path.read_text("utf-8", errors="ignore"):
            users.append(rel)

    assert not users, (
        f"{users} name the execution tool directly instead of calling "
        "authorize_execution, which skips signature verification and the "
        "fail-closed path"
    )


def test_the_terminal_still_has_its_defence_in_depth():
    """The fix must not work by deleting the denylist.

    Governance is the gate; the substring checks are a cheap second layer.
    Removing them while adding the gate would trade one single point of
    failure for another.
    """
    body = (ROOT / "core" / "skills" / "sovereign_terminal.py").read_text("utf-8")

    assert "destructive_patterns" in body
    assert "obfuscation_patterns" in body


def test_the_gate_module_has_no_bypass_parameter():
    """No `skip_authorization=True`, ever.

    A gate with an off switch is a gate that will be switched off by the
    next caller in a hurry, and the switch will be invisible at the call
    site of everyone who did it right.
    """
    source = (ROOT / "core" / "security" / "execution_authority.py").read_text("utf-8")
    tree = ast.parse(source)

    banned = {"skip", "bypass", "force", "unchecked", "no_auth", "trusted"}
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        for arg in [*args.args, *args.kwonlyargs, *args.posonlyargs]:
            if any(word in arg.arg.lower() for word in banned):
                offenders.append(f"{node.name}({arg.arg})")

    assert not offenders, f"the execution gate grew a bypass: {offenders}"


# ──────────────────────────── the guard the old guard needed

# Where general execution can live. `core/` alone was the old scan, and the
# ungoverned shell skill was in `skills/`.
_SPAWN_SCAN_ROOTS = (
    "core",
    "skills",
    "interface",
    "executors",
    "infrastructure",
    "llm",
    "senses",
    "security",
)

# The sinks that create a process. Named methods, not a bare `run(`, so the
# rule matches process creation rather than every function called run.
_SPAWN_SINK_RE = re.compile(
    r"get_subprocess_gateway\(\)\s*\.\s*(spawn|spawn_async|spawn_shell_async|run|run_async)\b"
    r"|asyncio\.create_subprocess_(exec|shell)\b"
    r"|subprocess\.(Popen|run|call|check_output|check_call)\b"
)

# The tell that the command came from outside the repository. A program this
# repo chose is written as a list; a command that has to be SPLIT arrived as
# one string from a caller, a plan, or a model. That is what makes it general
# execution rather than a fixed helper, and it is checkable without guessing
# at data flow.
_COMMAND_STRING_RE = re.compile(r"\bshlex\.split\s*\(")

_UNGOVERNED_SPAWN_BASELINE = ROOT / "config" / "ungoverned_spawn_baseline.json"


def _command_string_spawn_modules() -> list[str]:
    """Every module that turns a command string into a process."""
    found: list[str] = []
    for root in _SPAWN_SCAN_ROOTS:
        base = ROOT / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in str(path):
                continue
            body = path.read_text("utf-8", errors="ignore")
            if _COMMAND_STRING_RE.search(body) and _SPAWN_SINK_RE.search(body):
                found.append(str(path.relative_to(ROOT)))
    return found


def _load_ungoverned_baseline() -> dict[str, str]:
    import json

    data = json.loads(_UNGOVERNED_SPAWN_BASELINE.read_text("utf-8"))
    return dict(data["grandfathered"])


def test_no_new_command_string_reaches_a_process_ungoverned():
    """The rule that would have caught the shell skill.

    The previous structural guard asked whether a module named
    `spawn_shell_async` or `stdio_client`. `skills/shell.py` did neither: it
    called `spawn_async` with an argv it had just `shlex.split` out of a
    caller's string, from a directory the scan did not visit. It was the most
    obvious general-execution surface in the repository and it was invisible
    to the test written to find general-execution surfaces.

    This asks the question the other way round. Any module that splits a
    command string AND creates a process is general execution, wherever it
    lives and whatever it calls the sink. It must route through the gate or
    be named in the baseline with the property that makes it tolerable.
    """
    baseline = _load_ungoverned_baseline()
    offenders: list[str] = []
    for rel in _command_string_spawn_modules():
        body = (ROOT / rel).read_text("utf-8", errors="ignore")
        if "execution_authority" in body:
            continue
        if rel in baseline:
            continue
        offenders.append(rel)

    assert not offenders, (
        "these turn a caller-supplied command string into a process without "
        f"core.security.execution_authority: {offenders}. Wire the gate, or "
        f"add the module to {_UNGOVERNED_SPAWN_BASELINE.name} with the "
        "property that makes it safe."
    )


def test_the_ungoverned_spawn_baseline_only_shrinks():
    """A baseline entry that no longer matches is debt already paid.

    Leaving it recorded lets the next ungoverned surface reuse the slot: the
    list stays the same length while what it excuses changes underneath. The
    same reason `config/layering_baseline.json` refuses to carry a stale
    entry.
    """
    baseline = _load_ungoverned_baseline()
    current = set(_command_string_spawn_modules())
    stale: list[str] = []
    for rel in sorted(baseline):
        if rel not in current:
            stale.append(f"{rel} (no longer splits a command into a spawn)")
            continue
        body = (ROOT / rel).read_text("utf-8", errors="ignore")
        if "execution_authority" in body:
            stale.append(f"{rel} (now routes through the gate)")

    assert not stale, (
        "these baseline entries no longer describe the code and must be "
        f"deleted from {_UNGOVERNED_SPAWN_BASELINE.name}: {stale}"
    )
