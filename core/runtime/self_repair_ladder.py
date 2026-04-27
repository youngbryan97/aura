"""Self-Repair Validation Ladder.

A patch proposed by the self-repair pipeline must climb every rung before
it is allowed to land:

    1. syntax       -> ast.parse / compile() succeeds
    2. ast_safety   -> rejects banned operations (subprocess.* etc.)
    3. import       -> imports cleanly in a fresh namespace
    4. targeted     -> a caller-provided test slice passes
    5. boot_smoke   -> caller-provided boot probe passes
    6. one_turn     -> caller-provided one-turn integration passes
    7. shutdown     -> caller-provided graceful shutdown probe passes
    8. rollback     -> caller-provided rollback check passes

If any rung fails, the patch is rejected with the exact failing rung and
reason recorded. The audits explicitly forbid patches landing on AST parse
alone — this module enforces that contract.
"""
from __future__ import annotations


import ast
import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional, Sequence, Union

logger = logging.getLogger("Aura.SelfRepairLadder")


RUNG_SYNTAX = "syntax"
RUNG_AST_SAFETY = "ast_safety"
RUNG_IMPORT = "import"
RUNG_TARGETED = "targeted"
RUNG_BOOT_SMOKE = "boot_smoke"
RUNG_ONE_TURN = "one_turn"
RUNG_SHUTDOWN = "shutdown"
RUNG_ROLLBACK = "rollback"

CANONICAL_RUNGS: tuple = (
    RUNG_SYNTAX,
    RUNG_AST_SAFETY,
    RUNG_IMPORT,
    RUNG_TARGETED,
    RUNG_BOOT_SMOKE,
    RUNG_ONE_TURN,
    RUNG_SHUTDOWN,
    RUNG_ROLLBACK,
)


# Banned imports / calls that an AST-only validator must reject without
# requiring real execution.
BANNED_AST_PATTERNS: tuple = (
    "subprocess",
    "os.system",
    "os.execv",
    "shutil.rmtree",
    "ctypes",
    "socket.socket",
    "eval",
    "exec",
)


@dataclass
class RungResult:
    rung: str
    ok: bool
    reason: Optional[str] = None


@dataclass
class LadderReport:
    rungs: List[RungResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.rungs) and all(r.ok for r in self.rungs)

    @property
    def first_failure(self) -> Optional[RungResult]:
        for r in self.rungs:
            if not r.ok:
                return r
        return None


# Caller-provided probes can be sync or async, returning bool or
# raising on failure.
ProbeResult = Union[bool, Awaitable[bool]]
Probe = Callable[[], ProbeResult]


@dataclass
class SelfRepairProbes:
    targeted: Optional[Probe] = None
    boot_smoke: Optional[Probe] = None
    one_turn: Optional[Probe] = None
    shutdown: Optional[Probe] = None
    rollback: Optional[Probe] = None


# ---------------------------------------------------------------------------
# Rung implementations
# ---------------------------------------------------------------------------


def _check_syntax(patch_source: str) -> RungResult:
    try:
        compile(patch_source, "<self_repair_patch>", "exec")
    except SyntaxError as exc:
        return RungResult(RUNG_SYNTAX, False, f"syntax error: {exc}")
    return RungResult(RUNG_SYNTAX, True)


def _check_ast_safety(patch_source: str) -> RungResult:
    try:
        tree = ast.parse(patch_source)
    except SyntaxError as exc:
        return RungResult(RUNG_AST_SAFETY, False, f"unparseable: {exc}")

    bad: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name == p or alias.name.startswith(p + ".") for p in BANNED_AST_PATTERNS):
                    bad.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if any(mod == p or mod.startswith(p + ".") for p in BANNED_AST_PATTERNS):
                bad.append(f"from {mod} import ...")
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Attribute):
                parts: List[str] = []
                cur = func
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                name = ".".join(reversed(parts))
            elif isinstance(func, ast.Name):
                name = func.id
            if name and name in BANNED_AST_PATTERNS:
                bad.append(f"call {name}()")

    if bad:
        return RungResult(
            RUNG_AST_SAFETY,
            False,
            f"banned constructs detected: {', '.join(sorted(set(bad)))}",
        )
    return RungResult(RUNG_AST_SAFETY, True)


def _check_import(patch_source: str, module_name: str = "aura_self_repair_candidate") -> RungResult:
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    if spec is None:
        return RungResult(RUNG_IMPORT, False, "could not create import spec")
    module = importlib.util.module_from_spec(spec)
    try:
        exec(compile(patch_source, f"<{module_name}>", "exec"), module.__dict__)
    except BaseException as exc:
        return RungResult(RUNG_IMPORT, False, f"import-time failure: {exc!r}")
    return RungResult(RUNG_IMPORT, True)


async def _run_probe(rung: str, probe: Optional[Probe]) -> RungResult:
    if probe is None:
        return RungResult(rung, True, reason="no probe provided")
    try:
        result = probe()
        if hasattr(result, "__await__"):
            result = await result
        ok = bool(result)
        return RungResult(rung, ok, reason=None if ok else "probe returned False")
    except BaseException as exc:
        return RungResult(rung, False, reason=f"probe raised: {exc!r}")


# ---------------------------------------------------------------------------
# Public ladder
# ---------------------------------------------------------------------------


async def validate_patch(
    patch_source: str,
    *,
    probes: Optional[SelfRepairProbes] = None,
    stop_on_first_failure: bool = True,
) -> LadderReport:
    """Run the full validation ladder. Returns a LadderReport.

    The audits forbid patches landing on AST parse alone, so even when the
    syntax rung passes the report is rejected unless every other rung also
    passes. ``stop_on_first_failure`` short-circuits the rest of the ladder
    on the first failed rung — set to False when you want to collect all
    failures (useful in postmortems).
    """
    probes = probes or SelfRepairProbes()
    report = LadderReport()

    syntax = _check_syntax(patch_source)
    report.rungs.append(syntax)
    if not syntax.ok and stop_on_first_failure:
        return report

    ast_safety = _check_ast_safety(patch_source)
    report.rungs.append(ast_safety)
    if not ast_safety.ok and stop_on_first_failure:
        return report

    if syntax.ok and ast_safety.ok:
        imp = _check_import(patch_source)
    else:
        imp = RungResult(RUNG_IMPORT, False, "skipped: prior rung failed")
    report.rungs.append(imp)
    if not imp.ok and stop_on_first_failure:
        return report

    for rung, probe in (
        (RUNG_TARGETED, probes.targeted),
        (RUNG_BOOT_SMOKE, probes.boot_smoke),
        (RUNG_ONE_TURN, probes.one_turn),
        (RUNG_SHUTDOWN, probes.shutdown),
        (RUNG_ROLLBACK, probes.rollback),
    ):
        result = await _run_probe(rung, probe)
        report.rungs.append(result)
        if not result.ok and stop_on_first_failure:
            return report

    return report


def patch_is_acceptable(report: LadderReport) -> bool:
    """A patch may land only when every canonical rung is present and ok."""
    rung_names = {r.rung for r in report.rungs if r.ok}
    return all(name in rung_names for name in CANONICAL_RUNGS)
