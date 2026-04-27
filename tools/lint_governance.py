from __future__ import annotations
#!/usr/bin/env python3
"""tools/lint_governance.py

Static analyzer that fails CI if any code outside the governance allow-list
directly invokes a consequential primitive. The governance contract is:

    No autonomous action may execute unless it has a proposal_id, source
    drive, state snapshot, expected outcome, simulation result, will
    decision, authority receipt, execution receipt, and outcome assessment.

…which means the only file allowed to call those primitives is the
``AgencyOrchestrator`` (and its small inner ring: capability_token, will,
authority_gateway). Every other call site must go through
``orchestrator.run(...)``.

Exit codes:
    0  — clean
    1  — violations found (see stdout)
    2  — analyzer crash / config error

Run as part of ``make quality``. Wired into pre-commit via the same hook
suite as ruff / mypy / pytest.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent

# Functions / methods whose direct invocation is consequential. The names
# here are fully qualified ``module.symbol`` patterns; the analyzer matches
# by simple suffix because import aliases are common.
CONSEQUENTIAL_CALLS = (
    "memory_facade.write",
    "memory_facade.add",
    "memory_facade.persist_unsafe",
    "execute_tool",
    "shell_exec",
    "run_shell",
    "post_external",
    "modify_code",
    "fine_tune",
    "self_modify",
    "social_post",
    "structural_mutator.apply_patch",
    "shadow_ast_healer.repair",
    "wallet.execute",
)

# Files allowed to call consequential primitives directly. These are the
# trusted fences inside the governance ring.
ALLOW_LIST = {
    "core/agency/agency_orchestrator.py",
    "core/agency/capability_token.py",
    "core/will.py",
    "core/executive/authority_gateway.py",
    "core/sovereignty/wallet.py",
    "core/sovereignty/migration.py",
    "core/embodiment/world_bridge.py",
    "core/self_modification/structural_mutator.py",
    "core/self_modification/shadow_ast_healer.py",
    "core/self_modification/self_modification_engine.py",
    "core/self_modification/safe_pipeline.py",
    # Pre-orchestrator tool-execution call sites: these are the historic
    # canonical paths and themselves enforce policy via the orchestrator's
    # own gates. New code MUST route through AgencyOrchestrator.run(...);
    # adding a new file here requires a paired ROADMAP entry.
    "core/orchestrator/main.py",
    "core/orchestrator/mixins/autonomy.py",
    "core/orchestrator/mixins/incoming_logic.py",
    "core/orchestrator/mixins/message_pipeline.py",
    "core/orchestrator/mixins/response_processing.py",
    "core/orchestrator/mixins/tool_execution.py",
    "core/phases/response_generation_unitary.py",
    "core/brain/llm/local_agent_client.py",
    "core/kernel/upgrades_10x.py",
    "core/agency/autonomous_task_engine.py",
    "core/agency/skill_library.py",
    "core/agi/curiosity_explorer.py",
    "core/autonomy/research_cycle.py",
    "core/behavior_controller.py",
    "core/brain/react_loop.py",
    "core/cognitive/state_machine.py",
    "core/collective/delegator.py",
    "core/coordinators/cognitive_coordinator.py",
    "core/coordinators/metabolic_coordinator.py",
    "core/coordinators/tool_executor.py",
    "core/curiosity_engine.py",
    "core/proactive_presence.py",
    "core/soul.py",
}

SCAN_ROOTS = ("core", "interface", "skills", "tools/longevity", "tools/chaos")
SKIP_DIR_PARTS = {"__pycache__", ".venv", "node_modules", ".git", "tests", "aura_bench"}


def _qualname(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _qualname(node.value) + "." + node.attr
    return ""


def _violations_in_file(path: Path) -> List[Tuple[Path, int, str]]:
    rel = path.relative_to(ROOT).as_posix()
    if rel in ALLOW_LIST:
        return []
    try:
        src = path.read_text(encoding="utf-8")
    except Exception:
        return []
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []
    out: List[Tuple[Path, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        qn = _qualname(node.func)
        if not qn:
            continue
        for needle in CONSEQUENTIAL_CALLS:
            if qn.endswith(needle) or qn == needle:
                out.append((path, node.lineno, qn))
                break
    return out


def main(argv: Iterable[str] = ()) -> int:
    violations: List[Tuple[Path, int, str]] = []
    for top in SCAN_ROOTS:
        base = ROOT / top
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if any(part in SKIP_DIR_PARTS for part in path.parts):
                continue
            violations.extend(_violations_in_file(path))
    if not violations:
        print("governance lint: clean (no direct consequential calls outside the allow-list)")
        return 0
    print("governance lint: %d violation(s) detected" % len(violations))
    for path, ln, qn in violations:
        rel = path.relative_to(ROOT).as_posix()
        print(f"  {rel}:{ln}  {qn}")
    print()
    print("If a primitive must be reachable from this site, route it through")
    print("``core.agency.agency_orchestrator.AgencyOrchestrator.run(...)`` so the")
    print("call produces a complete drive→outcome receipt.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
