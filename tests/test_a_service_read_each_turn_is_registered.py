"""Every service a turn reads by name is registered somewhere that runs.

On 24 September her capacity sat at 0.5 and her usefulness at 0.0 for every
run: three readings asked for a runtime service named "agency_ledger", which
nothing registers, and got None without a word (64c4fc443). This scans the
modules a turn runs through for every name read from the container or the
runtime-service registry, and every name registered anywhere in the tree, and
fails on a read with nothing registered behind it.

A registration inside a module-level function that nothing refers to does not
count. Three conversational engines registered themselves from accessors no
code called, so their readers got None on every turn while looking wired.

The scan is static. A name built into a string at run time is invisible to it,
and a registration in a boot step that does not run on some path (a subject
run, a foreground-only boot) still counts as a registration.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
TREE = ("core", "interface", "skills", "llm", "executors", "security", "aura_main.py")

#: The modules a turn runs through: the phases, the kernel, and the per-turn
#: helpers they call that read services by name.
PER_TURN = (
    "core/phases/",
    "core/kernel/",
    "core/self/canonical_self.py",
    "core/consciousness/executive_closure.py",
    "core/consciousness/unified_audit.py",
    "core/consciousness/whole_system_phi_service.py",
    "core/agency/self_play.py",
    "core/agency/agency_shard_work.py",
    "core/runtime/conversation_support.py",
)

#: Names a turn reads that are registered nowhere, left that way on purpose
#: until Bryan decides, each with why. Registering one is a change to how she
#: behaves, not a wiring fix. The test fails when an entry becomes registered,
#: so the list cannot outlive its reason.
_PROMPT_TEXT = (
    "its only per-turn output is guidance prose for her prompt; switching it on "
    "is a decision about prompt text"
)
_UNBUILT_ORGAN = (
    "one of the organs the cognitive integration phase skips every tick because "
    "nothing builds them; switching them on changes her dynamics"
)
DORMANT: dict[str, str] = {
    "conversation_intelligence": _PROMPT_TEXT,
    "humor_engine": _PROMPT_TEXT,
    "relational_intelligence": _PROMPT_TEXT,
    "sentiment_tracker": _UNBUILT_ORGAN,
    "autonomous_resilience_mesh": _UNBUILT_ORGAN,
    "adaptive_immune_system": _UNBUILT_ORGAN,
    "strange_loop": _UNBUILT_ORGAN,
    "homeostatic_rl": _UNBUILT_ORGAN,
    "topology_evolution": _UNBUILT_ORGAN,
    "autopoiesis": _UNBUILT_ORGAN,
    "alife_dynamics": _UNBUILT_ORGAN,
    "alife_extensions": _UNBUILT_ORGAN,
    "endogenous_fitness": _UNBUILT_ORGAN,
    "substrate_governor": (
        "the governor is built only by core/startup/aura_3_boot.py, which no boot "
        "path imports, and there under the name 'governor'"
    ),
    "initiative_queue": "no module builds a queue; the kernel's morphic insights have nowhere to go",
    "conversational_profiler": _PROMPT_TEXT,
    "user_model": "the reader calls update_from_interaction, which no class in the tree defines",
    "theory_of_mind_user_model": "the reader calls update_from_interaction, which no class in the tree defines",
    "continuous_learning": (
        "boot builds this engine as orchestrator.learning_engine and its post_think hook "
        "already records each exchange; registering it here would record every one twice"
    ),
    "continuous_learning_engine": (
        "boot builds this engine as orchestrator.learning_engine and its post_think hook "
        "already records each exchange; registering it here would record every one twice"
    ),
    "structural_opacity_monitor": (
        "no module builds the monitor, so the unified audit's opacity reading is 0.0 on "
        "every run; building it is a change to what the audit claims"
    ),
}

_READERS = {"get_runtime_service", "optional_service", "has_runtime_service", "require_service"}
_CONTAINER_READERS = {"get", "peek", "has", "require"}
_CONTAINERS = {"ServiceContainer", "container", "cls"}
_NAME_KEYWORDS = {"name", "service_name", "alias", "target"}


def _callee(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _on_container(call: ast.Call) -> bool:
    func = call.func
    return isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in _CONTAINERS


def _literal(node: ast.AST, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "ServiceNames":
        return constants.get(node.attr)
    return None


def _is_registration(call: ast.Call) -> bool:
    name = _callee(call) or ""
    if name == "set":
        return _on_container(call)
    lowered = name.lower()
    return "register" in lowered and "unregister" not in lowered and "deregister" not in lowered


def _registered_names(call: ast.Call, constants: dict[str, str]) -> set[str]:
    # The name is the first argument, or the second for a helper that takes a
    # report or a container first: `_register(report, "will_engine", engine)`.
    found = {_literal(arg, constants) for arg in call.args[:2]}
    found |= {_literal(k.value, constants) for k in call.keywords if k.arg in _NAME_KEYWORDS}
    return {name for name in found if name}


def _read_names(call: ast.Call, constants: dict[str, str]) -> set[str] | None:
    name = _callee(call)
    if name in _READERS or (name in _CONTAINER_READERS and _on_container(call)):
        return {n for n in (_literal(arg, constants) for arg in call.args) if n}
    return None


def scan(sources: dict[str, str], per_turn: tuple[str, ...], constants: dict[str, str]) -> set[str]:
    """Names read on the per-turn path with no registration anywhere that runs."""
    registered: set[str] = set()
    referenced: set[str] = set()
    guarded: list[tuple[str, set[str]]] = []
    reads: list[set[str]] = []
    for path, text in sources.items():
        tree = ast.parse(text)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced.add(node.attr)
            elif isinstance(node, ast.alias):
                referenced.add(node.asname or node.name.rsplit(".", 1)[-1])
            if not isinstance(node, ast.Call):
                continue
            if _is_registration(node):
                names = _registered_names(node, constants)
                enclosing = parents.get(node)
                while enclosing is not None and not isinstance(enclosing, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    enclosing = parents.get(enclosing)
                # A module-level function counts only if something refers to
                # it. A method is reached through its object, which a static
                # scan cannot follow, so it counts.
                if isinstance(enclosing, (ast.FunctionDef, ast.AsyncFunctionDef)) and isinstance(parents.get(enclosing), ast.Module):
                    guarded.append((enclosing.name, names))
                else:
                    registered |= names
                continue
            if not any(path.startswith(scope) for scope in per_turn):
                continue
            names = _read_names(node, constants)
            if not names:
                continue
            # `get("a") or get("b")` is one read satisfied by either name.
            parent = parents.get(node)
            if isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.Or):
                if parent.values[0] is not node:
                    continue
                for value in parent.values[1:]:
                    if isinstance(value, ast.Call):
                        names |= _read_names(value, constants) or set()
            reads.append(names)
    for function, names in guarded:
        if function in referenced:
            registered |= names
    return {name for names in reads if not names & registered for name in names}


def _tree_sources() -> dict[str, str]:
    sources: dict[str, str] = {}
    for root in TREE:
        base = REPO / root
        for path in [base] if base.is_file() else sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            sources[str(path.relative_to(REPO))] = path.read_text(encoding="utf-8", errors="ignore")
    return sources


def _constants() -> dict[str, str]:
    from core.service_names import ServiceNames

    return {key: value for key, value in vars(ServiceNames).items() if isinstance(value, str) and not key.startswith("_")}


@pytest.fixture(scope="module")
def unregistered() -> set[str]:
    return scan(_tree_sources(), PER_TURN, _constants())


def test_every_service_a_turn_reads_is_registered_or_named_dormant(unregistered: set[str]) -> None:
    unexplained = sorted(unregistered - set(DORMANT))
    assert not unexplained, (
        f"a turn reads these services by name and nothing registers them: {unexplained}. "
        "Read the owner's accessor, register it at boot, or name it in DORMANT with why."
    )


def test_every_dormant_name_is_still_unregistered(unregistered: set[str]) -> None:
    stale = sorted(set(DORMANT) - unregistered)
    assert not stale, f"now registered or no longer read each turn; take them out of DORMANT: {stale}"
    assert all(reason.strip() for reason in DORMANT.values())


def test_the_scan_catches_the_shape_that_hid_the_agency_ledger() -> None:
    sources = {
        "core/phases/readings.py": (
            "from core.runtime.service_registry import get_runtime_service\n"
            "def capacity():\n"
            "    return get_runtime_service('agency_ledger', default=None)\n"
            "def mood():\n"
            "    return ServiceContainer.get('affect', default=None)\n"
            "def either():\n"
            "    return ServiceContainer.get('old_name', default=None) or ServiceContainer.get('affect', default=None)\n"
        ),
        "core/affect/boot.py": "def boot(report):\n    _register(report, 'affect', object())\nboot(None)\n",
        "core/humor.py": "def get_humor():\n    ServiceContainer.register_instance('humor', object())\n",
        "core/phases/talk.py": "def talk():\n    return ServiceContainer.get('humor', default=None)\n",
    }
    assert scan(sources, ("core/phases/",), {}) == {"agency_ledger", "humor"}
