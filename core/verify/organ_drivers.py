"""Which named organs are actually driven, proven from the import graph.

An external reader went through the consciousness layers by hand and found six
modules that were instantiated, registered, exposed in a snapshot, and never
driven. `MinimalSelfhood.update()` had no caller. `AutopoiesisEngine.start()`
had no caller and its own getter's docstring said so. `EndogenousFitness` was
fetched into an attribute that nothing read.

Hand auditing found them once. This finds them every run.

An organ here is a class with a method that advances it — a tick, an update, a
start. The organ is DRIVEN when some production module calls that method, and
production excludes tests and the module that defines it, because a class
calling its own method proves nothing and a test proving a thing works does not
make it run. Anything undriven is named with the reason, so a claim about it
can be checked rather than believed.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Organ", "ORGANS", "KNOWN_UNDRIVEN", "undriven_organs", "callers_of",
           "methods_that_do_not_exist", "still_undriven"]

#: Where production code lives. An archived script and a snapshot of a training
#: run under artifacts/ are neither production nor evidence about it.
_PRODUCTION_ROOTS = ("core", "interface", "skills", "llm", "executors", "security")


@dataclass(frozen=True, slots=True)
class Organ:
    """A named subsystem and the call that advances it."""

    name: str
    module: str
    method: str
    note: str = ""
    #: How a caller reaches it: the dotted module, and the service name it is
    #: registered under. A file that calls the method without naming either is
    #: calling something else that happens to share the name.
    reached_by: tuple[str, ...] = ()


#: The organs whose drivers this gate proves. Each was found undriven at least
#: once; the entry is what stops it happening again quietly.
ORGANS: tuple[Organ, ...] = (
    Organ(
        "minimal_selfhood",
        "core/consciousness/minimal_selfhood.py",
        "update",
        "chemotaxis layer; without update() current_state() is None forever",
        ("core.consciousness.minimal_selfhood", "minimal_selfhood"),
    ),
    Organ(
        "recursive_self_knowing",
        "core/consciousness/recursive_self_knowing.py",
        "observe_claim",
        "second-order observer; cognitive_engine reads second_order_strength off it",
        ("core.consciousness.recursive_self_knowing", "recursive_self_knowing"),
    ),
    Organ(
        "automatic_self_knowing",
        "core/consciousness/automatic_self_knowing.py",
        "tick",
        "continuous self-observation, as opposed to one hardcoded chat event",
        ("core.consciousness.automatic_self_knowing", "automatic_self_knowing"),
    ),
    Organ(
        "autopoiesis",
        "core/cognitive/autopoiesis.py",
        "tick",
        "without a tick, get_vitality() returns the constructor's number forever",
        ("core.cognitive.autopoiesis", "autopoiesis"),
    ),
    Organ(
        "endogenous_fitness",
        "core/consciousness/endogenous_fitness.py",
        "current_crisis",
        "survival thresholds against live readings",
        ("core.consciousness.endogenous_fitness", "endogenous_fitness"),
    ),
    Organ(
        "selfhood_tick",
        "core/consciousness/selfhood_tick.py",
        "drive_selfhood",
        "the driver above is itself only real if a registered phase calls it",
        ("core.consciousness.selfhood_tick",),
    ),
)


#: Named things that nothing in production drives, and are not claimed to.
#:
#: Recorded rather than removed, because the cost of an impressively named
#: module is that somebody counts it. Held here, the fact survives the person
#: who found it. If one of these gets a driver, move it to ORGANS above — the
#: test below fails until the record matches the code either way.
KNOWN_UNDRIVEN: tuple[Organ, ...] = (
    Organ(
        "personhood_engine",
        "core/autonomy/personhood_engine.py",
        "start",
        "a spontaneous-speech daemon nothing starts; its own docstring says the "
        "name asserts a conclusion the module does not reach",
        ("core.autonomy.personhood_engine",),
    ),
    Organ(
        "life_tick",
        "core/organism/life_tick.py",
        "execute_tick",
        "for a boxed organism simulation; production cognition is MindTick and "
        "the kernel, which its own header says",
        ("core.organism.life_tick",),
    ),
)


def _production_files(root: Path) -> list[Path]:
    """Every source file whose calls count as production."""
    found: list[Path] = []
    for top in _PRODUCTION_ROOTS:
        base = root / top
        if not base.is_dir():
            continue
        # The parts are read RELATIVE to the root: this checkout lives under
        # ~/.aura, so testing the absolute path for a leading dot hid every
        # file in the tree and the gate reported everything undriven.
        found.extend(
            path
            for path in sorted(base.rglob("*.py"))
            if not any(
                part.startswith(".") or part == "__pycache__"
                for part in path.relative_to(root).parts
            )
        )
    return found


def callers_of(organ: Organ, root: Path | None = None) -> tuple[str, ...]:
    """Every production file that calls this organ's driving method."""
    base = root or Path(__file__).resolve().parents[2]
    defining = (base / organ.module).resolve()
    found: list[str] = []
    for path in _production_files(base):
        if path.resolve() == defining:
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Both halves are required: this file reaches the organ, and it calls
        # the method. Either alone is a coincidence of naming.
        if organ.reached_by and not any(handle in source for handle in organ.reached_by):
            continue
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # `engine.tick()` and `drive_selfhood(state)` are both drivers; the
            # second form is what a module-level function looks like.
            called = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            if called == organ.method:
                found.append(f"{path.relative_to(base)}:{node.lineno}")
                break
    return tuple(found)


def methods_that_do_not_exist(root: Path | None = None) -> tuple[str, ...]:
    """Entries naming a method their own module does not define.

    A record pointing at a method nobody wrote reports "undriven" forever no
    matter what the code does, which is the failure this whole module exists to
    catch, one level up. `life_tick` was recorded against `process_tick` and the
    method is `execute_tick`.
    """
    base = root or Path(__file__).resolve().parents[2]
    missing: list[str] = []
    for organ in (*ORGANS, *KNOWN_UNDRIVEN):
        path = base / organ.module
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError, ValueError):
            missing.append(f"{organ.name}: {organ.module} could not be read")
            continue
        defined = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if organ.method not in defined:
            missing.append(f"{organ.name}: {organ.module} defines no {organ.method}()")
    return tuple(missing)


def still_undriven(root: Path | None = None) -> tuple[str, ...]:
    """Which recorded-undriven things have quietly gained a driver."""
    return tuple(
        f"{organ.name} is now driven from {', '.join(callers_of(organ, root))} — "
        f"move it to ORGANS"
        for organ in KNOWN_UNDRIVEN
        if callers_of(organ, root)
    )


def undriven_organs(root: Path | None = None) -> tuple[str, ...]:
    """Every organ with no production caller, named with why that matters."""
    return tuple(
        f"{organ.name}: nothing calls {organ.method}() outside {organ.module} — {organ.note}"
        for organ in ORGANS
        if not callers_of(organ, root)
    )
