"""Assert what the source DOES without asserting where it sits.

A test that reads source for a call site is the cheapest way to hold a
property no runtime path can reach — an ordering between two statements, a
call that must exist, a spelling that must not. It is also the easiest
thing in this tree to break without changing behaviour, and the full
chunked suite of 2026-09-18 found five of them red in four chunks:

- ``inspect.getsource(record_degradation)`` searched for text the
  method-size sweep had moved into an extracted helper
- a test named ``_api_chat_turn`` and the call moved to
  ``_api_chat_turn_part_10``
- a 700-character window missed a derivation that was still seven
  statements away, because comments were written between
- ``source.index(needle, at)`` raised ValueError, not an assertion, when
  the delimiter moved ahead of the marker it was meant to follow

Every one failed on a refactor that changed nothing, and every one would
have passed on a change that kept the words and broke the meaning. The
fix in each case was the same shape, so it is written once here.

Two rules. Find the code by what it does, not by the name of the function
it currently lives in. Assert an ORDER or a PRESENCE, never a distance —
a proximity window is a measurement of formatting.
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
from types import ModuleType

__all__ = [
    "class_with_its_bases",
    "module_and_the_mixins_it_builds_with",
    "family_text",
    "module_family_sources",
    "declared_in",
    "function_containing",
    "function_with_its_helpers",
    "in_order",
    "module_source",
    "reached_from_a_finally",
]


def module_source(module: ModuleType) -> str:
    """The whole module's text."""

    return inspect.getsource(module)


def module_family_sources(module: ModuleType) -> list[tuple[str, str]]:
    """``(name, text)`` for the module and every module lifted out of it.

    A lift names its new module after the one it came from:
    ``inference_gate_turn_setup`` beside ``inference_gate``,
    ``mlx_client_waiting`` beside ``mlx_client``, ``mind_tick_loop_steps``
    beside ``mind_tick``. What the original module DOES is now spread over
    that family, and a reader that stops at the first file reports a line
    as gone that runs on every turn. Read from disk, so an unimported
    sibling counts too.
    """

    from pathlib import Path

    own = Path(getattr(module, "__file__", "") or "")
    family = [(module.__name__, inspect.getsource(module))]
    if own.suffix != ".py":
        return family
    for sibling in sorted(own.parent.glob(f"{own.stem}_*.py")):
        try:
            family.append((f"{module.__name__}:{sibling.stem}", sibling.read_text(encoding="utf-8")))
        except OSError:  # pragma: no cover - a file that vanished mid-read
            continue
    # A sibling read from disk and the same module reached through a mixin
    # base are one text; naming both put every call site in it twice.
    package = module.__name__.rsplit(".", 1)[0]
    seen = {module.__name__} | {
        f"{package}.{sibling.stem}" for sibling in own.parent.glob(f"{own.stem}_*.py")
    }
    for base_home in _mixin_homes(module):
        if base_home.__name__ in seen:
            continue
        seen.add(base_home.__name__)
        try:
            family.append((base_home.__name__, inspect.getsource(base_home)))
        except (OSError, TypeError):  # pragma: no cover - C or dynamic base
            continue
    return family


def _mixin_homes(module: ModuleType) -> list[ModuleType]:
    """The modules the classes of ``module`` inherit their behaviour from.

    The split a class-level lift produces does not always take the module's
    name: ``MLXLocalClient`` builds on ``mlx_unified_recurrent`` and
    ``mlx_latent_reasoning``, beside ``mlx_client``. The class carries the
    pointer; follow it.
    """

    homes: list[ModuleType] = []
    for _name, obj in vars(module).items():
        if not inspect.isclass(obj) or obj.__module__ != module.__name__:
            continue
        for base in obj.__mro__[1:]:
            if base is object:
                continue
            home = sys.modules.get(base.__module__)
            if home is None or home is module or home in homes:
                continue
            path = str(getattr(home, "__file__", "") or "")
            if "site-packages" in path or not path:
                continue
            homes.append(home)
    return homes


def family_text_at(path: str | os.PathLike[str]) -> str:
    """``family_text`` for a module named by its file, unimported.

    For a test that reads a module as text from the start and pins a line
    that a lift has since moved into a sibling.
    """

    from pathlib import Path

    own = Path(path)
    parts = [own.read_text(encoding="utf-8")]
    for sibling in sorted(own.parent.glob(f"{own.stem}_*.py")):
        try:
            parts.append(sibling.read_text(encoding="utf-8"))
        except OSError:  # pragma: no cover - a file that vanished mid-read
            continue
    return "\n".join(parts)


def family_tree(path: str | os.PathLike[str]) -> ast.Module:
    """The module and every module lifted out of it, parsed as one tree.

    For a test that walks the AST for a function by name and finds the
    function now defined in a sibling. The families' bodies are joined;
    nothing is imported.
    """

    from pathlib import Path

    own = Path(path)
    bodies: list[ast.stmt] = []
    for member in (own, *sorted(own.parent.glob(f"{own.stem}_*.py"))):
        try:
            bodies.extend(ast.parse(member.read_text(encoding="utf-8")).body)
        except OSError:  # pragma: no cover - a file that vanished mid-read
            continue
    return ast.Module(body=bodies, type_ignores=[])


def family_text(module: ModuleType) -> str:
    """The module and every module lifted out of it, as one text.

    For a test that pins a line's PRESENCE somewhere in what a module does,
    rather than an order inside one function.
    """

    return "\n".join(text for _name, text in module_family_sources(module))


def function_containing(module: ModuleType, needle: str) -> tuple[str, str]:
    """Return ``(name, body)`` of whichever function contains ``needle``.

    Survives extraction: when a sweep moves a block into a helper, this
    follows it, and when a lift moves the helper into a sibling module
    (``<module>_<part>.py``) it follows it there. Raises with the needle in
    the message when nothing in the family contains it, which is the case
    worth failing on — the call site really is gone.
    """

    for _name, source in module_family_sources(module):
        lines = source.splitlines()
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = "\n".join(lines[node.lineno - 1 : (node.end_lineno or node.lineno)])
            if needle in body:
                return node.name, body
    raise AssertionError(
        f"no function in {module.__name__} or the modules lifted out of it contains "
        f"{needle!r}; the call site is gone, not merely moved"
    )


def in_order(source: str, *needles: str) -> None:
    """Assert the needles appear in this order, at any distance.

    Each is searched from the end of the previous one, so repeats of an
    earlier needle cannot satisfy a later one.
    """

    if len(needles) < 2:
        raise ValueError("an ordering needs at least two things to order")
    cursor = 0
    seen: list[tuple[str, int]] = []
    for needle in needles:
        found = source.find(needle, cursor)
        if found == -1:
            after = f" after {seen[-1][0]!r}" if seen else ""
            raise AssertionError(f"{needle!r} does not appear{after}")
        seen.append((needle, found))
        cursor = found + len(needle)


def declared_in(needle: str, *modules: ModuleType) -> tuple[str, str]:
    """Return ``(module_name, body)`` for whichever module declares ``needle``.

    ``function_containing`` follows a block that moved into a helper. This
    follows one that moved into another FILE, which is the same refactor
    one directory up and breaks a test the same way.

    LIVE 2026-09-18: ``purpose="research_document_synthesis"`` was read out
    of ``core/skills/desktop_task.py`` by three tests. A method-size sweep
    moved the synthesis call to ``core/skills/desktop_research.py`` with
    every property it was being checked for intact — the origin, both
    internal flags — and ``source.index`` raised ValueError. The tests were
    holding a location, and the location was never the point.

    Search order is the order given, so the caller says which module is the
    expected home and which are the places it may have gone.
    """

    if not modules:
        raise ValueError("a declaration has to be looked for somewhere")
    for module in modules:
        source = inspect.getsource(module)
        if needle in source:
            return module.__name__, source
    looked = ", ".join(module.__name__ for module in modules)
    raise AssertionError(
        f"{needle!r} is declared in none of {looked}; "
        "it is gone, not merely moved"
    )


def reached_from_a_finally(module: ModuleType, needle: str) -> bool:
    """Whether ``needle`` runs on every exit, directly or one call away.

    "Closed in a finally" is a property of the exit path, not of where the
    line sits. The method-size sweep lifts a whole finally-body into a
    helper and leaves the CALL in the finally, so a test that walks for
    the statement inside a ``finalbody`` stops finding it while the
    guarantee is untouched — measured on ``_close_provenance_tick``, which
    moved into ``_run_thinking_loop_closed_rather_after`` and is still
    invoked from the finally at the end of the thinking loop.

    So: true when the needle is in a finally body, or when it sits in a
    function that a finally body calls. One level, because one level is
    what an extraction produces; a deeper chain should be asserted on
    purpose rather than discovered here.
    """

    source = inspect.getsource(module)
    tree = ast.parse(source)
    lines = source.splitlines()

    def _spans(node: ast.AST) -> tuple[int, int]:
        return node.lineno, getattr(node, "end_lineno", node.lineno) or node.lineno

    finallys: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and node.finalbody:
            low = node.finalbody[0].lineno
            high = max(_spans(s)[1] for s in node.finalbody)
            finallys.append((low, high))

    def _in_a_finally(line: int) -> bool:
        return any(low <= line <= high for low, high in finallys)

    # Directly.
    for index, text in enumerate(lines, start=1):
        if needle in text and _in_a_finally(index):
            return True

    # Or inside a function a finally calls.
    holders = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        low, high = _spans(node)
        if any(needle in line for line in lines[low - 1 : high]):
            holders.add(node.name)
    if not holders:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name in holders and _in_a_finally(node.lineno):
            return True
    return False


def function_with_its_helpers(
    module: ModuleType, qualified_name: str, *, depth: int = 2
) -> str:
    """One function's source plus the source of what it calls in its module.

    The unit a source-reading test actually means. ``_run_loop`` is not
    the loop any more — the method-size sweep lifted runs of statements
    out of it into helpers beside it, and a test reading only the method
    sees a shell that calls names. Nine tests in
    ``test_mind_tick_runtime_contract`` went red that way at once, most
    with ``ValueError: substring not found`` from a bare ``source.index``.

    ``qualified_name`` is "Class.method" or "function". Depth 2 follows a
    helper's own helpers, which is where a second sweep pass puts things.
    """

    whole = inspect.getsource(module)
    tree = ast.parse(whole)
    lines = whole.splitlines()

    bodies: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bodies.setdefault(
                node.name,
                "\n".join(lines[node.lineno - 1 : (node.end_lineno or node.lineno)]),
            )

    wanted = qualified_name.rsplit(".", 1)[-1]
    # A helper the sweep then lifted into a mixin lives in another module.
    # The class knows where: its MRO reaches every base, and inspect reads
    # the method from the module that defines it. LIVE 2026-09-19:
    # ``MindTick._run_loop``'s steps moved to ``_RunsTheTickLoopSteps`` in
    # ``mind_tick_loop_steps`` and nine tests here went red a second time.
    owner: type | None = None
    if "." in qualified_name:
        owner = getattr(module, qualified_name.rsplit(".", 1)[0], None)
        if not isinstance(owner, type):
            owner = None

    def _inherited(name: str) -> str | None:
        if owner is None:
            return None
        member = getattr(owner, name, None)
        if member is None:
            return None
        try:
            return inspect.getsource(member)
        except (OSError, TypeError):
            return None

    if wanted not in bodies and _inherited(wanted) is None:
        raise AssertionError(
            f"{module.__name__} has no function named {wanted!r}; "
            "it was renamed or removed, not merely moved"
        )

    collected: list[str] = []
    seen: set[str] = set()

    def _pull(name: str, remaining: int) -> None:
        if name in seen:
            return
        body = bodies.get(name) or _inherited(name)
        if body is None:
            return
        seen.add(name)
        collected.append(body)
        if remaining <= 0:
            return
        for called in ast.walk(ast.parse(_dedent(body))):
            if isinstance(called, ast.Call):
                target = called.func
                called_name = getattr(target, "attr", None) or getattr(
                    target, "id", None
                )
                if called_name:
                    _pull(called_name, remaining - 1)

    _pull(wanted, depth)
    return "\n".join(collected)


def _dedent(body: str) -> str:
    """A method's body parses on its own once its indent is removed."""

    import textwrap

    return textwrap.dedent(body)
def class_with_its_bases(cls: type) -> str:
    """``cls``'s own source plus every base it inherits behaviour from.

    The third shape the method-size work produces: a block is lifted out of
    a class into a mixin the class then inherits. What the class DOES is
    unchanged, and ``inspect.getsource(TheClass)`` stops containing it.

    LIVE 2026-09-18: the rule that sends exactly the number of sources the
    person asked for — ``num_results = requested``, with the key omitted
    when they asked for none — moved from ``DesktopTaskSkill`` into
    ``_ResearchesBeforeItWrites``. Two tests read the class and found
    neither, while the skill still behaved exactly as they describe.

    ``object`` is skipped, and so is anything whose source cannot be read
    (a C extension, a dynamically built class), because a base that cannot
    be read is not evidence either way.
    """

    seen: list[str] = []
    for base in cls.__mro__:
        if base is object:
            continue
        try:
            seen.append(inspect.getsource(base))
        except (OSError, TypeError) as exc:  # pragma: no cover - C bases
            del exc
    if not seen:
        raise AssertionError(f"no source could be read for {cls!r} or its bases")
    return "\n".join(seen)


def module_and_the_mixins_it_builds_with(module: ModuleType, cls: type) -> str:
    """A module's own text, plus the modules its class's bases come from.

    The fourth place a source read loses its subject, and the one a split
    produces rather than an extraction: a cluster of methods moves out of a
    class into a mixin in a NEW module, and the class inherits it. Reading
    the original module finds neither the methods nor the lines inside them,
    while the class still has every one.

    `class_with_its_bases` answers the same question for the class alone.
    This one keeps the module's own top level too, for a test that reads
    both — a spawn site in a method and a constant beside it.
    """

    seen: list[str] = [inspect.getsource(module)]
    for base in cls.__mro__:
        if base is object:
            continue
        home = sys.modules.get(base.__module__)
        if home is None or home is module:
            continue
        try:
            text = inspect.getsource(home)
        except (OSError, TypeError):  # pragma: no cover - C or dynamic base
            continue
        if text not in seen:
            seen.append(text)
    return "\n".join(seen)
