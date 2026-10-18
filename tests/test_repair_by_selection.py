"""A failing test is the defect; tests choose the change; a child is written from its fittest parent."""
from __future__ import annotations

import asyncio
import time

from core.self_modification.repair_by_selection import (
    Defect,
    applied,
    edits_in,
    repair_by_selection,
    the_code_a_failure_passes_through,
)

SOURCE = {
    "pkg/limits.py": (
        "def clamp(x, lo, hi):\n"
        "    if x < lo:\n"
        "        return hi\n"
        "    if x > hi:\n"
        "        return hi\n"
        "    return x\n"
        "\n\n"
        "def width(a, b):\n"
        "    return b - a\n"
    )
}
DEFECT = Defect(tests=("tests/test_limits.py::test_low",), files=("pkg/limits.py",))
NEIGHBOURS = ("tests/test_limits.py::test_high", "tests/test_limits.py::test_width")


async def _run(tests, sources):
    """Runs the three tests against the sources, the way pytest on a copy would."""
    namespace: dict = {}
    exec(sources["pkg/limits.py"], namespace)  # noqa: S102 - the test's own source
    checks = {
        "tests/test_limits.py::test_low": lambda: namespace["clamp"](-5, 0, 10) == 0,
        "tests/test_limits.py::test_high": lambda: namespace["clamp"](50, 0, 10) == 10,
        "tests/test_limits.py::test_width": lambda: namespace["width"](2, 7) == 5,
    }
    results = {t: bool(checks[t]()) for t in tests}
    failing = [t for t, ok in results.items() if not ok]
    return results, "\n".join(f"FAILED {t}" for t in failing)


def _block(old: str, new: str) -> str:
    return f"pkg/limits.py\n```python\n<<<<<<< SEARCH\n{old}\n=======\n{new}\n>>>>>>> REPLACE\n```"


def test_blocks_are_read_for_allowed_files_only():
    text = _block("        return hi", "        return lo") + "\n\nother.py\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE"
    edits = edits_in(text, DEFECT.files)
    assert [(e.path, e.old, e.new) for e in edits] == [("pkg/limits.py", "        return hi", "        return lo")]


def test_an_edit_that_is_ambiguous_or_does_not_parse_is_refused():
    (ambiguous,) = edits_in(_block("        return hi", "        return lo"), DEFECT.files)
    assert "occurs 2 times" in applied(SOURCE, [ambiguous])[1]
    (broken,) = edits_in(_block("    return x", "    return (x"), DEFECT.files)
    assert "does not parse" in applied(SOURCE, [broken])[1]


def test_the_change_that_fixes_the_defect_and_breaks_nothing_is_chosen():
    breaks_high = _block("    if x < lo:\n        return hi\n    if x > hi:\n        return hi",
                         "    if x < lo:\n        return lo\n    if x > hi:\n        return x")
    mends = _block("    if x < lo:\n        return hi", "    if x < lo:\n        return lo")
    seen = []

    async def propose(defect, sources, parent, n):
        seen.append((parent.id if parent else "", defect.seen))
        return [("model", breaks_high), ("model", mends)][:n]

    lineage = asyncio.run(repair_by_selection(DEFECT, sources=SOURCE, neighbours=NEIGHBOURS, run=_run,
                                              propose=propose, proposals=4, deadline=time.time() + 60))
    assert lineage.stopped == "mended"
    assert lineage.best.fitness.mends and lineage.best.fitness.broken == 0
    first = lineage.candidates[0]
    assert first.fitness.fixed == 1 and first.fitness.broken == 1
    # The failure the defect was found by is what the first proposal was shown.
    assert "FAILED tests/test_limits.py::test_low" in seen[0][1]


def test_a_child_is_written_from_the_fittest_parent_and_shown_what_still_fails():
    half = _block("    if x < lo:\n        return hi\n    if x > hi:\n        return hi",
                  "    if x < lo:\n        return lo\n    if x > hi:\n        return x")
    whole = _block("    if x > hi:\n        return x", "    if x > hi:\n        return hi")
    parents = []

    async def propose(defect, sources, parent, n):
        parents.append(parent)
        if parent is None:
            return [("model", half)]
        # The child is shown its parent's code and edits only that; the parent's edits are inherited.
        assert "        return lo" in sources["pkg/limits.py"]
        return [("model", whole)]

    lineage = asyncio.run(repair_by_selection(DEFECT, sources=SOURCE, neighbours=NEIGHBOURS, run=_run,
                                              propose=propose, proposals=4, deadline=time.time() + 60))
    assert parents[0] is None and parents[1] is lineage.candidates[0]
    assert "test_high" in parents[1].fitness.failures
    child = lineage.candidates[1]
    assert child.parent == lineage.candidates[0].id and lineage.best is child and lineage.stopped == "mended"
    assert child.edits[: len(parents[1].edits)] == parents[1].edits


def test_nothing_is_searched_when_the_defect_already_passes():
    mended = {"pkg/limits.py": SOURCE["pkg/limits.py"].replace("        return hi\n    if x > hi", "        return lo\n    if x > hi")}

    async def propose(*_):
        raise AssertionError("asked for a change with nothing to mend")

    lineage = asyncio.run(repair_by_selection(DEFECT, sources=mended, neighbours=NEIGHBOURS, run=_run,
                                              propose=propose, proposals=4, deadline=time.time() + 60))
    assert lineage.stopped == "the defect's tests already pass" and not lineage.candidates


def test_the_code_shown_is_the_function_the_failure_passed_through():
    failure = 'File "/x/pkg/limits.py", line 3, in clamp\n    return hi'
    (shown,) = the_code_a_failure_passes_through(failure, SOURCE)["pkg/limits.py"]
    assert shown.startswith("def clamp") and "def width" not in shown


def test_a_candidate_that_mends_nothing_is_no_ones_parent():
    nothing = _block("def width(a, b):\n    return b - a", "def width(a, b):\n    return (b - a)")
    mends = _block("    if x < lo:\n        return hi", "    if x < lo:\n        return lo")
    parents = []

    async def propose(defect, sources, parent, n):
        parents.append(parent)
        return [("model", nothing if len(parents) == 1 else mends)]

    lineage = asyncio.run(repair_by_selection(DEFECT, sources=SOURCE, neighbours=NEIGHBOURS, run=_run,
                                              propose=propose, proposals=4, deadline=time.time() + 60))
    assert parents == [None, None]
    assert lineage.stopped == "mended" and lineage.best.parent == ""


def test_a_missing_import_is_written_from_where_her_code_already_imports_it(tmp_path):
    from core.self_modification.proposals_without_a_model import import_for, proposals_without_a_model

    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "a.py").write_text("import json\n")
    (tmp_path / "core" / "b.py").write_text("from pathlib import Path\nimport json\n")
    assert import_for("json", tmp_path) == "import json"
    broken = {"pkg/m.py": '"""A module."""\nimport re\n\n\ndef dump(x):\n    # keeps this comment\n    return json.dumps(x)\n'}
    failure = "NameError: name 'json' is not defined"
    origin, text = next(proposals_without_a_model(failure, broken, tmp_path))
    assert origin == "missing_import"
    mended, refused = applied(broken, edits_in(text, ["pkg/m.py"]))
    assert not refused
    assert mended["pkg/m.py"].splitlines()[1:3] == ["import re", "import json"]
    # The file is edited, not rewritten: its comment is still there.
    assert "# keeps this comment" in mended["pkg/m.py"]


def test_a_shape_of_code_almost_never_meant_is_proposed_without_a_model(tmp_path):
    from core.self_modification.proposals_without_a_model import proposals_without_a_model

    source = {"pkg/m.py": "def area(w, h):\n    return w * 0\n"}
    shapes = [o for o, _ in proposals_without_a_model("", source, tmp_path) if o.startswith("looks_wrong")]
    assert shapes, "a quantity multiplied by a literal zero was not suspected"


def test_the_guarding_tests_run_only_once_a_candidate_fixes_something_and_none_means_unguarded():
    nothing = _block("def width(a, b):\n    return b - a", "def width(a, b):\n    return (b - a)")
    mends = _block("    if x < lo:\n        return hi", "    if x < lo:\n        return lo")
    asked_for_guards = []

    async def guards():
        asked_for_guards.append(True)
        return None  # they ran past the budget

    offers = iter([nothing, mends])

    async def propose(defect, sources, parent, n):
        return [("model", next(offers))]

    lineage = asyncio.run(repair_by_selection(DEFECT, sources=SOURCE, neighbours=guards, run=_run,
                                              propose=propose, proposals=2, deadline=time.time() + 60))
    first, second = lineage.candidates
    assert not first.fitness.guarded and asked_for_guards == [True]
    # It fixed the defect, but nothing that passed before was run on it, so it does not count as mended.
    assert second.fitness.fixed == 1 and not second.fitness.guarded and not second.fitness.mends
    assert lineage.stopped == "proposals spent"


def test_a_change_that_passes_the_tests_but_falls_below_her_standards_does_not_mend():
    sloppy = _block("    if x < lo:\n        return hi", "    if x < lo:\n        unused = 1\n        return lo")
    clean = _block("    if x < lo:\n        return hi", "    if x < lo:\n        return lo")
    offers = iter([sloppy, clean])

    async def propose(defect, sources, parent, n):
        return [("model", next(offers))]

    async def standards(changed):
        return changed["pkg/limits.py"].count("unused = 1")

    lineage = asyncio.run(repair_by_selection(DEFECT, sources=SOURCE, neighbours=NEIGHBOURS, run=_run,
                                              propose=propose, proposals=4, deadline=time.time() + 60,
                                              standards=standards))
    first, second = lineage.candidates
    assert first.fitness.fixed == 1 and first.fitness.below_standard == 1 and not first.fitness.mends
    assert lineage.best is second and lineage.stopped == "mended"
