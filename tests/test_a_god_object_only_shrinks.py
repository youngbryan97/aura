"""Module size, as a ratchet.

`interface/routes/chat.py` is 29,481 lines with 457 module-level functions.
`core/brain/llm/mlx_client.py` is 15,423 with a 165-method class.
`core/brain/inference_gate.py` is 13,416 with a 193-method class handling worker
processes, cloud fallback, health probing, warm-up, desktop resource guards,
background deferral, PII scrubbing, PBKDF2 offloading, RAM diagnostics and UI
prompt strings. Thirty-two files are over three thousand lines.

None of that is fixable in one commit. What is fixable in one commit is the
direction of travel: nothing stopped chat.py reaching forty thousand lines, and
nothing stopped the next God object being created from scratch.
"""
from __future__ import annotations

import json
from pathlib import Path


from tools.lint_module_size import (
    MAX_NEW_CLASS_METHODS,
    MAX_NEW_MODULE_LINES,
    Measurement,
    check,
    load_baseline,
    load_budget,
    measure_tree,
    oversize_total,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "config" / "module_size_baseline.json"


def _live() -> tuple[dict[str, Measurement], dict[str, dict[str, int]]]:
    return measure_tree(), load_baseline(BASELINE)


def test_the_tree_is_within_its_baseline():
    measurements, baseline = _live()
    failures, stale = check(measurements, baseline, budget=load_budget(BASELINE))

    assert failures == [], "\n".join(failures)
    assert stale == [], "\n".join(stale)


def test_the_baseline_records_the_known_offenders():
    """A baseline that omitted them would pass while they grew."""
    baseline = load_baseline(BASELINE)

    for path in (
        "interface/routes/chat.py",
        "core/brain/inference_gate.py",
        "core/brain/llm/mlx_client.py",
    ):
        assert path in baseline, path


def test_uncompensated_growth_fails():
    """A file may grow only if another shrinks by more."""
    measurements, baseline = _live()
    grown = dict(measurements)
    target = grown["core/brain/inference_gate.py"]
    grown[target.path] = Measurement(
        target.path, target.lines + 200, target.max_class_methods, target.largest_class
    )

    failures, _ = check(grown, baseline, budget=load_budget(BASELINE))

    assert any("total oversize" in f for f in failures), failures


def test_the_decomposition_this_gate_exists_to_encourage_passes():
    """Moving four hundred lines out of chat.py into two new modules must PASS.
    The tool's first design pinned every file individually and failed exactly
    this, which is how a gate gets deleted: it blocks the work it was meant to
    cause."""
    measurements, baseline = _live()
    traded = dict(measurements)
    chat = traded["interface/routes/chat.py"]
    traded[chat.path] = Measurement(
        chat.path, chat.lines - 400, chat.max_class_methods, chat.largest_class
    )
    for name in ("chat_streaming", "chat_persistence"):
        traded[f"interface/routes/{name}.py"] = Measurement(
            f"interface/routes/{name}.py", 190, 4, "Small"
        )

    failures, _ = check(traded, baseline, budget=load_budget(BASELINE))

    assert not any("total oversize" in f for f in failures), failures
    assert not any("chat_streaming" in f for f in failures), failures


def test_a_god_class_may_never_grow_even_while_its_file_shrinks():
    """Method count is pinned per class rather than budgeted: splitting a God
    class is the point, growing one is never the trade."""
    measurements, baseline = _live()
    grown = dict(measurements)
    target = grown["core/brain/inference_gate.py"]
    grown[target.path] = Measurement(
        target.path,
        target.lines - 5000,
        target.max_class_methods + 1,
        target.largest_class,
    )

    failures, _ = check(grown, baseline, budget=load_budget(BASELINE))

    assert any("methods from a baseline" in f for f in failures), failures


def test_a_refresh_cannot_raise_the_budget(tmp_path):
    """`--write-baseline` is the escape hatch, and it must not become a way to
    launder growth.

    It used to refuse outright, which meant a real shrink could never be
    banked while any file anywhere had grown — and a gate that cannot be
    satisfied is a gate somebody deletes. It clamps instead: the recorded
    number is the tightest ever seen, and the growth still fails the gate.
    """
    import json

    from tools.lint_module_size import write_baseline

    target = tmp_path / "baseline.json"
    target.write_text(json.dumps({"oversize_budget_lines": 10, "modules": {}}))
    measurements, _ = _live()

    write_baseline(target, measurements)

    assert json.loads(target.read_text())["oversize_budget_lines"] == 10


def test_the_budget_is_the_measured_total():
    measurements, _ = _live()

    assert load_budget(BASELINE) == oversize_total(measurements)


def test_a_new_oversized_module_is_never_grandfathered():
    measurements, baseline = _live()
    measurements = dict(measurements)
    measurements["core/a_brand_new_god.py"] = Measurement(
        path="core/a_brand_new_god.py",
        lines=MAX_NEW_MODULE_LINES + 1,
        max_class_methods=2,
        largest_class="Small",
    )

    failures, _ = check(measurements, baseline)

    assert any("a_brand_new_god" in f for f in failures), failures


def test_a_new_oversized_class_is_never_grandfathered():
    measurements, baseline = _live()
    measurements = dict(measurements)
    measurements["core/a_brand_new_class.py"] = Measurement(
        path="core/a_brand_new_class.py",
        lines=50,
        max_class_methods=MAX_NEW_CLASS_METHODS + 1,
        largest_class="Everything",
    )

    failures, _ = check(measurements, baseline)

    assert any("Everything" in f for f in failures), failures


def test_a_file_that_shrank_must_be_re_recorded():
    """A stale entry is headroom nobody earned, and it is how a ratchet quietly
    stops ratcheting."""
    measurements, baseline = _live()
    measurements = dict(measurements)
    real = measurements["core/brain/inference_gate.py"]
    measurements["core/brain/inference_gate.py"] = Measurement(
        path=real.path,
        lines=real.lines - 500,
        max_class_methods=real.max_class_methods,
        largest_class=real.largest_class,
    )

    _, stale = check(measurements, baseline)

    assert any("inference_gate" in s for s in stale), stale


def test_a_deleted_file_leaves_no_entry_behind():
    measurements, baseline = _live()
    measurements = dict(measurements)
    measurements.pop("core/brain/inference_gate.py")

    _, stale = check(measurements, baseline)

    assert any("no longer exists" in s for s in stale), stale


def test_an_ordinary_file_needs_no_entry():
    measurements, baseline = _live()
    measurements = dict(measurements)
    measurements["core/a_normal_module.py"] = Measurement(
        path="core/a_normal_module.py", lines=200, max_class_methods=6, largest_class="Ok"
    )

    failures, stale = check(measurements, baseline)

    assert not any("a_normal_module" in f for f in failures)
    assert not any("a_normal_module" in s for s in stale)


def test_the_thresholds_come_from_the_distribution_not_from_taste():
    """p98 of file length in this tree is 2,115 lines and p98 of class size is
    26 methods. A ceiling far from those would be an opinion."""
    import ast

    measurements, _ = _live()
    lines = sorted(m.lines for m in measurements.values())
    p98_lines = lines[int(len(lines) * 0.98)]
    assert 0.5 * p98_lines <= MAX_NEW_MODULE_LINES <= 1.5 * p98_lines

    # Measured over CLASSES, which is what the threshold governs and what the
    # tool's docstring cites. A per-file maximum is a different distribution
    # and would justify a different number, so checking that one would be
    # checking a claim nobody made.
    per_class: list[int] = []
    for root in ("core", "interface"):
        for path in (ROOT / root).rglob("*.py"):
            if "__pycache__" in str(path):
                continue
            try:
                tree = ast.parse(path.read_text("utf-8"))
            except (OSError, SyntaxError, UnicodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    per_class.append(
                        sum(
                            1
                            for item in node.body
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        )
                    )
    per_class.sort()
    p98_methods = per_class[int(len(per_class) * 0.98)]

    assert MAX_NEW_CLASS_METHODS >= p98_methods
    assert MAX_NEW_CLASS_METHODS <= 2 * p98_methods


def test_the_baseline_is_a_record_of_offenders_not_of_everything():
    payload = json.loads(BASELINE.read_text("utf-8"))

    assert payload["schema"] == "aura.module_size_baseline.v1"
    assert 0 < len(payload["modules"]) < 200, "a baseline of everything is noise"


def test_the_gate_is_wired_into_the_makefile():
    makefile = (ROOT / "Makefile").read_text("utf-8")

    assert "module-size:" in makefile
    assert "module-size-baseline:" in makefile
    assert "tools/lint_module_size.py" in makefile


# ── the two kinds of regression are told apart ────────────────────────────
#
# The gate has carried an inherited pile for weeks. A flat list of thirty
# complaints is one a reader stops reading, and a module that went over the
# ceiling TODAY sat somewhere in the middle of it looking like everything
# else. Nothing is forgiven — the same failures fail — but the two kinds are
# tagged so the report can tell them apart.


def _measure(path: str, lines: int, methods: int) -> Measurement:
    return Measurement(
        path=path, lines=lines, max_class_methods=methods, largest_class="C"
    )


def test_a_module_never_baselined_is_tagged_new():
    failures, _stale = check(
        {"a.py": _measure("a.py", MAX_NEW_MODULE_LINES + 1, 1)}, {}
    )
    assert failures and all(f.startswith("NEW ") for f in failures)


def test_a_class_never_baselined_is_tagged_new():
    failures, _stale = check(
        {"a.py": _measure("a.py", 10, MAX_NEW_CLASS_METHODS + 1)}, {}
    )
    assert failures and all(f.startswith("NEW ") for f in failures)


def test_a_class_past_its_baseline_is_tagged_grew_with_the_amount():
    failures, _stale = check(
        {"a.py": _measure("a.py", 10, 40)},
        {"a.py": {"lines": 10, "max_class_methods": 31}},
    )
    assert len(failures) == 1
    assert failures[0].startswith("GREW +9 "), failures[0]


def test_the_aggregate_is_tagged_so_it_does_not_read_as_one_more_file():
    failures, _stale = check(
        {"a.py": _measure("a.py", MAX_NEW_MODULE_LINES + 100, 1)},
        {"a.py": {"lines": MAX_NEW_MODULE_LINES + 100, "max_class_methods": 1}},
        budget=0,
    )
    assert len(failures) == 1
    assert failures[0].startswith("BUDGET ")


def test_tagging_forgives_nothing():
    """The same inputs that failed before still fail, and still all of them."""
    measurements = {
        "a.py": _measure("a.py", MAX_NEW_MODULE_LINES + 1, 1),
        "b.py": _measure("b.py", 10, 40),
        "c.py": _measure("c.py", 10, 1),
    }
    baseline = {"b.py": {"lines": 10, "max_class_methods": 31}}
    failures, _stale = check(measurements, baseline, budget=0)
    # One never baselined, one grown, one aggregate; c.py is inside every
    # limit and must not be reported at all.
    assert len(failures) == 3
    assert sum(f.startswith("NEW ") for f in failures) == 1
    assert sum(f.startswith("GREW") for f in failures) == 1
    assert sum(f.startswith("BUDGET") for f in failures) == 1
    assert not any("c.py" in f for f in failures)


def test_a_refresh_never_grandfathers_a_new_god_object(tmp_path):
    """--write-baseline took seven modules out of the zero-tolerance class.

    The clamp protected every entry that was already recorded and said
    nothing about the ones that were not, so a refresh run to bank an
    unrelated shrink granted headroom to every God object created since the
    last one — and "a new God object is never grandfathered" was one
    keystroke from false.
    """
    from tools.lint_module_size import write_baseline

    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "schema": "aura.module_size_baseline.v1",
                "oversize_budget_lines": 10_000,
                "modules": {
                    "known.py": {
                        "lines": MAX_NEW_MODULE_LINES + 500,
                        "max_class_methods": 40,
                    }
                },
            }
        )
    )
    measurements = {
        "known.py": _measure("known.py", MAX_NEW_MODULE_LINES + 400, 39),
        "brand_new.py": _measure("brand_new.py", MAX_NEW_MODULE_LINES + 900, 55),
    }
    write_baseline(path, measurements)

    recorded = json.loads(path.read_text())["modules"]
    assert "brand_new.py" not in recorded, "a new God object was grandfathered"
    # And the shrink on the known one was still banked.
    assert recorded["known.py"]["lines"] == MAX_NEW_MODULE_LINES + 400
    assert recorded["known.py"]["max_class_methods"] == 39

    # The gate still fails on the one that was refused.
    failures, _stale = check(measurements, recorded)
    assert any("brand_new.py" in f for f in failures)


def test_a_first_run_with_no_baseline_still_records_everything(tmp_path):
    """The refusal is about grandfathering, not about bootstrapping."""
    from tools.lint_module_size import write_baseline

    path = tmp_path / "baseline.json"
    measurements = {"a.py": _measure("a.py", MAX_NEW_MODULE_LINES + 100, 40)}
    write_baseline(path, measurements)
    assert "a.py" in json.loads(path.read_text())["modules"]
