"""A package reaching into another for a name it never reads.

A cross-package edge is the unit of verification surface: every one is a pair
of packages whose invariants can now interact. An edge that exists because of
a single import whose name is never used is the cheapest kind of coupling
there is to remove — nothing uses it and nothing notices.

Five were found and removed. This holds that no more appear.
"""

from __future__ import annotations

from tools.lint_dead_cross_package_imports import dead


def test_no_edge_rests_on_an_import_nobody_reads():
    found = dead()
    assert not found, "\n".join(
        f"{one['edge']} — {one['file']}:{one['line']} imports "
        f"{one['imports']} for {', '.join(one['names'])}, and reads none of them"
        for one in found
    )


def test_an_availability_probe_is_not_flagged():
    """An import inside a try that guards an optional subsystem is doing the
    work: it separates "not registered", which the container answers with
    None, from "not installed", which raises. The name is unused on purpose,
    and `# noqa: F401` is how this tree already says so."""

    from pathlib import Path

    said = Path("core/coherence/binding_engine.py").read_text(encoding="utf-8")
    assert "from core.consciousness.phenomenal_now import PhenomenalNowEngine  # noqa: F401" in said
    assert "The import IS the check" in said
    assert not [one for one in dead() if "coherence" in str(one["edge"])]
