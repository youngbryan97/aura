"""Which capability a request reaches must not depend on the process it ran in.

`distinctive_objects` keeps the nouns that separate one skill from the rest.
When the cutoff separates nothing — a small catalogue, or a skill whose
whole vocabulary is common — it fell back to `max(objects, key=weight)`.
Every weight in that case is equal, so `max` over a set returned whichever
string came first in set iteration order, and string hashing is randomised
per process.

Measured before the fix, over 24 fresh interpreters on the same two-skill
catalogue: "python" was chosen 15 times and "sandboxed" 9 times, so "run
some Python" reached the REPL or reached nothing depending on the run. The
same coin flip was there before the engineering work and is why
test_a_weak_semantic_neighbor_does_not_join_the_best_supported_skill failed
intermittently.

Nothing in the live 82-skill catalogue hits that fallback, so this changes
no live selection. It removes a way for selection to change across a
restart, which is the kind of thing that is nearly impossible to debug from
a report.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.intent.declared_capability import (
    declared_vocabulary,
    distinctive_objects,
    rank_declaration_matches,
)

REPO = Path(__file__).resolve().parents[1]

TWO_SKILLS = {
    "code_repl": ("code_repl", "Execute Python code in a sandboxed REPL"),
    "program_dna_equivalence_battery": (
        "program_dna_equivalence_battery",
        "Run a program equivalence battery across simulated applications",
    ),
}


def _catalogue():
    return {name: declared_vocabulary(*args) for name, args in TWO_SKILLS.items()}


def test_a_degenerate_catalogue_keeps_every_equally_distinctive_noun():
    """Discarding all but one threw away the noun the request actually said."""
    selective = distinctive_objects(_catalogue())
    assert selective["code_repl"] == frozenset({"python", "sandboxed"})
    # "code" is absent because it is one of the skill's own name words, which
    # are matched by name elsewhere. That exclusion is deliberate and is what
    # keeps the fallback from picking "lab" out of `quantum_lab`.
    assert "code" not in selective["code_repl"]


def test_the_request_reaches_the_skill_that_declared_it():
    catalogue = _catalogue()
    ranked = rank_declaration_matches(
        "Run some Python and tell me the result.",
        catalogue,
        distinctive_objects(catalogue),
    )
    assert [name for name, _score in ranked] == ["code_repl"]


#: One interpreter per hash seed. Sixteen is enough that the old behaviour,
#: which split roughly five to three, would fail this essentially always.
_TRIALS = 16

_PROBE = """
import sys
sys.path.insert(0, {repo!r})
from core.intent.declared_capability import (
    declared_vocabulary, distinctive_objects, rank_declaration_matches,
)
catalogue = {{
    "code_repl": declared_vocabulary(
        "code_repl", "Execute Python code in a sandboxed REPL"),
    "program_dna_equivalence_battery": declared_vocabulary(
        "program_dna_equivalence_battery",
        "Run a program equivalence battery across simulated applications"),
}}
ranked = rank_declaration_matches(
    "Run some Python and tell me the result.",
    catalogue, distinctive_objects(catalogue))
print(",".join(name for name, _score in ranked))
"""


@pytest.mark.slow
def test_the_same_catalogue_ranks_the_same_in_every_process():
    """Fresh interpreters, so string hashing is reseeded on each one."""
    seen = set()
    for _ in range(_TRIALS):
        finished = subprocess.run(
            [sys.executable, "-c", _PROBE.format(repo=str(REPO))],
            capture_output=True, text=True, cwd=str(REPO),
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "AURA_TESTING": "1"},
        )
        assert finished.returncode == 0, finished.stderr[-400:]
        seen.add(finished.stdout.strip().splitlines()[-1])
    assert seen == {"code_repl"}, f"ranking varied across processes: {seen}"


def test_the_live_catalogue_does_not_reach_the_fallback():
    """So this fix changes no live selection, only removes a way to vary."""
    import math
    from collections import Counter

    from core.capability_engine import CapabilityEngine

    catalogue = {
        name: declared_vocabulary(name, str(getattr(meta, "description", "") or ""))
        for name, meta in CapabilityEngine().skills.items()
        if getattr(meta, "enabled", True)
    }
    total = len(catalogue)
    frequency: Counter[str] = Counter()
    for _verbs, objects in catalogue.values():
        frequency.update(objects)
    cutoff = math.log(total / (1 + total * 0.5))
    stranded = [
        name
        for name, (_verbs, objects) in catalogue.items()
        if objects
        and not {w for w in objects if math.log(total / (1 + frequency[w])) > cutoff}
    ]
    assert stranded == [], f"these skills now depend on the fallback: {stranded}"
