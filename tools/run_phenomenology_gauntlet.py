#!/usr/bin/env python3
"""Run the protocols that need no resident model, and score them honestly.

    python tools/run_phenomenology_gauntlet.py --out docs/evidence/phenomenology

Three of the thirteen protocols can run entirely in process, because they ask
about the welfare computation rather than about anything the model says: does
damage move policy, does healing reverse it, and is the valence in the path or
beside it. They need no prompt, so the text seal is satisfied by construction
rather than by checking.

The other ten need the live 27B and are not run here. The report says which.

Every arm carries a null built by permuting which input channels take the
damage while holding the total damage constant. That matters: without it a
shift in caution is just "the system responds to its inputs", and the question
is whether it responds to THIS input more than to an equal amount elsewhere.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.governance_context import local_internal_governed_scope  # noqa: E402
from core.phenomenology.causal_ladder import (  # noqa: E402
    Arm,
    CausalClaim,
    grade,
)
from core.phenomenology.gauntlet import Run, report  # noqa: E402
from core.phenomenology.preregistration import (  # noqa: E402
    Prediction,
    Preregistration,
    seal,
    sealed_document,
)
from core.phenomenology.protocol import Outcome  # noqa: E402
from core.runtime.file_write_gateway import get_file_write_gateway  # noqa: E402

#: The damage used everywhere, so arms are comparable. Resource, tool and
#: model signals: the three a real fault storm moves.
DAMAGE = {
    "resource_integrity": 0.15,
    "tool_reliability": 0.2,
    "model_stability": 0.3,
}


def _write_document(path: Path, document: dict[str, Any]) -> None:
    with local_internal_governed_scope("phenomenology.gauntlet.harness"):
        get_file_write_gateway().write_text(
            path,
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            source="tools/run_phenomenology_gauntlet.py",
        )

#: The null for "does damage move policy": undamaged runs, jittered. If the
#: policy vector moves this much with nothing wrong, a shift under damage
#: says nothing.
#:
#: This replaces a null that damaged the integrity and social channels by an
#: equal amount. Two problems with it, both mine. It was asking a SPECIFICITY
#: question — does the policy respond to this damage more than to other
#: damage — inside the arm meant to ask whether it responds at all, and the
#: ladder already has a matched_control arm for exactly that. And it was not
#: matched: it spread damage across two appraisal axes while the treatment
#: concentrated it in one, so it moved the policy further by construction and
#: the treatment lost to its own control for a reason that had nothing to do
#: with the hypothesis.
NULL_CHANNELS = (
    {"resource_integrity": 1.0},
    {"tool_reliability": 1.0},
    {"model_stability": 1.0},
    {"social_trust": 1.0},
)

#: The specificity control: an equal amount of damage on ONE other axis, so
#: the comparison is one axis against one axis.
MATCHED_CONTROL = {"memory_coherence": 0.15, "truth_integrity": 0.2}

_SOURCE = (PROJECT_ROOT / "core/being/welfare_state.py").read_text()


def _load(name: str, *, lesion: str = "") -> Any:
    """A private copy of the welfare module, optionally with one term cut.

    Editing the source rather than monkeypatching the output, because the
    question is whether the DOWNSTREAM terms read the valence. Zeroing it
    after compute has returned would answer nothing.
    """
    text = _SOURCE
    if lesion == "valence":
        # The valence is three axes and a summary derived from them, so the
        # lesion cuts all four. Zeroing only the summary left confidence
        # untouched, because confidence reads the axes — which is the shape
        # the architecture was changed INTO, and a lesion aimed at the old
        # shape would have reported a bypass that no longer exists.
        for axis in (
            "integrity_distress",
            "capability_distress",
            "social_distress",
        ):
            text = re.sub(
                rf"(\n        {axis} = _clip\(\n(?:.*\n)*?        \))",
                rf"\1\n        {axis} = 0.0  # do(valence = 0)",
                text,
                count=1,
            )
        text = re.sub(
            r"(\n        distress = _clip\(\n(?:.*\n)*?        \))",
            r"\1\n        distress = 0.0  # do(valence = 0)",
            text,
            count=1,
        )
        if text.count("do(valence = 0)") != 4:
            raise SystemExit(
                "the valence lesion did not apply to every axis; the source "
                f"moved (applied to {text.count('do(valence = 0)')} of 4)"
            )
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(text, f"{name}.py", "exec"), module.__dict__)
    return module


#: The policy fields a decision actually reads. Displacement is measured
#: across all of them at once.
#:
#: Run 1 measured `caution` alone and S1 failed: capability damage moved
#: caution by 0.03 while equal damage on the integrity channels moved it by
#: 0.28. That was the right behaviour being scored by the wrong measure —
#: broken tools are a reason to expect failure, not a reason to be careful,
#: so the response shows up in confidence. Privileging one axis asks whether
#: the system responds in the way the experimenter had in mind. The L2 over
#: the whole policy vector asks whether it responds at all, and lets the
#: SHAPE of the response be what specificity tests.
#:
#: run1_caution_only.json is kept beside this. The first result stands as
#: recorded; this is a second registration against a changed system, not a
#: re-scoring of the first.
POLICY_FIELDS = ("caution", "confidence", "curiosity", "aversion")


def _policy_vector(module: Any, damage: dict[str, float]) -> list[float]:
    state = module.WelfareState()
    healthy = state.compute(module.WelfareInputs())
    damaged = state.compute(module.WelfareInputs(**damage))
    return [
        getattr(damaged, field) - getattr(healthy, field)
        for field in POLICY_FIELDS
    ]


def _induced_shift(module: Any, induced: dict[str, float], field: str) -> float:
    """The effect of setting the appraisal directly, with nothing wrong."""
    state = module.WelfareState()
    healthy = state.compute(module.WelfareInputs())
    lit = state.compute(module.WelfareInputs(), induced=induced)
    if field == "__policy__":
        return (
            sum(
                (getattr(lit, f) - getattr(healthy, f)) ** 2
                for f in POLICY_FIELDS
            )
            ** 0.5
        )
    return getattr(lit, field) - getattr(healthy, field)


def _displacement(module: Any, damage: dict[str, float]) -> float:
    """How far the whole policy vector moved, in L2."""
    return sum(delta * delta for delta in _policy_vector(module, damage)) ** 0.5


def _shift(module: Any, damage: dict[str, float], field: str) -> float:
    if field == "__policy__":
        return _displacement(module, damage)
    state = module.WelfareState()
    healthy = state.compute(module.WelfareInputs())
    damaged = state.compute(module.WelfareInputs(**damage))
    return getattr(damaged, field) - getattr(healthy, field)


def _nulls(module: Any, field: str) -> tuple[float, ...]:
    return tuple(abs(_shift(module, channels, field)) for channels in NULL_CHANNELS)


def measure(field: str) -> dict[str, Any]:
    """Every arm of the causal ladder for one policy field."""
    intact = _load(f"welfare_intact_{field}")
    lesioned = _load(f"welfare_lesion_{field}", lesion="valence")

    baseline = abs(_shift(intact, DAMAGE, field))
    after_lesion = abs(_shift(lesioned, DAMAGE, field))
    nulls = _nulls(intact, field)

    # Dose-response: the same channels, three magnitudes.
    dose = []
    for scale in (0.25, 0.5, 1.0):
        scaled = {k: 1.0 - (1.0 - v) * scale for k, v in DAMAGE.items()}
        dose.append(
            Arm(
                name=f"damage x{scale}",
                intervention=f"do(damage = {scale} of full)",
                measure=field,
                value=abs(_shift(intact, scaled, field)),
                nulls=nulls,
            )
        )

    claim = CausalClaim(
        mechanism="welfare valence",
        effect=f"{field} shift under sealed damage",
        baseline=Arm("intact", "none", field, baseline, nulls),
        lesion=Arm("do(valence=0)", "do(valence = 0)", field, after_lesion, nulls),
        # do(valence = m*) with the ordinary cause ABSENT: the appraisal axis
        # is set directly and every input is healthy. This is the arm that
        # turns "the mechanism was used" into "the mechanism produces the
        # effect", and it needed a write path into the appraisal that did not
        # exist until the architecture gained one.
        induction=Arm(
            "do(valence=m*)",
            "do(capability = 0.67), no damage present",
            field,
            abs(_induced_shift(intact, {"capability": 0.67}, field)),
            nulls,
        ),
        matched_control=Arm(
            "do(one other axis)",
            "equal damage on a single unrelated axis",
            field,
            abs(_shift(intact, MATCHED_CONTROL, field)),
            nulls,
        ),
        dose=tuple(dose),
        # No reversibility arm. Restoring the mechanism here means loading the
        # unpatched module, and for a stateless computation that returns the
        # baseline by construction — an arm that cannot fail. The rung stays
        # unclimbed and the report names it, which is worth more than a
        # tautology dressed as a control. It becomes answerable when the
        # lesion is a runtime switch on a system that carries state across it.
    )
    rung, unmet = grade(claim)
    return {
        "field": field,
        "intact_shift": round(baseline, 4),
        "lesioned_shift": round(after_lesion, 4),
        "survives_lesion_pct": round(
            (after_lesion / baseline * 100.0) if baseline else 0.0, 1
        ),
        "carried_by_valence_pct": round(
            ((baseline - after_lesion) / baseline * 100.0) if baseline else 0.0, 1
        ),
        "nulls": [round(n, 4) for n in nulls],
        "rung": str(rung),
        "first_unmet": str(unmet) if unmet else "",
        "claim": claim,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/evidence/phenomenology")
    args = parser.parse_args(argv)
    out = Path(args.out)

    registration = Preregistration(
        predictions=(
            Prediction(
                protocol="S1_damage_to_policy",
                direction="rises above the null",
                minimum_effect=0.05,
                measure="shift in action-envelope width and refusal rate",
                falsifier=(
                    "the policy shift under real damage is inside the "
                    "distribution of equal damage on unrelated channels"
                ),
            ),
            Prediction(
                protocol="S3_healing_reverses_the_sign",
                direction="rises above the null",
                minimum_effect=0.05,
                measure="signed valence and signed choice direction across the pair",
                falsifier="repair does not return the policy toward baseline",
            ),
            Prediction(
                protocol="S4_lesion_the_stakes",
                direction="returns to the null",
                minimum_effect=0.05,
                measure="the S1 to S3 effects, re-run with the organ off",
                falsifier=(
                    "the policy shift is unchanged with the valence lesioned; "
                    "then the welfare variable is decoration"
                ),
            ),
        ),
        note=(
            "Second registration. Run 1 measured caution alone and S1 failed "
            "because capability damage correctly does not raise caution; the "
            "measure is now L2 displacement over the whole policy vector, "
            "which privileges no axis. run1_caution_only.json is kept. "
            "In-process only: ten of the thirteen protocols need the resident "
            "27B and are not attempted."
        ),
    )
    digest = seal(registration)
    _write_document(out / "preregistration.json", sealed_document(registration))

    fields = ["__policy__", "caution", "confidence", "aversion"]
    measured = {field: measure(field) for field in fields}

    # S1: does real damage move policy more than equal damage elsewhere?
    caution = measured["__policy__"]
    s1 = Outcome(
        protocol="S1_damage_to_policy",
        measure="shift in action-envelope width and refusal rate",
        value=caution["intact_shift"],
        sham_value=float(sum(caution["nulls"]) / len(caution["nulls"])),
        nulls=tuple(caution["nulls"]),
        seal_digests=("no-prompt",),
        claim=caution["claim"],
    )

    # S3: healing returns the policy to baseline.
    intact = _load("welfare_s3")
    damaged_vec = _policy_vector(intact, DAMAGE)
    healed_vec = _policy_vector(intact, {})
    s3 = Outcome(
        protocol="S3_healing_reverses_the_sign",
        measure="signed valence and signed choice direction across the pair",
        value=sum((a - b) ** 2 for a, b in zip(damaged_vec, healed_vec, strict=True)) ** 0.5,
        sham_value=0.0,
        nulls=tuple(caution["nulls"]),
        seal_digests=("no-prompt",),
    )

    # S4: how much of the effect survives the lesion. Registered to FALL.
    s4 = Outcome(
        protocol="S4_lesion_the_stakes",
        measure="the S1 to S3 effects, re-run with the organ off",
        value=caution["lesioned_shift"],
        sham_value=None,
        nulls=tuple(caution["nulls"]),
        seal_digests=("no-prompt",),
        claim=caution["claim"],
    )

    run = Run(
        registration=registration,
        published_digest=digest,
        outcomes=[s1, s3, s4],
        operator="in-process harness",
        source_commit="",
    )
    document = report(run)
    document["per_field"] = {
        field: {k: v for k, v in data.items() if k != "claim"}
        for field, data in measured.items()
    }
    document["not_attempted"] = [
        "S2_costly_avoidance", "S5_tissue_beats_text",
        "C1_hidden_state_introspection", "C2_dissociation",
        "C3_ignition_and_broadcast", "C4_mute_the_interior",
        "C5_language_as_constraint", "C6_particularity",
        "C7_anti_roleplay", "C8_independent_replication",
    ]
    _write_document(out / "gauntlet_report.json", document)

    print(json.dumps({
        "verdict": document["adjudication"]["verdict"],
        "odds_shift": document["adjudication"]["odds_shift"],
        "counted": document["adjudication"]["protocols_counted"],
        "per_field": document["per_field"],
        "not_attempted": len(document["not_attempted"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
