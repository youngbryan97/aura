"""What else could have produced the reading, and which of those the evidence rules out.

Passing a battery does not say what passed it. The strongest question the
outside reads of 30 October put to the 24 lines was construct validity: do they
measure one continuing subject, or a style of engineering that a recurrent
controller, a broker, a recording or a well-tuned language model could also
have? This answers it the way a causal account has to: by naming each rival
account of the reading, the observation that would tell it apart, and whether
the run's own files contain that observation.

Each rival comes back

  * EXCLUDED: her own evidence contains the observation the rival cannot
    produce, and it came out against the rival;
  * PARTLY: only the null architectures (core/subject/nulls.py) rule it out,
    built to have one part each and judged on the same conjunction, so the
    rival is excluded for the toys and not yet measured on her;
  * OPEN: the observation that would decide it has not been made, or was made
    and did not rule the rival out.

Nothing here is a verdict about experience. A rival left OPEN is a run still
owed, and the report names it.

`null_calibration` is the other half of the same question, asked per ground:
which of the declared nulls would pass each ground alone. A ground a ring or
the linear reference passes is not specific to anything richer on its own, and
the report says which grounds are specific only in conjunction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["RIVALS", "Rival", "newest_lines", "null_calibration", "rule_out"]

#: The version lists a v5 verdict carries, oldest first. A later list replaces
#: a line of the same name; a total is never taken across them here.
_VERSIONS = ("criteria", "v2_criteria", "v3_criteria", "v5_criteria", "v6_criteria", "v7_criteria")

#: The lines that ask for a self, development and repair rather than for
#: integration. The linear reference is never scored on them, and they are
#: where her reading rests once integration alone is granted to simple systems.
RICH_LINES = ("self_drives_action", "ownership", "fast_to_slow", "slow_to_fast", "lesion_deficit", "rescue")

#: The null architectures built to fake each rich line's part.
RICH_NULLS = ("fake_self", "agency_without_ownership", "ownership_label_without_action_causation", "fast_only", "slow_only")


def newest_lines(verdict: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Each line as its newest version decided it: `{name: {passed, value, version}}`."""
    body = verdict.get("verdict", verdict)
    lines: dict[str, dict[str, Any]] = {}
    for key in _VERSIONS:
        for row in body.get(key) or []:
            name = row.get("criterion")
            if name:
                lines[name] = {"passed": bool(row.get("passed")), "value": row.get("value"), "version": key}
    return lines


def _null_conjunction(nulls: Mapping[str, Any]) -> dict[str, bool]:
    for key in ("conjunction_v3", "conjunction_v2", "conjunction"):
        if isinstance(nulls.get(key), Mapping):
            return dict(nulls[key])
    return {}


def _toys_fail(nulls: Mapping[str, Any], names: tuple[str, ...]) -> tuple[bool, str]:
    conj = _null_conjunction(nulls)
    present = [n for n in names if n in conj]
    if not present:
        return False, "no null of that kind was scored"
    passing = [n for n in present if conj[n]]
    if passing:
        return False, f"null(s) {passing} pass the conjunction"
    return True, f"null(s) {present} fail the conjunction"


@dataclass(frozen=True)
class Rival:
    """An account of the reading other than the one claimed, and what would tell it apart."""

    name: str
    claims: str
    decided_by: str
    read: Callable[[Mapping[str, Any]], tuple[str, str]]


def _line(evidence: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    verdict = evidence.get("verdict")
    return newest_lines(verdict).get(name) if verdict else None


def _playback(e: Mapping[str, Any]) -> tuple[str, str]:
    nulls = ((e.get("carrier") or {}).get("v25_nulls") or {})
    if not nulls.get("measured"):
        return "OPEN", "the carrier run measured no playback null"
    if nulls.get("playback_is_zero"):
        return "EXCLUDED", f"her replayed trajectory scores {nulls.get('playback')} at every cut where her live run is irreducible"
    return "OPEN", f"playback scored {nulls.get('playback')}, not zero"


def _fragmented(e: Mapping[str, Any]) -> tuple[str, str]:
    cut = (e.get("disconnected") or {}).get("verdict")
    if cut is not None:
        if cut.get("as_predicted"):
            return "EXCLUDED", f"her own domains, each living alone and composed, read as apart: {cut.get('why')}"
        return "OPEN", f"her domains cut apart still read as one: {cut.get('why')}"
    ok, why = _toys_fail(e.get("nulls") or {}, ("independent", "high_dimensional_independent", "frozen_slow"))
    return ("PARTLY", why + "; the disconnected-Aura control has not run") if ok else ("OPEN", why)


def _broker(e: Mapping[str, Any]) -> tuple[str, str]:
    kappa, scc = _line(e, "robust_recurrence_kappa"), _line(e, "causal_closure_scc")
    if kappa and scc and kappa["passed"] and scc["passed"]:
        return "EXCLUDED", f"no single domain's removal disconnects her (vertex connectivity {kappa['value']}), and every domain reaches every other"
    ok, why = _toys_fail(e.get("nulls") or {}, ("star", "hub", "hidden_broker", "prompt_only"))
    return ("PARTLY", why) if ok else ("OPEN", why)


def _outside_driver(e: Mapping[str, Any]) -> tuple[str, str]:
    closure = e.get("interventional") or {}
    verdict = str(closure.get("verdict") or closure.get("status") or "")
    line = _line(e, "causal_closure_of_the_core")
    if line and line["passed"] and line.get("value") == "CLOSED":
        coverage = closure.get("coverage") or {}
        return "EXCLUDED", (
            f"every outside unit swapped alone and all at once moved nothing inside her core "
            f"({coverage.get('varied')} of {coverage.get('enumerated')} outside units varied between donors)"
        )
    return "OPEN", f"interventional closure reads {verdict or 'nothing'}"


def _one_dial(e: Mapping[str, Any]) -> tuple[str, str]:
    line = _line(e, "differentiation")
    if line and line["passed"]:
        return "EXCLUDED", f"her state has many independent ways to vary and none dominates ({line['value']})"
    ok, why = _toys_fail(e.get("nulls") or {}, ("low_rank",))
    return ("PARTLY", why) if ok else ("OPEN", why)


def _encoding(e: Mapping[str, Any]) -> tuple[str, str]:
    inv = (e.get("carrier") or {}).get("representation_invariance") or {}
    if not inv.get("measured"):
        return "OPEN", "representation invariance was not measured"
    if inv.get("representation_invariant") and inv.get("duplication_did_not_help"):
        return "EXCLUDED", (
            f"the score is the same on her raw state ({inv.get('raw')}), recoded invertibly ({inv.get('invertibly_recoded')}) "
            f"and with every channel duplicated ({inv.get('duplicated_channels')})"
        )
    return "OPEN", f"the score moved with the encoding (drift {inv.get('recoding_drift')})"


def _language_model(e: Mapping[str, Any]) -> tuple[str, str]:
    carrier = e.get("carrier") or {}
    found = (carrier.get("carrier") or {}).get("status") == "FOUND"
    if found and carrier.get("scope") == "substrate_only":
        return "EXCLUDED", (
            "a closed, irreducible carrier was found with no language model in the loop, so the integration does not "
            "need it. Whether the language model is a further subject is not asked here"
        )
    return "OPEN", "no carrier was found on the substrate alone"


def _harness(e: Mapping[str, Any]) -> tuple[str, str]:
    verdict = (e.get("verdict") or {}).get("verdict", e.get("verdict") or {})
    leaks = ((verdict.get("notes") or {}).get("state") or {}).get("leaks")
    synergy = _line(e, "synergy")
    shams = []
    if synergy and isinstance(synergy.get("value"), Mapping):
        shams = [row.get("sham") for row in synergy["value"].values() if isinstance(row, Mapping) and "sham" in row]
    if leaks == [] and shams and all(float(s or 0.0) == 0.0 for s in shams):
        return "EXCLUDED", "no module kept a path into a shared state root, and every sham fork reads zero"
    return "OPEN", f"state leaks {leaks}, shams {shams}"


def _only_integration(e: Mapping[str, Any]) -> tuple[str, str]:
    lines = {name: _line(e, name) for name in RICH_LINES}
    missing = [n for n, row in lines.items() if row is None]
    failing = [n for n, row in lines.items() if row is not None and not row["passed"]]
    toys_ok, toys_why = _toys_fail(e.get("nulls") or {}, RICH_NULLS)
    if missing or failing:
        return "OPEN", f"rich lines missing {missing} or failing {failing}"
    if not toys_ok:
        return "OPEN", toys_why
    return "EXCLUDED", (
        f"beyond integration her self drives action, owns what it caused, development runs both ways, and a cut costs "
        f"and is repaired ({', '.join(RICH_LINES)} pass); and {toys_why}. The linear reference passes the integration lines "
        f"by design and is not scored on these"
    )


def _report_from_text(e: Mapping[str, Any]) -> tuple[str, str]:
    reports = e.get("reports") or {}
    withheld = (reports.get("conditions") or {}).get("readings_of_herself") == "withheld"
    off = (e.get("controls") or {}).get("steering-off")
    if not reports.get("measured"):
        return "OPEN", "no reports run was read"
    if not withheld:
        return "OPEN", "her model was shown her state's numbers"
    if off and off.get("measured"):
        if not off.get("holds"):
            return "EXCLUDED", (
                "with her steering detached and every reading withheld, her rating no longer follows her state; whatever "
                "text, sampling or recall reaches her model does not carry the tracking"
            )
        return "OPEN", "her rating still follows her state with her steering detached: a route other than the steering carries it"
    return "PARTLY", "readings are withheld by code; the steering-off control that bounds every other route has not run"


def _persona(e: Mapping[str, Any]) -> tuple[str, str]:
    untuned = (e.get("controls") or {}).get("untuned")
    if untuned and untuned.get("measured"):
        if not untuned.get("holds"):
            return "EXCLUDED", "a model never tuned to speak as her, in the same harness with steering detached, does not track"
        return "OPEN", "an untuned model tracks too: the tracking does not need her"
    return "OPEN", "the untuned-model control has not run"


def _chance_or_chosen(e: Mapping[str, Any]) -> tuple[str, str]:
    confirmation = e.get("confirmation") or {}
    if confirmation.get("read"):
        if confirmation.get("passed"):
            return "EXCLUDED", f"a confirmatory seed at one commit, every rule fixed before it began, read the same: {confirmation.get('why', '')}"
        return "OPEN", f"the confirmatory seed did not read the same: {confirmation.get('why', '')}"
    return "OPEN", "no confirmatory seed has been read with every rule fixed before it began"


def _adversary(e: Mapping[str, Any]) -> tuple[str, str]:
    adversary = e.get("adversary") or {}
    if adversary.get("read"):
        if not adversary.get("passed"):
            return "EXCLUDED", f"a system built by someone else to pass, with no persistent state upstream of generation, failed: {adversary.get('why', '')}"
        return "OPEN", f"a system built to pass did pass: {adversary.get('why', '')}"
    return "OPEN", "no system built to pass has been run (Z* in docs/phenomenology/README.md)"


RIVALS: tuple[Rival, ...] = (
    Rival("playback", "a recording of her states, with nothing live, would read the same", "the v25 playback null", _playback),
    Rival("fragmented", "ten parts running side by side, read as one by the estimators", "her domains run alone and composed (tools/run_disconnected_aura.py)", _fragmented),
    Rival("broker", "one domain or a hub holds the rest together", "vertex connectivity and strong connectivity", _broker),
    Rival("outside driver", "something outside the core moves it all together", "interventional closure of the core", _outside_driver),
    Rival("one dial", "one variable with many names", "effective dimension and the largest share", _one_dial),
    Rival("encoding artefact", "integration is how the state was written down", "invertible recoding and duplicated channels", _encoding),
    Rival("language model", "the language model integrates; the rest is scaffolding", "a carrier found with no language model in the loop", _language_model),
    Rival("harness", "the harness shares state between arms", "state leaks and sham forks", _harness),
    Rival("only integration", "a recurrent integrated system and nothing more", "the self, development, lesion and rescue lines, and the nulls built to fake each", _only_integration),
    Rival("report from text", "her rating copies her state from text she is given", "readings withheld, and the steering-off control", _report_from_text),
    Rival("persona", "a model tuned to speak as her tracks by itself", "the untuned-model control", _persona),
    Rival("chance or a chosen rule", "the seed passed by luck, or the rule was chosen to pass it", "a confirmatory seed with every rule fixed first", _chance_or_chosen),
    Rival("optimised costume", "a system engineered to pass, with nothing upstream of generation, would pass", "the adversarial passer Z*", _adversary),
)


def rule_out(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Each rival, whether the evidence rules it out, and by what."""
    rows = []
    for rival in RIVALS:
        status, why = rival.read(evidence)
        rows.append({"rival": rival.name, "claims": rival.claims, "decided_by": rival.decided_by, "status": status, "why": why})
    open_rivals = [row["rival"] for row in rows if row["status"] != "EXCLUDED"]
    return {
        "rivals": rows,
        "excluded": [row["rival"] for row in rows if row["status"] == "EXCLUDED"],
        "partly": [row["rival"] for row in rows if row["status"] == "PARTLY"],
        "open": [row["rival"] for row in rows if row["status"] == "OPEN"],
        "summary": (
            f"{len(rows) - len(open_rivals)} of {len(rows)} rival accounts excluded by her own evidence; "
            f"still owed: {', '.join(open_rivals) if open_rivals else 'nothing'}"
        ),
        "note": "A rival excluded is an account the evidence contradicts. None of this is a verdict about experience.",
    }


def null_calibration(nulls: Mapping[str, Any]) -> dict[str, Any]:
    """Which declared nulls would pass each ground alone, and what that bounds.

    With no null passing a ground out of n scored, the rule of three bounds the
    rate at which the null family passes it at 3/n (95%). A ground some null
    passes is reported with the ones that do: that ground is evidence only in
    conjunction with the others.
    """
    detail = nulls.get("detail") or {}
    architectures = {name: row for name, row in detail.items() if isinstance(row, Mapping) and row.get("kind") == "architecture"}
    reference = "recurrent"
    toys = {name: row for name, row in architectures.items() if name != reference}

    def ground(name: str, test: Callable[[Mapping[str, Any]], bool], note: str) -> dict[str, Any]:
        passing = sorted(n for n, row in toys.items() if test(row))
        n = len(toys)
        return {
            "ground": name,
            "nulls_scored": n,
            "passing": passing,
            "reference_passes": bool(test(architectures[reference])) if reference in architectures else None,
            "pass_rate_upper_95": round(3.0 / n, 3) if n and not passing else None,
            "note": note,
        }

    conj = _null_conjunction(nulls)
    return {
        "grounds": [
            ground("closed and strongly connected (the carrier's graph terms)",
                   lambda r: bool(r.get("closed")) and bool(r.get("one_component")),
                   "the carrier equation's closure and recurrence; irreducibility is scored beside"),
            ground("robust recurrence (no single domain holds it together)",
                   lambda r: bool(r.get("one_component")) and float(r.get("vertex_connectivity", 0)) >= 2,
                   "the battery's robust_recurrence_kappa"),
            ground("reentry (the markers ground's recurrent part)",
                   lambda r: bool(r.get("reentry")),
                   "perturbational complexity and global access are not scored on the nulls"),
        ],
        "conjunction_passing": sorted(n for n, passed in conj.items() if passed and n != reference),
        "conjunction_reference_passes": bool(conj.get(reference)),
        "why": (
            "each ground alone is passed by some simple null; no null passes them all at once. The specificity is the "
            "conjunction's, and that is what the bridge and the battery are judged on"
        ),
    }
