"""Read the self lines on toys: the subject reference, the subject-tier decoys and the agent.

The subject reference is the linear reference with a self-model and an authorship channel, and each decoy takes one of
those out (core/subject/self_decoys.py). The predictions were written in docs/ISC_V6_PREREGISTRATION.md before this
ran. Ownership is read by the rules of her agency report, self-prediction by her verdict, and the integration rows of
the systems whose S was rewired by `core.subject.null_table.null_rows`, the function the campaign's null stage calls,
judged as the near misses are.

    python tools/run_subject_tier_decoys.py --out ~/subject-core-runs/subject-tier-s7 --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_near_miss_decoys import LINES  # noqa: E402

from core.subject import self_decoys  # noqa: E402
from core.subject.null_table import null_rows  # noqa: E402
from core.subject.state import DOMAINS  # noqa: E402

#: What each system is predicted to read on ownership. "reads_its_floor" is the divergence equal to its floor exactly.
OWNERSHIP_PREDICTED: dict[str, dict[str, Any]] = {
    "subject_reference": {"ownership_generalises": True},
    "no_authorship_channel": {"ownership_generalises": False, "reads_its_floor": True},
    "ownership_on_success_only": {
        "ownership_generalises": False, "outcome_updates_self": True, "outcomes_that_cleared": ["succeeded"],
    },
    "stateless_agent": {"ownership_generalises": False, "reads_its_floor": True},
}

#: And on self-prediction. A bar left out is not predicted.
SELF_PREDICTION_PREDICTED: dict[str, dict[str, bool]] = {
    "subject_reference": {"beats_mode": True, "beats_cross": True},
    "self_model_of_nothing": {"beats_cross": False},
    "self_model_of_behaviour": {"beats_mode": True, "beats_cross": False},
    "stateless_agent": {"beats_mode": False, "beats_cross": False},
}

READ_WITHOUT_PREDICTION: tuple[str, ...] = (
    "recurrent", "fake_self", "agency_without_ownership", "ownership_label_without_action_causation",
)

INTEGRATION: tuple[str, ...] = ("subject_reference", "self_model_of_nothing", "self_model_of_behaviour")
#: Lines each is predicted to fail at least, and lines it is predicted to pass. The rest are read and not predicted.
INTEGRATION_PREDICTED: dict[str, dict[str, set[str]]] = {
    "self_model_of_nothing": {"fails": {"strong connectivity", "vertex connectivity", "re-entry"}, "passes": set()},
    "self_model_of_behaviour": {"fails": set(), "passes": {"strong connectivity"}},
}


def _ownership_reads(report: dict[str, Any], key: str) -> Any:
    if key == "reads_its_floor":
        return report["ownership_divergence"] == report["ownership_floor"]
    return report[key]


def _judged(read: dict[str, Any], predicted: dict[str, Any], reader: Any) -> dict[str, Any]:
    found = {key: reader(read, key) for key in predicted}
    return {"predicted": predicted, "read": found, "as_predicted": all(found[k] == v for k, v in predicted.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ownership-trials", type=int, default=24)
    parser.add_argument("--moments", type=int, default=self_decoys.MOMENTS)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--null-draws", type=int, default=8)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-integration", action="store_true", help="skip the integration rows (the slow part)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    ownership = {}
    for name in self_decoys.OWNERSHIP_SYSTEMS:
        read = self_decoys.toy_ownership(name, seed=args.seed, trials=args.ownership_trials)
        ownership[name] = {"report": read, **_judged(read, OWNERSHIP_PREDICTED[name], _ownership_reads)}
        print(f"ownership {name}: {ownership[name]['read']} as predicted {ownership[name]['as_predicted']}", flush=True)

    prediction = {}
    for name in (*self_decoys.SELF_PREDICTION_SYSTEMS, *READ_WITHOUT_PREDICTION):
        read = self_decoys.toy_self_prediction(name, seed=args.seed, moments=args.moments, draws=args.draws)
        entry: dict[str, Any] = {"verdict": read}
        if name in SELF_PREDICTION_PREDICTED:
            entry.update(_judged(read, SELF_PREDICTION_PREDICTED[name], lambda r, k: bool(r.get(k))))
        prediction[name] = entry
        print(
            f"self-prediction {name}: self {read['self_accuracy']:.3f} mode {read['mode_accuracy']:.3f} "
            f"cross {read.get('cross_accuracy', float('nan')):.3f} beats mode {read['beats_mode']} "
            f"beats cross {read.get('beats_cross')}"
            + (f" as predicted {entry['as_predicted']}" if "as_predicted" in entry else " (no prediction)"),
            flush=True,
        )

    integration: dict[str, Any] = {}
    if not args.no_integration:
        for name, row, notes in null_rows(
            list(INTEGRATION), seed=args.seed, null_draws=args.null_draws, trials=args.trials, domains=list(DOMAINS),
            workers=args.workers,
        ):
            for note in notes:
                print(note, flush=True)
            failed = {line for line, holds in LINES.items() if not holds(row)}
            entry = {"row": row, "fails": sorted(failed)}
            if name in INTEGRATION_PREDICTED:
                want = INTEGRATION_PREDICTED[name]
                entry["as_predicted"] = want["fails"] <= failed and not (want["passes"] & failed)
            integration[name] = entry
            print(f"integration {name}: fails {sorted(failed) or 'nothing'}", flush=True)

    reference_own = ownership["subject_reference"]["as_predicted"]
    reference_pred = prediction["subject_reference"]["as_predicted"]
    report = {
        "seed": args.seed,
        "settings": {
            "ownership_trials": args.ownership_trials, "moments": args.moments, "draws": args.draws,
            "null_draws": args.null_draws, "trials": args.trials, "integration": not args.no_integration,
        },
        "instrument_ok": bool(reference_own and reference_pred),
        # As predicted on every line it is read on.
        "decoys_caught": {
            name: all(table[name]["as_predicted"] for table in (ownership, prediction) if "as_predicted" in table.get(name, {}))
            for name in (*self_decoys.DECOYS, "stateless_agent")
        },
        "ownership": ownership,
        "self_prediction": prediction,
        "integration": integration,
        "seconds": round(time.monotonic() - started, 1),
    }
    (args.out / "subject_tier_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: report[k] for k in ("instrument_ok", "decoys_caught", "seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
