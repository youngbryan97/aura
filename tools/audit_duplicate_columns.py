#!/usr/bin/env python3
"""Columns in two domains that are the same number.

The state schema splits the organism into ten domains and measures the coupling
between them. A column that is a verbatim copy of another domain's column
breaks that: the copy arrives with no cognition in between, so an edge is
manufactured in both directions and the partition score is computed over a
system that cannot be partitioned there.

Four of recurrent cognition's features were exactly this — the phenomenal field
rebuilds them once a turn from affect, the energy budget and the coherence
score. They were removed by reading the source. This reads the recording
instead, so a copy nobody has noticed shows up as a number.

    python tools/audit_duplicate_columns.py artifacts/subject_core/run_017

Correlation alone is not a copy: two columns can move together because one
causes the other, which is the thing the battery exists to find. What this
reports is a correlation at or above the threshold BETWEEN domains, which is
where a copy does its damage, and the reader decides. A pair at 1.0 to six
decimals is not a coupling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="a run directory holding core_state.npz")
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    from core.subject.state import DOMAINS, domain_width, feature_names

    blob = np.load(args.run / "core_state.npz", allow_pickle=True)
    matrix = np.asarray(blob["x"], dtype=np.float64)
    manifest = args.run / "core_state_manifest.json"
    names: list[str] = []
    if manifest.is_file():
        import json

        record = json.loads(manifest.read_text())
        for key in ("columns", "features", "schema"):
            value = record.get(key)
            if isinstance(value, list) and len(value) == matrix.shape[1]:
                names = [str(item) for item in value]
                break
    if not names:
        names = list(feature_names())
    if len(names) != matrix.shape[1]:
        # A recording taken under a different schema. Name what can be named
        # and say so, rather than lining the wrong labels up against the data.
        print(
            f"the recording has {matrix.shape[1]} columns and this schema names "
            f"{len(names)}; columns are reported by index"
        )
        names = [f"col_{index}" for index in range(matrix.shape[1])]
        owner = [f"blk_{index // 16}" for index in range(matrix.shape[1])]
    else:
        owner = []
        if matrix.shape[1] == sum(domain_width(d) for d in DOMAINS):
            for domain in DOMAINS:
                owner.extend([domain] * domain_width(domain))
        else:
            owner = [str(name).split(".")[0] for name in names]

    spread = matrix.std(axis=0)
    live = spread > 1e-12
    print(f"{matrix.shape[0]} rows, {matrix.shape[1]} columns, {int(live.sum())} of them move")

    centred = matrix - matrix.mean(axis=0)
    scaled = np.zeros_like(centred)
    scaled[:, live] = centred[:, live] / spread[live]
    correlation = (scaled.T @ scaled) / max(1, matrix.shape[0] - 1)

    pairs: list[tuple[float, str, str]] = []
    for i in range(matrix.shape[1]):
        if not live[i]:
            continue
        for j in range(i + 1, matrix.shape[1]):
            if not live[j] or owner[i] == owner[j]:
                continue
            r = float(correlation[i, j])
            if abs(r) >= args.threshold:
                pairs.append((abs(r), str(names[i]), str(names[j])))

    # A counter is the turn index in disguise.
    #
    # Every monotone column correlates with every other monotone column at
    # one, whatever domains they sit in, so a pair of them manufactures an
    # edge in both directions and makes a system look integrated where nothing
    # is coupled. What a counter carries is when it advanced, not how far it
    # has got.
    rising = [
        index
        for index in range(matrix.shape[1])
        if live[index] and bool(np.all(np.diff(matrix[:, index]) >= -1e-12))
    ]
    if rising:
        print(f"\n{len(rising)} column(s) never decrease across the recording:")
        for index in rising:
            print(f"  {names[index]}")

    pairs.sort(reverse=True)
    if not pairs:
        print(f"\nno cross-domain pair correlates at {args.threshold} or above.")
        return 0
    print(f"\n{len(pairs)} cross-domain pair(s) at |r| >= {args.threshold}:")
    for r, left, right in pairs[: args.top]:
        print(f"  {r:.6f}  {left:30s} {right}")
    if len(pairs) > args.top:
        print(f"  ... and {len(pairs) - args.top} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
