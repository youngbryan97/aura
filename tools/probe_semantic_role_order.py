#!/usr/bin/env python3
"""Test whether resident source features carry noncommutative role order."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _role_row(item):
    import numpy as np

    from core.learning.semantic_span_pointer import _hidden_array

    instructions = item.ir.instructions
    if len(instructions) != 2 or len(instructions[1].args) != 2:
        return None
    result = len(item.public_inputs)
    args = instructions[1].args
    if args.count(result) != 1:
        return None
    hidden = np.asarray(_hidden_array(item.hidden_states), dtype=np.float32)
    span = instructions[1].operation_span
    span.validate_bound(len(hidden))
    if span.start == span.end or hidden.ndim != 2 or not np.isfinite(hidden).all():
        raise ValueError("role order probe needs finite measured operation features")
    operation = hidden[span.start:span.end].mean(axis=0)
    context = hidden.mean(axis=0)
    reference = None
    reference_label = None
    if (item.register_definition_origin == "explicit_annotation"
            and len(item.register_definition_spans) > result
            and len(instructions[1].argument_spans) == 2):
        def vector(bound):
            bound.validate_bound(len(hidden))
            return hidden[bound.start:bound.end].mean(axis=0)

        surface = sorted(zip(instructions[1].argument_spans, args, strict=True),
                         key=lambda pair: (pair[0].start, pair[0].end))
        if surface[0][0] == surface[1][0]:
            raise ValueError("reference probe cannot order identical mentions")
        left, right = (vector(bound) for bound, _owner in surface)
        reference_label = int(surface[0][1] == result)
        result_definition = vector(item.register_definition_spans[result])
        other = args[1] if args[0] == result else args[0]
        reserved_definition = vector(item.register_definition_spans[other])
        reference = np.asarray((left @ result_definition, left @ reserved_definition,
                                right @ result_definition, right @ reserved_definition,
                                (left - right) @ (result_definition - reserved_definition)),
                               dtype=np.float32)
    return {"source": item.ir.source_text_sha256,
            "construction": item.construction_id,
            "label": int(args[0] == result),
            "operation": operation, "context": context, "reference": reference,
            "reference_label": reference_label}


def _accuracy(rows, predicted):
    grouped = defaultdict(list)
    for row, value in zip(rows, predicted, strict=True):
        grouped[row["construction"]].append(int(value == row["label"]))
    return {"population": len(rows),
            "correct": sum(int(value == row["label"])
                           for row, value in zip(rows, predicted, strict=True)),
            "by_construction": {name: {"population": len(values),
                                       "correct": sum(values)}
                                for name, values in sorted(grouped.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("transducer", "source-report", "feature-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
    )

    configure_refit_environment(args.output)
    source_raw = args.source_report.read_bytes()
    source_report = json.loads(source_raw)
    transducer = compositional_semantic_program_transducer_from_dict(
        json.loads(args.transducer.read_bytes()))
    bundles = [name + "=" + str(args.feature_root / name) for name in
               source_report["representation_compatibility"]
               ["source_feature_manifest_sha256s"]]
    examples = load_source_examples(transducer, source_report, bundles)
    groups = {split: [row for item in examples if item.split == split
                      if (row := _role_row(item)) is not None]
              for split in ("train", "validation", "test")}
    train = groups["train"]
    if (len(train) < 2 or len({row["label"] for row in train}) != 2
            or len({row["source"] for row in train}) != len(train)):
        raise ValueError("source training lacks independent role-order contrasts")
    majority = Counter(row["label"] for row in train).most_common(1)[0][0]
    results = {}
    for view in ("operation", "context", "operation_context", "reference"):
        view_groups = ({split: [{**row, "label": row["reference_label"]}
                                for row in rows if row["reference"] is not None]
                        for split, rows in groups.items()} if view == "reference" else groups)
        view_train = view_groups["train"]
        if not view_train or len({row["label"] for row in view_train}) < 2:
            continue
        def matrix(rows, *, view=view):
            if view == "operation_context":
                return np.stack([np.concatenate((row["operation"], row["context"]))
                                 for row in rows])
            return np.stack([row[view] for row in rows])

        scaler = StandardScaler()
        source_features = scaler.fit_transform(matrix(view_train))
        classifier = LogisticRegression(solver="liblinear", max_iter=300, random_state=0)
        classifier.fit(source_features, [row["label"] for row in view_train])
        results[view] = {}
        for split, rows in view_groups.items():
            if not rows:
                continue
            predicted = classifier.predict(scaler.transform(matrix(rows))).tolist()
            results[view][split] = _accuracy(rows, predicted)
    baseline = {split: _accuracy(rows, [majority] * len(rows))
                for split, rows in groups.items() if rows}
    reference_counts = {
        split: dict(sorted(Counter(row["reference_label"] for row in rows
                                  if row["reference"] is not None).items()))
        for split, rows in groups.items()
    }
    body = {"schema": "aura.semantic_role_order_identifiability.v4",
            "serving_authority": False, "development_only": True,
            "feature_source_report_sha256": hashlib.sha256(source_raw).hexdigest(),
            "transducer_sha256": hashlib.sha256(args.transducer.read_bytes()).hexdigest(),
            "majority_label": majority,
            "source_labels_used_only_for_training": True,
            "validation_and_test_labels_used_only_for_grading": True,
            "reference_view_uses_gold_spans_for_upper_bound_only": True,
            "reference_mentions_sorted_by_text_position": True,
            "reference_label_counts": reference_counts,
            "baseline": baseline, "views": results}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    payload = json.dumps({**body, "receipt_sha256": digest}, sort_keys=True).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != payload:
        raise ValueError("role-order probe output already differs")
    args.output.write_bytes(payload)
    print(json.dumps({name: {split: {"population": score["population"],
                                    "correct": score["correct"]}
                             for split, score in splits.items()}
                      for name, splits in {"baseline": baseline, **results}.items()}), flush=True)


if __name__ == "__main__":
    main()
