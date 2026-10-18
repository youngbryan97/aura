#!/usr/bin/env python3
"""Train her turn reader: what a message expresses, read from her own embedding of it.

    python tools/train_turn_reader.py --output ~/.aura/rlc-evidence/turn-reader-20261018
    python tools/train_turn_reader.py --output DIR --install     # also put it where her runtime reads it

GoEmotions (Demszky et al. 2020; google-research-datasets/go_emotions, the
"simplified" configuration): 58k Reddit comments, each labelled by people with
any of 27 emotions or neutral, split train / validation / test by its authors.
Each comment is embedded the way her runtime embeds a message it reads
(core/memory/vector_memory_engine.EmbeddingEngine.embed_query with the
"feeling_in_a_message" instruction), so the reader sees at run time exactly the
vectors it was trained on.

One logistic head per label over the unit-length vector. Each head's
regularisation is the one, of the grid, that scores best on the validation
split; the test split is scored once, at the end, and its numbers are the
report. Nothing is installed unless asked, and nothing at all is written when
the test macro F1 is no better than the frequency baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: Regularisation strengths tried per label; the validation split chooses among them.
C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


def _load_goemotions():
    from datasets import load_dataset

    return load_dataset("google-research-datasets/go_emotions", "simplified")


def _embed_all(texts: list[str], task: str, cache: Path):
    import numpy as np

    if cache.exists():
        stored = np.load(cache)
        if stored.shape[0] == len(texts):
            return stored
    from core.memory.vector_memory_engine import EmbeddingEngine

    engine = EmbeddingEngine()
    engine.embed_query("a probe", task=task)
    if getattr(engine, "_model", None) is None:
        raise SystemExit("her embedding model did not load (the model lane is held elsewhere); nothing trained")
    rows = []
    started = time.monotonic()
    for index, text in enumerate(texts):
        rows.append(np.asarray(engine.embed_query(text, task=task), dtype=np.float32))
        if index and index % 2000 == 0:
            rate = index / (time.monotonic() - started)
            print(json.dumps({"embedded": index, "of": len(texts), "per_second": round(rate, 1)}), flush=True)
    matrix = np.stack(rows)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, matrix)
    return matrix


def _multi_hot(label_lists, n: int):
    import numpy as np

    y = np.zeros((len(label_lists), n), dtype=np.int8)
    for row, labels in enumerate(label_lists):
        for label in labels:
            y[row, int(label)] = 1
    return y


def _f1(y_true, y_pred) -> float:
    tp = int((y_true & y_pred).sum())
    fp = int(((1 - y_true) & y_pred).sum())
    fn = int((y_true & (1 - y_pred)).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def main() -> int:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from core.interiority.what_a_turn_did import LABELS, TASK, TurnReader, TurnReading, head_path

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    out = args.output.expanduser()
    out.mkdir(parents=True, exist_ok=True)

    data = _load_goemotions()
    names = data["train"].features["labels"].feature.names
    if tuple(names) != LABELS:
        raise SystemExit(f"the dataset's labels are not the reader's: {names}")
    splits = {}
    for split in ("train", "validation", "test"):
        texts = list(data[split]["text"])
        splits[split] = (_embed_all(texts, TASK, out / f"embedded_{split}.npy"),
                         _multi_hot(data[split]["labels"], len(LABELS)), texts)

    x_train, y_train, _ = splits["train"]
    x_val, y_val, _ = splits["validation"]
    x_test, y_test, test_texts = splits["test"]
    weights = np.zeros((len(LABELS), x_train.shape[1]), dtype=np.float32)
    bias = np.zeros(len(LABELS), dtype=np.float32)
    chosen: dict[str, float] = {}
    for k, label in enumerate(LABELS):
        best = None
        for c in C_GRID:
            model = LogisticRegression(C=c, max_iter=2000)
            model.fit(x_train, y_train[:, k])
            score = _f1(y_val[:, k], (model.predict_proba(x_val)[:, 1] >= 0.5).astype(np.int8))
            if best is None or score > best[0]:
                best = (score, c, model)
        chosen[label] = best[1]
        weights[k] = best[2].coef_[0]
        bias[k] = best[2].intercept_[0]

    probabilities = 1.0 / (1.0 + np.exp(-(x_test @ weights.T + bias)))
    predicted = (probabilities >= 0.5).astype(np.int8)
    per_label = {label: round(_f1(y_test[:, k], predicted[:, k]), 4) for k, label in enumerate(LABELS)}
    macro = float(np.mean(list(per_label.values())))
    # The baseline: every label predicted at its training frequency's majority (all zeros but the most common).
    common = int(y_train.sum(axis=0).argmax())
    baseline_pred = np.zeros_like(y_test)
    baseline_pred[:, common] = 1
    baseline_macro = float(np.mean([_f1(y_test[:, k], baseline_pred[:, k]) for k in range(len(LABELS))]))

    # Valence as the reader computes it, against the valence of the labels people gave.
    agree = total = 0
    for row in range(len(test_texts)):
        truth = TurnReading(feelings={LABELS[k]: float(y_test[row, k]) for k in range(len(LABELS))}).valence
        read = TurnReading(feelings={LABELS[k]: float(probabilities[row, k]) for k in range(len(LABELS))}).valence
        if truth is None or read is None or truth == 0.0:
            continue
        total += 1
        agree += int((truth > 0) == (read > 0))
    report = {
        "dataset": "google-research-datasets/go_emotions (simplified)",
        "embedding": {"task": TASK, "dimension": int(x_train.shape[1])},
        "sizes": {split: int(splits[split][0].shape[0]) for split in splits},
        "chosen_c": chosen,
        "test_macro_f1": round(macro, 4),
        "test_macro_f1_baseline_most_common_label": round(baseline_macro, 4),
        "test_f1_per_label": per_label,
        "test_valence_sign_agreement": {"agree": agree, "of": total,
                                        "rate": round(agree / total, 4) if total else None},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("test_macro_f1", "test_macro_f1_baseline_most_common_label",
                                             "test_valence_sign_agreement")}, indent=2))
    if macro <= baseline_macro:
        print("No better than the baseline; no reader written.")
        return 2
    reader = TurnReader(weights, bias, LABELS, embedding_model="Qwen/Qwen3-Embedding-0.6B", report=report)
    reader.save(out / "turn_reader_head.npz")
    if args.install:
        installed = reader.save(head_path())
        print(f"installed: {installed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
