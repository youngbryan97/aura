#!/usr/bin/env python3
"""Build her recall store: what her offline Wikipedia says about what people ask her, in her own hidden states.

    python tools/build_her_recall_store.py --dry-run          # which passages, no model
    python tools/build_her_recall_store.py                    # her app must be down: this loads her 27B

The foreground recall path (core/brain/nonparametric_worker.py) is built and
wired into her worker: a tap on the hidden state each decode step already
computes, and a processor that mixes in the next tokens the nearest stored
states predicted. It refuses to steer until the store holds 50,000 entries,
because a 30-entry store of arithmetic steered two live turns into garbled
words on 29 July, and the condition the floor stands for is coverage of the
queries she meets. Her store on 17 October held about 550 entries, all
arithmetic, under an identity that names no checkpoint.

So the store is built from the questions she has been asked. Each distinct
message a person sent her (newest first) is put to her paragraph index of
the offline Wikipedia, and the passages it finds are read by her own model,
title as context and paragraph as continuation, one entry per paragraph
token, until the store reaches the floor. Each entry is under the shared
principal and names its source and paragraph, so the whole source can be
revoked in one call. The passages are sourced, not checked, and are marked
unverified. The old store is kept beside the new one.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def people_asked(conversations: Path) -> list[str]:
    """Every distinct message a person sent her, newest first."""
    with sqlite3.connect(f"file:{conversations}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT content FROM turns WHERE role = 'user' ORDER BY created_at DESC").fetchall()
    seen: set[str] = set()
    out: list[str] = []
    for (content,) in rows:
        text = str(content or "").strip()
        if len(text.split()) >= 3 and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def passages_in_rounds(questions: list[str], search):
    """Her index's passages for the questions: every question's best, then every question's second, and so on.

    Rounds rather than a fixed number per question, so how deep it reaches is set by
    how many entries the store needs, and every question is covered before any is
    covered twice.
    """
    found: dict[str, list] = {}
    seen: set[tuple[int, int]] = set()
    depth = 0
    while True:
        depth += 1
        any_left = False
        for question in questions:
            results = found.get(question)
            if results is None or len(results) < depth:
                results = found[question] = list(search(question, limit=depth))
            if len(results) < depth:
                continue
            any_left = True
            passage = results[depth - 1]
            key = (int(passage.doc_id), int(passage.position))
            if key in seen or not str(passage.paragraph or "").strip():
                continue
            seen.add(key)
            yield {"title": str(passage.title), "paragraph": str(passage.paragraph),
                   "doc_id": int(passage.doc_id), "position": int(passage.position), "asked": question[:120]}
        if not any_left:
            return


def main() -> int:
    from core.brain.nonparametric_worker import _MIN_ENTRIES_TO_STEER_GENERATION
    from core.knowledge.passage_index import PassageIndex

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversations", type=Path, default=Path("~/.aura/data/conversations.db").expanduser())
    parser.add_argument("--plan", type=Path, default=REPO / "artifacts/migration/27b/recovery/steering_plan.json")
    parser.add_argument("--target-entries", type=int, default=_MIN_ENTRIES_TO_STEER_GENERATION)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    questions = people_asked(args.conversations)
    index = PassageIndex()
    rounds = passages_in_rounds(questions, lambda q, limit: index.search(q, limit=limit))
    summary = {"questions": len(questions), "target_entries": args.target_entries}
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        # Paragraph words are fewer than the tokens her model reads them as, so this
        # under-counts the entries the passages would give: enough words is enough tokens.
        words, taken = 0, []
        for passage in rounds:
            taken.append(passage)
            words += len(passage["paragraph"].split())
            if words >= args.target_entries:
                break
        print(json.dumps({"passages": len(taken), "paragraph_words": words,
                          "first": [p["title"] for p in taken[:8]]}, indent=2))
        return 0

    import shutil

    from mlx_lm import load

    from core.brain.nonparametric_binding import ingest_provenance
    from core.brain.nonparametric_generation import MLXEncoder
    from core.brain.nonparametric_ingest import NonParametricIngestor
    from core.brain.nonparametric_memory import SHARED_MEMORY_PRINCIPAL, get_nonparametric_memory
    from core.runtime.state_ownership import state_root

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    model, tokenizer = load(str(plan["model_path"]))
    encoder = MLXEncoder(model, tokenizer)
    runtime = state_root() / "data/runtime"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for old in runtime.glob(f"nonparametric_memory_{encoder.dim}*"):
        if ".before-her-recall-store-" not in old.name:
            shutil.copy2(old, old.with_name(old.name + f".before-her-recall-store-{stamp}"))
    memory = get_nonparametric_memory(encoder.dim)
    if memory is None:
        print("no datastore for this width", file=sys.stderr)
        return 1
    # A store that cannot hold the target never reaches it; on 18 October one capped at 4,096 read
    # 12,650 passages over two hours before anyone looked.
    capacity = int(getattr(memory, "_max", 0) or 0)
    if capacity < args.target_entries:
        print(f"the store holds at most {capacity:,} entries and the target is {args.target_entries:,}",
              file=sys.stderr)
        return 1
    corpus_date = getattr(index, "corpus_date", "") or "2026-07-03"
    added = 0
    started = time.monotonic()
    stalled = 0
    for number, passage in enumerate(rounds, 1):
        if len(memory) >= args.target_entries:
            break
        before = len(memory)
        ingestor = NonParametricIngestor(memory, provenance=ingest_provenance(
            principal=SHARED_MEMORY_PRINCIPAL, source_id=f"offline_wikipedia:{corpus_date}",
            trust="unverified", evidence_id=f"{passage['doc_id']}:{passage['position']}"))
        added += ingestor.ingest_sequence(passage["title"], passage["paragraph"], encoder)
        stalled = stalled + 1 if len(memory) <= before else 0
        if stalled >= 100:
            print(f"the store stopped growing at {len(memory):,} entries; stopping", file=sys.stderr)
            break
        if number % 50 == 0:
            print(json.dumps({"passages": number, "entries": len(memory),
                              "seconds": round(time.monotonic() - started)}), flush=True)
    memory.persist()
    report = {**summary, "entries": len(memory), "added": added, "seconds": round(time.monotonic() - started),
              "reached_floor": len(memory) >= args.target_entries, "model_path": plan["model_path"]}
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2))
    return 0 if report["reached_floor"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
