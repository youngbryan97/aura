#!/usr/bin/env python3
"""Build (or resume building) the index of every paragraph in her corpus (core/knowledge/passage_index.py).

Reads corpus.db read-only and writes passages.db beside it. Started again, it
resumes after the last article it recorded.

Usage:
    build_passage_index.py [--corpus DB] [--output DB] [--until DOC_ID]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from core.knowledge.local_corpus import default_corpus_db_path
    from core.knowledge.passage_index import build, default_passage_db_path

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--until", type=int, default=None)
    args = parser.parse_args()
    corpus = (args.corpus or default_corpus_db_path()).expanduser()
    output = (args.output or default_passage_db_path()).expanduser()
    if not corpus.exists():
        raise SystemExit(f"no corpus at {corpus}")
    summary = build(corpus, output, until=args.until, on_progress=lambda p: print(json.dumps(p), flush=True))
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
