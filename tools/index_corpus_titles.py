#!/usr/bin/env python3
"""Build the index her library finds a document by its title with, where it has none.

A library made before the index was kept with it (core/knowledge/local_corpus.py, TITLE_INDEX) reads every title to
find one: 9.9 s for "Quiet hours" over 6,589,097 documents on 28 November 2026. What an exchange is about is read
by the things it names that her library knows (core/conation/what_an_exchange_is_about.py), several titles an
exchange, so without the index she reads no subjects at all.

The library is in WAL mode, so her reads go on while it is built. Her writes to it wait on the write lock.

Usage:
    python tools/index_corpus_titles.py [--db PATH]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str] | None = None) -> int:
    from core.knowledge.local_corpus import LocalCorpusStore, default_corpus_db_path

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=None, help="the library (default: hers)")
    args = parser.parse_args(argv)
    store = LocalCorpusStore(args.db or default_corpus_db_path())
    if not store.db_path.exists():
        print(f"no library at {store.db_path}")
        return 2
    if store.has_title_index():
        print(f"{store.db_path} has its title index")
        return 0
    began = time.monotonic()
    built = store.index_titles()
    print(f"{store.db_path}: title index {'built' if built else 'NOT built'} in {time.monotonic() - began:.1f} s")
    return 0 if built else 1


if __name__ == "__main__":
    raise SystemExit(main())
