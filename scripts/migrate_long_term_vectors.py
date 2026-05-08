#!/usr/bin/env python3
"""Migrate legacy JSON vector memories into local SQLite BLOB storage."""
from __future__ import annotations

import argparse
from pathlib import Path

from core.memory.sqlite_vector_store import SQLiteVectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="memory_store/long_term.json",
        help="Legacy JSON file containing records with a vector field.",
    )
    parser.add_argument(
        "--dest",
        default="data/memory/long_term_vectors.sqlite3",
        help="Destination SQLite database path.",
    )
    parser.add_argument(
        "--collection",
        default="long_term",
        help="Vector collection name.",
    )
    parser.add_argument(
        "--remove-source",
        action="store_true",
        help="Delete the legacy JSON file after a successful migration.",
    )
    args = parser.parse_args()

    store = SQLiteVectorStore(Path(args.dest), collection_name=args.collection)
    count = store.migrate_legacy_json(
        Path(args.source),
        collection=args.collection,
        remove_source=args.remove_source,
    )
    print(f"migrated={count} dest={args.dest} collection={args.collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
