from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from core.social.dialogue_cognition import DialogueCognitionEngine


async def _run(root: Path, pattern: str) -> int:
    engine = DialogueCognitionEngine()
    loaded = 0

    for source_id in engine.default_source_ids():
        source_dir = root / source_id
        if not source_dir.exists() or not source_dir.is_dir():
            continue
        profile = await engine.ingest_source_transcript_directory(
            source_id,
            source_dir,
            pattern=pattern,
        )
        loaded += 1
        print(
            f"[dialogue] loaded {source_id}: turns={profile.interactions_analyzed} "
            f"answer_first={profile.answer_first_preference:.2f} "
            f"attunement={profile.attunement_preference:.2f}"
        )

    engine.save()
    return loaded


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest local dialogue corpora into Aura's dialogue cognition layer.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("research") / "dialogue_corpora",
        help="Root directory containing one folder per source archetype.",
    )
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Glob pattern for transcript files inside each source directory.",
    )
    args = parser.parse_args()

    loaded = asyncio.run(_run(args.root, args.pattern))
    if loaded == 0:
        print(f"[dialogue] no source corpora found under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
