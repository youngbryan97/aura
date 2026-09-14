"""How far below a note the singers on the records start, and how long they take to arrive.

The voice lane bends into a line from underneath (core/voice/duplex/pitch.py).
How deep and how long are taken from here, from the records, so neither is a
number chosen for the voice.

For every voiced segment long enough to have a settled pitch, the onset is its
first 60 ms and the settled note is the median pitch after that. The bend is
the settled note minus the onset pitch, in cents. The glide is the time until
the pitch comes within 50 cents of the settled note, which is the point where
the nearest semitone is the note itself. It is timed only for onsets that start
more than that far under: an onset already within half a semitone has no glide
to time, and counting it as zero measures the onsets rather than the bends.
Segments whose onset is more than 300 cents from the settled note are dropped,
because at that distance the tracker has caught the previous note or an octave
error rather than a bend.

    /Users/bryan/.aura/live-source/.venv/bin/python tools/measure_onset_bend.py \\
        --source ~/Downloads --out artifacts/soul/onset_bend.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SAMPLE_RATE = 22_050
HOP = 256
ONSET_MS = 60.0
ARRIVED_CENTS = 50.0
TRACKER_LIMIT_CENTS = 300.0
LISTEN_SECONDS = 150.0


def _segments(f0):
    import numpy as np

    i, n = 0, len(f0)
    while i < n:
        if not np.isfinite(f0[i]):
            i += 1
            continue
        j = i
        while j < n and np.isfinite(f0[j]):
            j += 1
        yield f0[i:j]
        i = j


def measure(path: Path) -> dict:
    import librosa
    import numpy as np

    y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, duration=LISTEN_SECONDS)
    harmonic = librosa.effects.harmonic(y, margin=3.0)
    f0, _voiced, _ = librosa.pyin(
        harmonic, fmin=80, fmax=1000, sr=sr, frame_length=2048, hop_length=HOP, fill_na=np.nan
    )
    frame_ms = HOP / sr * 1000.0
    onset_frames = max(1, int(round(ONSET_MS / frame_ms)))
    bends: list[float] = []
    glides: list[float] = []
    for segment in _segments(f0):
        if len(segment) < onset_frames * 4:
            continue
        settled = float(np.median(segment[onset_frames:]))
        onset = float(np.median(segment[:onset_frames]))
        cents = 1200.0 * float(np.log2(settled / onset))
        if abs(cents) > TRACKER_LIMIT_CENTS:
            continue
        bends.append(cents)
        if cents > ARRIVED_CENTS:
            arrived = next(
                (k for k, v in enumerate(segment) if abs(1200.0 * np.log2(settled / v)) <= ARRIVED_CENTS),
                None,
            )
            if arrived is not None:
                glides.append(arrived * frame_ms)
    below = [b for b in bends if b > 0.0]
    return {
        "onsets": len(bends),
        "median_bend_cents": round(statistics.median(bends), 1) if bends else None,
        "share_from_below": round(len(below) / len(bends), 3) if bends else None,
        "median_from_below_cents": round(statistics.median(below), 1) if below else None,
        "median_glide_ms": round(statistics.median(glides), 1) if glides else None,
    }


def summarise(rows: dict[str, dict]) -> dict:
    def median_of(key: str, places: int) -> float | None:
        values = [row[key] for row in rows.values() if row.get(key) is not None]
        return round(statistics.median(values), places) if values else None

    return {
        "records": len(rows),
        "median_from_below_cents": median_of("median_from_below_cents", 1),
        "median_glide_ms": median_of("median_glide_ms", 1),
        "median_share_from_below": median_of("share_from_below", 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "soul" / "onset_bend.json")
    args = parser.parse_args()

    from core.runtime.atomic_writer import atomic_write_text

    names = sorted({p.name for p in args.source.glob("*.mp3")} | {p.name for p in args.source.glob("*.mp4")})
    # A browser saves a second download of the same file as "name (1)".
    names = [name for name in names if "(1)" not in name]
    rows: dict[str, dict] = {}
    failed: dict[str, str] = {}
    for name in names:
        try:
            rows[name] = measure(args.source / name)
        except (OSError, ValueError, RuntimeError) as exc:
            failed[name] = f"{type(exc).__name__}: {exc}"
            print(f"[fail] {name}: {failed[name]}", flush=True)
            continue
        print("[done]", name, rows[name], flush=True)
    summary = summarise(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        args.out,
        json.dumps(
            {
                "summary": summary,
                "parameters": {
                    "sample_rate": SAMPLE_RATE,
                    "hop": HOP,
                    "onset_ms": ONSET_MS,
                    "arrived_cents": ARRIVED_CENTS,
                    "tracker_limit_cents": TRACKER_LIMIT_CENTS,
                    "listen_seconds": LISTEN_SECONDS,
                },
                "records": rows,
                "failed": failed,
            },
            indent=1,
        ),
    )
    print("SUMMARY", summary, flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
