#!/usr/bin/env python3
"""Drive her over the reading contrasts a meta-analysis of 163 studies used.

Turker et al. (2025) pooled 3,031 subjects and asked what a brain recruits when
it reads, at four levels — letters, words, sentences, text — and under two tasks
on identical text: read it, or judge it. This drives Aura over the same
contrasts through the machinery she actually reads with, so the same questions
can be asked of her recording.

The conditions, and why each is the one it is:

    letter      one character at a time, through her tokeniser and her
                text-shaping surface
    word        identifiers she already holds — names from her own source
    pseudoword  well-formed names she has never seen, built by recombining the
                morphemes of the real ones so they are pronounceable and legal
                and mean nothing. This is the paper's word/pseudoword contrast,
                and it is the one that separates the two routes
    sentence    one clause of her own documentation
    text        a paragraph of it
    judge_word  the same identifiers as `word`, under a lexical decision —
                is this a name this system holds? — rather than under reading

`judge_word` against `word` is the task contrast on identical input, and it is
the comparison the paper found moved the network most. Everything else varies
the stimulus at a fixed task.

Nothing here asks a model anything. What is being measured is which of her own
cells fire while text of each kind passes through her text machinery, so the
answer is about her code rather than about a completion.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("AURA_TESTING", "1")

#: A clause and a paragraph of her own writing. Her documentation rather than a
#: corpus, because what is being measured is her reading her own world.
SENTENCE = (
    "A function is a cell, a module is a neuropil, and one call site is one synapse."
)
TEXT = (
    "The structural connectome says what can happen. The interesting object is what "
    "does: the influence one cell has on another under a given condition. Same "
    "anatomy, different active circuits, which is how biological brains work and is "
    "what the recorder's condition field was for. An effective connectome averaged "
    "over every condition is the average of circuits that are never active together, "
    "and it describes none of them."
)


def _real_names(limit: int = 240) -> list[str]:
    """Identifiers her own source actually holds."""
    names: set[str] = set()
    for path in sorted((REPO / "core" / "connectome").glob("*.py")):
        for match in re.finditer(r"^\s*def ([a-z_][a-z0-9_]{3,})", path.read_text(), re.M):
            names.add(match.group(1))
        if len(names) >= limit:
            break
    return sorted(names)[:limit]


def _pseudonames(real: list[str], seed: int = 11) -> list[str]:
    """Well-formed names she has never seen.

    Built by recombining the morphemes of the real ones, so they obey the same
    spelling rules and mean nothing — which is what a pseudoword is. Anything
    that collides with a real name is dropped rather than kept, because a
    pseudoword that turns out to be a word is the contrast collapsing.
    """
    rng = random.Random(seed)
    parts = sorted({piece for name in real for piece in name.split("_") if len(piece) > 2})
    if len(parts) < 2:
        return []
    known = set(real)
    made: list[str] = []
    while len(made) < len(real) and len(made) < 400:
        candidate = "_".join(rng.sample(parts, k=rng.choice((2, 3))))
        if candidate not in known and candidate not in made:
            made.append(candidate)
    return made


#: The packages text passes through on its way in. Not a list of functions —
#: naming four of them by hand is how the first run of this recorded eight
#: cells and called it her reading pathway.
_READING_PACKAGES: tuple[str, ...] = (
    "core.language",
    "core.conversation",
    "core.intent",
    "core.brain.imagination_text",
    "core.brain.compression",
    "core.cognition",
    "core.perception",
)

#: Names that read a string rather than change something with it. A probe that
#: writes is not a reading probe, and one that takes a filename is not either.
_REFUSED = (
    "write", "save", "delete", "remove", "install", "promote", "record", "emit",
    "publish", "send", "run_", "execute", "apply", "set_", "reset", "clear",
    "open_", "load_", "fetch", "post", "put_", "spawn", "kill", "shutdown",
)

_PROBES: list[Any] = []


def _reading_probes() -> list[Any]:
    """Every function in her text packages that takes one string and reads it.

    Discovered rather than listed. The first version of this called four
    functions somebody had picked, three of which did not exist, and recorded
    eight cells — a measurement of the probe.
    """
    global _PROBES
    if _PROBES:
        return _PROBES
    import importlib
    import inspect
    import pkgutil

    modules: list[str] = []
    for package in _READING_PACKAGES:
        try:
            found = importlib.import_module(package)
        except BaseException:  # noqa: BLE001 - a package that will not import is skipped
            continue
        modules.append(package)
        path = getattr(found, "__path__", None)
        if path is None:
            continue
        for entry in pkgutil.iter_modules(path):
            if not entry.name.startswith("_"):
                modules.append(f"{package}.{entry.name}")

    probes: list[Any] = []
    for name in modules:
        try:
            module = importlib.import_module(name)
        except BaseException:  # noqa: BLE001
            continue
        for attribute in sorted(dir(module)):
            if attribute.startswith("_") or attribute.startswith(_REFUSED):
                continue
            value = getattr(module, attribute, None)
            if not callable(value) or inspect.isclass(value):
                continue
            if getattr(value, "__module__", None) != name:
                continue
            if inspect.iscoroutinefunction(value) or inspect.isasyncgenfunction(value):
                continue
            try:
                signature = inspect.signature(value)
            except (TypeError, ValueError):
                continue
            required = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.default is inspect.Parameter.empty
                and parameter.kind
                in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            ]
            if len(required) != 1:
                continue
            annotation = required[0].annotation
            takes_text = annotation in (str, "str", inspect.Parameter.empty) or (
                isinstance(annotation, str) and "str" in annotation
            )
            if not takes_text:
                continue
            probes.append(value)
    _PROBES = probes
    return probes


def _read_like(text: str) -> None:
    """Put one string through every surface that reads a string."""
    for probe in _reading_probes():
        try:
            probe(text)
        except BaseException:  # noqa: BLE001 - a probe must never end the run
            continue


def _judge(text: str, known: set[str]) -> bool:
    """A lexical decision: is this a name this system holds?

    The paper's contrast is between reading a string and JUDGING one, on
    identical input. So this reads it the same way and then does the extra
    thing — settles a question about it and reports the answer — which is the
    only difference between the two conditions.
    """
    from core.brain.imagination_text import imagination_subject

    _read_like(text)
    subject = imagination_subject(text)
    verdict = subject.strip() in known
    try:
        from core.cognition.how_sure_she_is import enough_families_to_say

        enough_families_to_say(at_least=0.5 if verdict else 0.4)
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=12.0, help="seconds per condition")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--frame-seconds", type=float, default=0.002)
    parser.add_argument(
        "--out", type=Path, default=REPO / "artifacts" / "connectome" / "reading"
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(args.out / "logs"))

    from core.connectome.activity import ActivityRecorder, RecorderConfig

    real = _real_names()
    fake = _pseudonames(real)
    known = set(real)
    letters = [character for character in "abcdefghijklmnopqrstuvwxyz_0123456789"]
    conditions: dict[str, Any] = {
        "letter": letters,
        "word": real,
        "pseudoword": fake,
        "sentence": [SENTENCE],
        "text": [TEXT],
        "judge_word": real,
    }
    print(
        json.dumps(
            {name: len(items) for name, items in conditions.items()}, indent=1
        ),
        flush=True,
    )
    if not fake:
        print("no pseudowords could be built; the dual-route contrast is unavailable")

    recorder = ActivityRecorder(
        REPO,
        RecorderConfig(
            frame_seconds=args.frame_seconds,
            capture_edges=True,
            max_wall_seconds=args.budget * len(conditions) * args.rounds + 300,
            max_frames=32_768,
        ),
    )
    log: list[dict[str, Any]] = []
    recorder.start("letter")
    for round_index in range(args.rounds):
        for condition, items in conditions.items():
            if not items:
                continue
            recorder.set_condition(condition)
            started = time.monotonic()
            deadline = started + args.budget
            passes = 0
            while time.monotonic() < deadline:
                for item in items:
                    if time.monotonic() >= deadline:
                        break
                    if condition == "judge_word":
                        _judge(item, known)
                    else:
                        _read_like(item)
                passes += 1
            entry = {
                "round": round_index,
                "condition": condition,
                "items": len(items),
                "passes": passes,
                "seconds": round(time.monotonic() - started, 1),
            }
            log.append(entry)
            print(json.dumps(entry), flush=True)
    trace = recorder.stop()

    import numpy as np

    np.savez_compressed(args.out / "activity.npz", spikes=trace.matrix())
    (args.out / "activity_manifest.json").write_text(
        json.dumps(
            {
                "uids": list(trace.uids),
                "conditions": list(trace.conditions),
                "frame_seconds": trace.frame_seconds,
                "summary": trace.summary(),
                "log": log,
                "pseudowords": fake[:20],
            },
            indent=2,
        )
    )
    (args.out / "observed_edges.json").write_text(
        json.dumps(
            {
                "counts": {
                    f"{pre}>{post}": count
                    for (pre, post), count in recorder.observed.counts.items()
                },
                "summary": recorder.observed.summary(),
            },
            indent=2,
        )
    )
    print(json.dumps({"trace": trace.summary()}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
