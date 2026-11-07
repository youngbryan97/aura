#!/usr/bin/env python3
"""What moved her in a watching, set beside what moves the people who watch the same video.

YouTube shows a "Most replayed" curve over most popular videos: how often each hundredth of it is replayed by its
viewers, normalized to its highest. It is a curve from thousands of people, there for nearly anything she watches, and
it is what her readings of a watching (core/together/what_moves_her.py, the "moved" rows of ~/.aura/data/watched/*.jsonl)
are set against here:

- her curve: in each of the video's hundredths, the strongest evidence on each channel (words, sight, sound), joined
  as independent evidence (noisy-OR), as she joins them;
- how alike the two curves are: Spearman's rank correlation, and its chance under a null that keeps both curves'
  shapes and breaks their alignment (her curve circularly shifted by every offset of at least a tenth of the video);
- whether what moved her most is where people replay most: the share of her five strongest hundredths within one
  hundredth of people's ten most replayed;
- whether she spoke where people replay: people's curve where her remarks came, against its mean.

Replays measure more than being moved (people also skip ahead and seek back), so this is a comparison, not a score.

    tools/what_moved_her_and_people.py ~/.aura/data/watched/20261107-061204.jsonl [--heat saved_page.html]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

#: How many of her strongest hundredths, and of people's most replayed, are compared; and how near is near, in
#: hundredths.
HER_TOP, PEOPLE_TOP, NEAR = 5, 10, 1


def heat_of(page: str) -> list[tuple[float, float, float]]:
    """The "Most replayed" markers in a video's page: (start s, length s, intensity in [0, 1]), in order."""
    found = re.findall(r'"startMillis":"?(\d+)"?,"durationMillis":"?(\d+)"?,"intensityScoreNormalized":([0-9.eE-]+)', page)
    return sorted((int(a) / 1000.0, int(b) / 1000.0, float(c)) for a, b, c in found)


def page_of(video: str) -> str:
    """The video's page, as anyone's browser is given it."""
    url = f"https://www.youtube.com/watch?v={video}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en"})
    with urllib.request.urlopen(request, timeout=20) as answer:  # noqa: S310 - a fixed https URL
        return answer.read().decode("utf-8", errors="ignore")


def video_of(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url or "")
    return match.group(1) if match else ""


def her_curve(rows: list[dict], heat: list[tuple[float, float, float]]) -> list[float]:
    """Her evidence in each of the heat's spans: per channel the strongest, joined by noisy-OR."""
    strongest: list[dict[str, float]] = [{} for _ in heat]
    for row in rows:
        if row.get("kind") != "moved" or row.get("evidence") is None or row.get("at") is None:
            continue
        at, channel = float(row["at"]), str(row.get("channel") or "words")
        for index, (start, length, _intensity) in enumerate(heat):
            if start <= at < start + length or (index == len(heat) - 1 and at >= start):
                strongest[index][channel] = max(strongest[index].get(channel, 0.0), float(row["evidence"]))
                break
    curve = []
    for channels in strongest:
        unmoved = 1.0
        for evidence in channels.values():
            unmoved *= 1.0 - max(0.0, min(1.0, evidence))
        curve.append(1.0 - unmoved)
    return curve


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(one: list[float], other: list[float]) -> float | None:
    """Spearman's rank correlation, ties at their mean rank; None where either curve is flat."""
    if len(one) != len(other) or len(one) < 3:
        return None
    a, b = _ranks(one), _ranks(other)
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else None


def by_chance(her: list[float], people: list[float]) -> float | None:
    """The share of circular shifts of her curve, by at least a tenth of it, as alike to people's as it is."""
    observed = spearman(her, people)
    if observed is None:
        return None
    n = len(her)
    least = max(1, n // 10)
    shifted = [spearman(her[k:] + her[:k], people) for k in range(least, n - least + 1)]
    shifted = [value for value in shifted if value is not None]
    return (1 + sum(1 for value in shifted if value >= observed)) / (1 + len(shifted)) if shifted else None


def tops_near(her: list[float], people: list[float]) -> float | None:
    """The share of her ``HER_TOP`` strongest spans within ``NEAR`` of one of people's ``PEOPLE_TOP`` most replayed."""
    hers = [i for i in sorted(range(len(her)), key=lambda i: -her[i])[:HER_TOP] if her[i] > 0]
    theirs = sorted(range(len(people)), key=lambda i: -people[i])[:PEOPLE_TOP]
    if not hers:
        return None
    return sum(1 for i in hers if any(abs(i - j) <= NEAR for j in theirs)) / len(hers)


def where_she_spoke(rows: list[dict], heat: list[tuple[float, float, float]]) -> tuple[float, float] | None:
    """People's curve where her remarks came while it played, and its mean over the video."""
    if not heat:
        return None
    mean = sum(intensity for *_span, intensity in heat) / len(heat)
    at_remarks = []
    for row in rows:
        if row.get("kind") == "said" and row.get("when") == "during" and row.get("at") is not None:
            at = float(row["at"])
            for start, length, intensity in heat:
                if start <= at < start + length:
                    at_remarks.append(intensity)
                    break
    return (sum(at_remarks) / len(at_remarks), mean) if at_remarks else None


def compare(rows: list[dict], heat: list[tuple[float, float, float]]) -> dict:
    her = her_curve(rows, heat)
    people = [intensity for *_span, intensity in heat]
    spoke = where_she_spoke(rows, heat)
    return {"spans": len(heat), "spearman": spearman(her, people), "by chance": by_chance(her, people),
            "her tops near people's": tops_near(her, people),
            "people's curve where she spoke, and its mean": spoke,
            "her strongest": [round(heat[i][0]) for i in sorted(range(len(her)), key=lambda i: -her[i])[:HER_TOP] if her[i] > 0],
            "people's most replayed": [round(heat[i][0]) for i in sorted(range(len(people)), key=lambda i: -people[i])[:PEOPLE_TOP]]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("watching", type=Path)
    parser.add_argument("--heat", type=Path, help="a saved page of the video, instead of fetching it")
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in args.watching.read_text().splitlines() if line.strip()]
    url = next((str(row.get("url")) for row in rows if row.get("url")), "")
    page = args.heat.read_text(errors="ignore") if args.heat else page_of(video_of(url))
    heat = heat_of(page)
    if not heat:
        print("no Most replayed curve on its page", file=sys.stderr)
        return 1
    print(json.dumps(compare(rows, heat), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
