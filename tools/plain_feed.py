#!/usr/bin/env python3
"""How much of her neural feed has a face in plain words, and which lines most need one.

Every line in the feed is a log record, and every record comes from a call in her code with a format string and a
logger. The plain sentence for a line lives in `config/plain_feed.json`, keyed by that logger and format string, and her
server puts it on the card's face (core/observability/plain_feed.py). This finds every such call in her code, matches a
captured feed against them, and says what share of the feed's events has a plain face, weighted by how often each line
occurs.

    python tools/plain_feed.py                         # the share, against tests/fixtures/neural_feed_today.json
    python tools/plain_feed.py --missing 40            # the lines most worth a sentence, most frequent first
    python tools/plain_feed.py --capture LOG --since 2026-11-16T19:31:40 --until 2026-11-16T20:41:56 --out FIXTURE
    python tools/plain_feed.py --check                 # fails below config/plain_feed_baseline.json, or on a stale entry
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "config" / "plain_feed.json"
BASELINE = ROOT / "config" / "plain_feed_baseline.json"
FIXTURE = ROOT / "tests" / "fixtures" / "neural_feed_today.json"
SCANNED = ("core", "interface", "skills", "llm", "executors", "security", "aura_main.py")
LEVELS = {"info", "warning", "warn", "error", "exception", "critical"}
_PERCENT = re.compile(r"%(?:\([^)]*\))?[-#0 +]*(?:\d+|\*)?(?:\.(?:\d+|\*))?[sdrifxXeEgGc]")

sys.path.insert(0, str(ROOT))
from core.observability.plain_feed import pattern_of  # noqa: E402


def _skeleton(node: ast.AST) -> tuple[str, str] | None:
    """The format string of a log call's first argument and its kind: "%" for a literal, "f" for an f-string."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, "%"
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value).replace("{", "{{").replace("}", "}}"))
            else:
                parts.append("{}")
        return "".join(parts), "f"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _skeleton(node.left), _skeleton(node.right)
        if left and right and left[1] == right[1] == "%":
            return left[0] + right[0], "%"
    return None


def _loggers_of(tree: ast.Module, module: str) -> dict[str, str]:
    """Module-level names bound to a logger, and the logger each names."""
    names: dict[str, str] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
            continue
        func = node.value.func
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if called not in {"getLogger", "get_logger"} or not node.value.args:
            continue
        arg = node.value.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            name = arg.value
        elif isinstance(arg, ast.Name) and arg.id == "__name__":
            name = module
        else:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names[target.id] = name
    return names


def _gives_its_own_sentence(call: ast.Call) -> bool:
    """Whether a log call hands its plain sentence with it: ``extra={"plain": ...}``."""
    return any(keyword.arg == "extra" and isinstance(keyword.value, ast.Dict)
               and any(isinstance(key, ast.Constant) and key.value == "plain" for key in keyword.value.keys)
               for keyword in call.keywords)


def _is_a_logger(owner: ast.AST) -> bool:
    name = owner.id if isinstance(owner, ast.Name) else owner.attr if isinstance(owner, ast.Attribute) else ""
    return "log" in name.lower()


def call_sites() -> list[dict[str, str]]:
    """Every log call at info or above in her code whose format string can be read from the source.

    The logger a call writes under is not always readable from its module: a module may log with a logger it imported
    from another (core/brain/llm/mlx_client_waiting.py writes as LLM.MLX). Which logger a line came from is read from
    the feed itself; what is known here is kept as a hint.
    """
    found = []
    for top in SCANNED:
        base = ROOT / top
        paths = [base] if base.is_file() else sorted(base.rglob("*.py"))
        for path in paths:
            rel = path.relative_to(ROOT).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            module = rel[:-3].replace("/", ".").removesuffix(".__init__")
            loggers = _loggers_of(tree, module)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args):
                    continue
                owner = node.func.value
                logs_at = node.func.attr == "log" and len(node.args) >= 2  # logger.log(level, "...", ...)
                if (node.func.attr not in LEVELS and not logs_at) or not _is_a_logger(owner):
                    continue
                said = _skeleton(node.args[1] if logs_at else node.args[0])
                if said is None:
                    continue
                hint = loggers.get(owner.id, "") if isinstance(owner, ast.Name) else ""
                found.append({"logger": hint, "says": said[0], "kind": said[1], "level": node.func.attr,
                              "at": f"{rel}:{node.lineno}", "gives_plain": _gives_its_own_sentence(node)})
    return found


def _says_anything(says: str, kind: str) -> bool:
    """Whether a format string has a word of its own. "%s" or "%s" writes whatever it is given, and one sentence
    cannot stand for all of that: such a line needs a format string with words in it where it is logged."""
    literal = _PERCENT.sub("", says) if kind == "%" else says.replace("{}", "")
    return bool(re.search(r"[A-Za-z]{2}", literal))


def _literal_length(says: str, kind: str) -> int:
    return len(_PERCENT.sub("", says) if kind == "%" else says.replace("{}", ""))


def _lead(says: str, kind: str) -> str:
    """The first three characters every line of a format string begins with; "" when it begins with a value."""
    hole = _PERCENT.search(says) if kind == "%" else re.search(r"(?<!\{)\{\}", says)
    lead = says[: hole.start()] if hole else says
    lead = lead.replace("%%", "%") if kind == "%" else lead.replace("{{", "{").replace("}}", "}")
    return lead[:3] if len(lead) >= 3 else ""


class Index:
    """Format strings by the logger they are known to write under and by how their lines begin."""

    def __init__(self, sites: list[dict[str, str]]):
        self.by_logger: dict[str, list] = collections.defaultdict(list)
        self.by_lead: dict[str, list] = collections.defaultdict(list)
        seen = set()
        for site in sites:
            key = (site["logger"], site["says"])
            if key in seen:
                continue
            seen.add(key)
            entry = (_literal_length(site["says"], site["kind"]), pattern_of(site["says"], site["kind"]), site)
            if site["logger"]:
                self.by_logger[site["logger"]].append(entry)
            self.by_lead[_lead(site["says"], site["kind"])].append(entry)
        for entries in (*self.by_logger.values(), *self.by_lead.values()):
            entries.sort(key=lambda entry: -entry[0])

    def which_call(self, logger: str, text: str) -> dict[str, str] | None:
        """The call a line came from: the most specific format string that writes it, among its own logger's and those
        whose logger could not be read, so "[MLX] %s" under the line's logger does not take a line that
        "[MLX] first-token ceiling %.1fs ..." wrote under a logger built elsewhere; then by how the line begins."""
        unread = [entry for entry in self.by_lead.get(text[:3], ()) if not entry[2]["logger"]]
        for _length, pattern, site in sorted((*self.by_logger.get(logger, ()), *unread), key=lambda e: -e[0]):
            if pattern.fullmatch(text):
                return site
        candidates = sorted((*self.by_lead.get(text[:3], ()), *self.by_lead.get("", ())), key=lambda e: -e[0])
        for _length, pattern, site in candidates:
            if pattern.fullmatch(text):
                return site
        return None


def catalogue() -> dict[tuple[str, str], str]:
    rows = json.loads(CATALOGUE.read_text(encoding="utf-8"))["lines"]
    return {(row["logger"], row["says"]): row["plain"] for row in rows}


def measure(fixture: Path = FIXTURE) -> dict:
    index = Index(call_sites())
    faces = catalogue()
    lines = json.loads(fixture.read_text(encoding="utf-8"))["lines"]
    plain = unplaced = without = bare = 0
    missing: collections.Counter = collections.Counter()
    example: dict[tuple[str, str], str] = {}
    for row in lines:
        if not re.search(r"[A-Za-z]", row["text"]):
            continue  # a rule of box-drawing characters: the card shows nothing for it (stripNeuralPictographs)
        site = index.which_call(row["logger"], row["text"])
        if site is None:
            unplaced += row["count"]
            missing[(row["logger"], "(not found in her code) " + row["text"][:120])] += row["count"]
            continue
        if site.get("gives_plain"):
            plain += row["count"]
            continue
        if not _says_anything(site["says"], site["kind"]):
            bare += row["count"]
            missing[(row["logger"], f"(a bare value, {site['at']}) " + row["text"][:110])] += row["count"]
            continue
        key = (row["logger"], site["says"])
        if key in faces:
            plain += row["count"]
        else:
            without += row["count"]
            missing[key] += row["count"]
            example.setdefault(key, f"{site['at']} {site['kind']} | {row['text'][:160]}")
    events = plain + unplaced + without + bare
    return {"events": events, "plain": plain, "share": plain / max(1, events), "not_found_in_code": unplaced,
            "bare_values": bare,
            "missing": [{"logger": k[0], "says": k[1], "count": n, "example": example.get(k, "")}
                        for k, n in missing.most_common()]}


def stale_entries() -> list[tuple[str, str]]:
    """Catalogue entries whose line is no longer in her code: renamed, reworded or removed."""
    present = {site["says"] for site in call_sites()}
    return [key for key in catalogue() if key[1] not in present]


def capture(log: Path, since: str, out: Path, until: str = "~") -> int:
    counts: collections.Counter = collections.Counter()
    with log.open("rb") as handle:
        for raw in handle:
            try:
                record = json.loads(raw)
            except ValueError:
                continue
            at = str(record.get("timestamp", ""))
            if not since <= at < until or record.get("level") not in {"info", "warning", "error", "critical"}:
                continue
            counts[(str(record.get("logger", "")), str(record.get("event", "")))] += 1
    lines = [{"logger": logger, "text": text, "count": n} for (logger, text), n in counts.most_common()]
    out.write_text(json.dumps({"captured_from": f"{log.name} from {since} to {until}", "lines": lines}, ensure_ascii=False,
                              indent=0) + "\n", encoding="utf-8")
    return sum(counts.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--missing", type=int, default=0)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--since", default="")
    parser.add_argument("--until", default="~")
    parser.add_argument("--out", type=Path, default=FIXTURE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.capture:
        print(f"{capture(args.capture, args.since, args.out, args.until)} events captured to {args.out}")
        return 0
    result = measure()
    print(f"{result['share']:.1%} of the feed's events have a plain face ({result['plain']:,} of {result['events']:,}); "
          f"{result['not_found_in_code']:,} came from lines not found in her code, "
          f"{result['bare_values']:,} from calls that log a bare value")
    for row in result["missing"][: args.missing]:
        print(f"{row['count']:>5}  {row['logger']}  {row['says'][:110]!r}  [{row['example'][:90]}]")
    if args.check:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["share"]
        stale = stale_entries()
        for key in stale:
            print(f"stale: {key[0]} {key[1][:100]!r} is no longer in her code", file=sys.stderr)
        if stale or result["share"] + 1e-9 < baseline:
            print(f"failed plain faces {result['share']:.1%} against a baseline of {baseline:.1%}", file=sys.stderr)
            return 1
        print(f"ok plain faces {result['share']:.1%}, baseline {baseline:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
