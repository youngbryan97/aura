#!/usr/bin/env python3
"""Flag the writing patterns that read as machine-generated.

The rules live in docs/WRITING_RULES.md; this file is the executable half.
Adding a pattern there without adding it here leaves the rule unenforced,
which is how the last set of writing conventions quietly stopped applying.
The triad rule sat documented-but-unenforced for exactly that reason.

Scope is deliberate. Append-only ledgers and dated records are NOT checked:
they are the record, and editing a July entry in August falsifies it. See
docs/DOC_STATUS.md for which documents are which.

Prose is not only Markdown. Docstrings and comments are read more often than
the guides are, so `--code` lints them from the same rulebook.

    python tools/lint_ai_writing.py                 # the front-facing docs
    python tools/lint_ai_writing.py --all           # every guide
    python tools/lint_ai_writing.py --code          # docstrings and comments
    python tools/lint_ai_writing.py FILE [FILE...]  # specific files
    python tools/lint_ai_writing.py --baseline      # write the ratchet file
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import sys
import token as token_mod
import tokenize
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "config" / "ai_writing_baseline.json"
#: Baselines are per-scope; an --all run is not comparable to a front-docs run.
SCOPES = ("front", "all", "code", "adhoc")

#: Documents that are records rather than prose we maintain.
EXCLUDE_PREFIX = (
    "docs/AURA_EXECUTION_TRACKER.md",
    "docs/RLC_SPARK_EXECUTION_LEDGER.md",
    "docs/AURA_EXECUTION_PLAN.md",
    "docs/AURA_PROMPT_COVERAGE_AUDIT.md",
    "docs/evidence/",
    "docs/runbooks/",
    "scoping/",
    "aura_bench/",
    "archive/",
    "scratch/",
    "dev_archive/",
    "models/",
    "artifacts/",
    "research/",
    "specs/",
    "tests/",
    "security/",
    "proof_kernel/",
    "training/",
    "demos/",
    "challenges/",
    ".github/",
    "data/",
    "docs/FMEA.md",
    "docs/RUNTIME_CONTRACT.md",
    "docs/ARCHITECTURE_MAP.md",
    "docs/AURA_PROGRESS.md",
)
EXCLUDE_DATED = re.compile(r"_20\d\d[_-]\d\d[_-]\d\d\.md$|_2026_\d\d\.md$|RESULTS\.md$")

#: Python trees whose prose is generated, vendored, or a fixture rather than ours.
EXCLUDE_CODE_PREFIX = (
    "tests/",
    "archive/",
    "dev_archive/",
    "scratch/",
    "training/",
    "demos/",
    "challenges/",
    "aura_bench/",
    "proof_kernel/",
    "third_party/",
    "vendor/",
)

#: The pages a visitor actually reads.
FRONT = (
    "README.md",
    "HOW_IT_WORKS.md",
    "ARCHITECTURE.md",
    "CONTRIBUTING.md",
    "INSTALL.md",
    "TESTING.md",
    "MODEL_CARD.md",
    "CHANGELOG.md",
    "AGENTS.md",
    "docs/README.md",
    "docs/RECURSIVE_LATENT_CORTEX.md",
    "docs/INTRINSIC_RECURRENCE.md",
    "docs/COGNITIVE_ARCHITECTURE_ADOPTION.md",
    "docs/MODEL_ROSTER.md",
    "docs/ENGINEERING_ADOPTION.md",
    "docs/USER_GUIDE.md",
    "docs/OPERATOR_GUIDE.md",
)

#: A rule is a name, a pattern, the one-line fix, and an optional predicate that
#: sees the match and can drop it. The predicate exists because several tells are
#: only tells in aggregate: one tricolon is craft, a tricolon per paragraph is
#: autopilot. Without it every rule has to be a boolean, and boolean rules on
#: legitimate constructions are how a linter earns the right to be ignored.
Rule = tuple[str, re.Pattern[str], str, Callable[[re.Match[str]], bool] | None]

#: The tell is not "three things" — this codebase enumerates three real states
#: constantly ("warming, recovering, handshaking") and those are counts, not
#: rhythm. The tell is three *evaluative* words that could each be swapped for
#: any other three without changing the claim: "faster, cheaper, smarter". So the
#: rule fires only when all three members come from this list. It is curated
#: rather than suffix-derived because every suffix rule tried here also caught
#: nouns (integrity, authority, containment) and domain adjectives (logical,
#: mathematical, physical). Add words as they turn up in drafts.
_EVALUATIVE = {
    "fast", "faster", "fastest", "quick", "quicker", "slow", "slower",
    "cheap", "cheaper", "cheapest", "costly", "expensive",
    "smart", "smarter", "clever", "intelligent",
    "simple", "simpler", "easy", "easier", "hard", "harder",
    "powerful", "scalable", "seamless", "elegant", "efficient",
    "reliable", "flexible", "modern", "intuitive", "lightweight",
    "versatile", "sophisticated", "innovative", "streamlined",
    "optimized", "enhanced", "improved", "better", "safer", "stronger",
    "leaner", "tighter", "richer", "deeper", "broader", "sharper",
    "cleaner", "clearer", "bigger", "smaller", "greater",
    "crucial", "vital", "essential", "compelling", "meaningful",
    "dynamic", "vibrant", "engaging", "exciting", "delightful",
    "refreshing", "rejuvenating", "invigorating", "transformative",
}


def _triad_of_evaluatives(m: re.Match[str]) -> bool:
    """Keep only triads whose three members are all evaluative words."""
    return all(g.lower() in _EVALUATIVE for g in m.groups() if g)


RULES: list[Rule] = [
    (
        "negation-flip",
        re.compile(
            r"(?:That's|That is|This is|It's|It is)\s+not\s+[^.!?\n]{1,60}[.!?]\s+"
            r"(?:That's|That is|This is|It's|It is)\s",
        ),
        '"That\'s not X. That\'s Y." Say the second half only.',
        None,
    ),
    (
        # The comma-spliced sibling of negation-flip, and the tell every source
        # in docs/WRITING_RULES.md's reading list names first. The rule above
        # requires a sentence boundary, so it never saw this form.
        "negation-flip-inline",
        re.compile(
            r"\b(?:is|are|was|were|it'?s|that'?s|this is|they'?re|we'?re|you'?re)\s+not\s+"
            r"(?:just|only|merely|simply|about)\s+[^.!?;\n]{2,60}[,;]\s*"
            r"(?:it'?s|that'?s|they'?re|we'?re|you'?re|but)\b"
            r"|\bnot only\b[^.!?\n]{2,80}\bbut also\b"
            r"|\bnot\s+[^.!?;\n]{2,50},\s*but rather\b",
            re.I,
        ),
        '"Not just X, it\'s Y." Same trick as negation-flip. Say the second half.',
        None,
    ),
    (
        "stapled-fragments",
        re.compile(r"(?<=[.!?:]\s)([A-Z][a-z]{2,12})\.\s([A-Z][a-z]{2,12})\.\s(?=[A-Z])"),
        "Two one-word sentences in a row. Pick one and write it as a sentence.",
        None,
    ),
    (
        # The comma is load-bearing. Without it the rule also ate correlatives
        # ("the less was known, the more likely it trained") and comparisons
        # ("on less energy is more fit"), which are arguments, not twin images.
        "twin-images",
        re.compile(r"\bless\s+(?:an?\s+)?[\w-]+\s*,\s*more\s+(?:an?\s+)?[\w-]+", re.I),
        '"Less a hammer, more a scalpel." Say what to do instead.',
        None,
    ),
    (
        # Rule 7 of docs/WRITING_RULES.md, documented since the file was written
        # and never enforced. Restricted to all-modifier triads so that real
        # lists of three things pass.
        "reflexive-triad",
        re.compile(
            r"\b([a-z]{4,14}),\s+([a-z]{4,14}),\s+(?:and\s+|or\s+)?([a-z]{4,14})\b"
            r"(?=[.;:,]|\s+(?:and|or)\b|\s*$)",
        ),
        "Three modifiers in a row is a rhythm, not a count. Cut to the one you mean.",
        _triad_of_evaluatives,
    ),
    (
        "self-applause",
        re.compile(
            r"\b(?:and that matters"
            r"|that'?s? (?:the part|what) (?:everyone|most people|nobody) (?:miss|gets?)"
            r"|which is exactly the point|that'?s (?:exactly )?the (?:whole )?point"
            # Bare "worth reading" also matched "before its precision is worth
            # reading", which is a statement about statistical support. The
            # clapping sense needs a demonstrative pointing at the text itself.
            r"|is the important part|(?:this|that|it) is worth reading|the part worth"
            r"|which is the whole point|and that'?s the thing"
            r"|cannot be overstated|it'?s worth stating)\b",
            re.I,
        ),
        "Clapping for itself. Delete it; you lose nothing.",
        None,
    ),
    (
        # A product name (Excel, Photoshop), not a word shouted for emphasis:
        # "it's the SURPRISE of load increasing" is a definition, not an analogy.
        "borrowed-analogy",
        re.compile(r"\b[Ii]t'?s the [A-Z][a-z][A-Za-z0-9-]* of\b"),
        '"It\'s the Excel of X." Only works if the reader knows both things.',
        None,
    ),
    (
        "throat-clearing",
        re.compile(
            r"(?:^|(?<=[.!?]\s)|(?<=\n))\s*(?:Here'?s the thing|Let me be clear"
            r"|The truth is|The reality is|The thing is|Here'?s what'?s"
            r"|What'?s (?:interesting|striking) (?:here )?is|Make no mistake)\b",
            re.I,
        ),
        "Warming up. Start one sentence later.",
        None,
    ),
    (
        "hedged-range",
        re.compile(
            # Not preceded by a digit-and-dash, so an ISO date does not read
            # as a range. "2026-04-27 second reduction" was flagged as
            # "04-27 second": the rule is for "took 4-27 seconds", and a date
            # followed by a unit word is the shape it must not match.
            r"(?<!\d-)(?<!\d)\b\d+\s*(?:to|–|—|-)\s*\d+\s*"
            r"(?:seconds?|minutes?|hours?|days?|weeks?|months?|s\b|ms\b)",
            re.I,
        ),
        "A range means it was never measured. Give the number.",
        None,
    ),
    (
        "recap-ending",
        re.compile(
            # "In short" needs the comma: `short` is an ordinary adjective and
            # the bare form matched "in short slices", "in short micro-batches",
            # and "in short windows" all over the runtime.
            r"(?:\bIn short[,:]"
            r"|\b(?:At the end of the day|To sum up|In summary"
            r"|In conclusion|The bottom line|The upshot is|All told)\b)",
            re.I,
        ),
        "The ending that repeats the post. Just stop typing.",
        None,
    ),
    (
        # A trailing participle that restates the clause it hangs off instead of
        # adding to it: "the system retries, ensuring reliability."
        "participial-tail",
        re.compile(
            r",\s+(?:highlighting|underscoring|showcasing|ensuring|reflecting"
            r"|emphasi[sz]ing|demonstrating|illustrating|signif(?:ying|ies)"
            r"|allowing for|paving the way|solidifying|cementing)\b",
            re.I,
        ),
        "A participle that restates the sentence. End the sentence instead.",
        None,
    ),
    (
        "disclaimer-hedge",
        re.compile(
            r"\b(?:it (?:is|'s) (?:important|crucial|essential|worth) to note"
            r"|it (?:is|'s) worth (?:noting|mentioning|pointing out)"
            r"|it should be noted|please note that"
            r"|generally speaking|broadly speaking|to some extent"
            r"|from a broader perspective|it (?:is|'s) important to (?:remember|understand))\b",
            re.I,
        ),
        "Hedging before the fact. State the fact.",
        None,
    ),
    (
        # This repo registers claims against the tests that validate them
        # (core/organism/model_validation.py). Unsourced attribution is a
        # writing tell and an unregistered claim at once.
        "vague-attribution",
        re.compile(
            r"\b(?:many (?:argue|believe|say|would argue)"
            r"|some (?:argue|believe|say|experts?|researchers?)"
            r"|it is (?:widely|generally|commonly) (?:believed|accepted|known|understood)"
            r"|studies (?:show|suggest|indicate)"
            r"|research (?:shows|suggests|indicates)"
            r"|experts (?:agree|say|suggest))\b",
            re.I,
        ),
        "Who? Name the source or cut the claim.",
        None,
    ),
    (
        "rhetorical-question",
        re.compile(
            r"(?:^|(?<=[.!?]\s)|(?<=\n))\s*(?:The (?:result|catch|problem|answer|kicker|point)"
            r"|Why (?:does (?:this|that) matter|it matters)"
            r"|Sound familiar|The best part|What (?:does (?:this|that) mean|gives))\?",
            re.I,
        ),
        "Asking yourself a question to answer it. Just answer it.",
        None,
    ),
    (
        "false-collaboration",
        re.compile(
            r"\b(?:let'?s (?:dive|delve|explore|take a look|walk through|unpack|jump)"
            r"|we'?ll (?:dive|delve|explore|walk through|unpack))\b",
            re.I,
        ),
        "Nobody is here with you. Say what the thing does.",
        None,
    ),
    (
        "cliche-opener",
        re.compile(
            r"\b(?:in today'?s [\w\s-]{0,20}(?:world|landscape|environment|era)"
            r"|in the (?:ever-|rapidly )?(?:evolving|changing|growing) (?:world|landscape|field)"
            r"|as the world continues to"
            r"|in the (?:realm|world|landscape) of)\b",
            re.I,
        ),
        "A stock opening. Start at the first real sentence.",
        None,
    ),
    (
        # Latinate inflation: the long word where the short one was exact.
        "inflated-diction",
        re.compile(
            r"\b(?:utili[sz]e[sd]?|utili[sz]ing"
            r"|facilitate[sd]?|endeavou?r to|in order to|subsequent to"
            # `prior` is a noun in this tree ("a reasonable prior to start
            # with"), so the preposition only counts without a determiner in
            # front of it. Rewriting that line to "a reasonable before" was
            # caught by review, not by the rule, which is why the guard exists.
            r"|(?<!\ba )(?<!\ban )(?<!\bthe )(?<!\bour )(?<!\bits )(?<!\bweak )"
            r"(?<!\bflat )(?<!\bstrong )(?<!\buniform )(?<!\breasonable )"
            r"(?<!\bconjugate )(?<!\binformative )(?<!\bBayesian )prior to"
            r"|delve[sd]? into|delving into"
            r"|bolster(?:s|ed|ing)?"
            r"|seamless(?:ly)?|pivotal|myriad|plethora"
            # No `beacon`: Salt's beacon→reactor architecture is real here, and
            # `DegradationBeacon` is a class. No `underscore`: in a Python tree
            # that word is the character, never the verb.
            r"|tapestry|cacophony"
            r"|elevate[sd]? the (?:experience|conversation|discourse|game)|unlock(?:s|ing)? the (?:power|potential)"
            r"|harness(?:es|ing)? the (?:power|potential)"
            r"|empower(?:s|ing|ed)? (?:you|users|teams|developers))\b",
            re.I,
        ),
        "The long word where the short one was exact. Use the short one.",
        None,
    ),
    (
        # Never seen in this tree, and the point is that it stays that way: these
        # strings only appear when a reply was pasted in without being read.
        "chatbot-remnant",
        re.compile(
            r"(?:as an? (?:AI|large )?language model"
            r"|I'?m sorry,? but as an"
            r"|would you like me to"
            r"|(?:I hope|hope) this helps"
            r"|let me know if you(?:'d| would) like"
            r"|utm_source=chatgpt\.com"
            r"|here'?s? (?:a|the) (?:revised|updated) version)",
            re.I,
        ),
        "A chat reply pasted in unread. Delete the whole line.",
        None,
    ),
]


#: A `backticked span` is a code reference and a "quoted span" is somebody else's
#: words; docs/WRITING_RULES.md exempts both, since quoted material keeps its
#: original wording even when the original commits one of the nine. Blanking them
#: (rather than deleting) keeps every column offset stable so reported line
#: numbers stay true. Without this the rules page trips its own examples, and a
#: docstring that names the phrase it detects is read as having written it.
INLINE_CODE = re.compile(r"`[^`\n]*`|\"[^\"\n]{0,120}\"|“[^”\n]{0,120}”")

#: Comment markers that are machine directives rather than prose.
DIRECTIVE = re.compile(
    r"^#\s*(?:type:|noqa|pylint:|pragma:|ruff:|mypy:|fmt:|isort:|!|-\*-|coding[:=])"
)


def prose_lines(text: str) -> list[tuple[int, str]]:
    """Lines that are prose: no code fences, tables, indented blocks, or code spans."""
    out: list[tuple[int, str]] = []
    fence = False
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        if stripped.startswith(("|", ">", "#")) or line.startswith(("    ", "\t")):
            continue
        out.append((i, INLINE_CODE.sub(lambda m: " " * len(m.group(0)), line)))
    return out


def python_prose_lines(text: str) -> list[tuple[int, str]]:
    """Docstrings and comments from a Python source file, with true line numbers.

    Code is never returned. A docstring's body is walked line by line so that a
    match inside a forty-line module docstring still reports the line it is on.
    """
    out: list[tuple[int, str]] = []

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out

    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        doc = ast.get_docstring(node, clean=False)
        if not doc:
            continue
        # The docstring expression's own end line, walked backwards over its text,
        # gives the first line of the literal without re-parsing the source.
        body = node.body[0]
        end = getattr(body, "end_lineno", None)
        if end is None:
            continue
        doc_lines = doc.splitlines()
        # The opening quotes share a line with the first line of content, so the
        # body starts one line later than a bare subtraction gives. Without the
        # +1 every docstring finding points at the `def` above it.
        start = end - len(doc_lines) + 1
        for offset, line in enumerate(doc_lines):
            out.append((start + offset, INLINE_CODE.sub(
                lambda m: " " * len(m.group(0)), line)))

    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type != token_mod.COMMENT:
                continue
            if DIRECTIVE.match(tok.string):
                continue
            body = tok.string.lstrip("#").lstrip()
            if body:
                out.append((tok.start[0], INLINE_CODE.sub(
                    lambda m: " " * len(m.group(0)), body)))
    except (tokenize.TokenError, IndentationError):
        pass

    return sorted(set(out))


def scan_file(rel: str) -> list[tuple[int, str, str]]:
    path = ROOT / rel
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    extract = python_prose_lines if path.suffix == ".py" else prose_lines
    lines = extract(text)
    # Join so patterns can span a hard-wrapped sentence.
    offsets: list[tuple[int, int]] = []
    pos = 0
    parts: list[str] = []
    for lineno, line in lines:
        offsets.append((pos, lineno))
        parts.append(line)
        pos += len(line) + 1
    blob = "\n".join(parts)

    def lineof(idx: int) -> int:
        best = offsets[0][1] if offsets else 0
        for start, lineno in offsets:
            if start <= idx:
                best = lineno
            else:
                break
        return best

    hits: list[tuple[int, str, str]] = []
    for name, rx, _, keep in RULES:
        for m in rx.finditer(blob):
            if keep is not None and not keep(m):
                continue
            frag = " ".join(m.group(0).split())[:80]
            hits.append((lineof(m.start()), name, frag))
    return sorted(hits)


def _tracked(pattern: str) -> list[str]:
    return subprocess.run(
        ["git", "ls-files", pattern], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()


def tracked_guides() -> list[str]:
    return [
        rel
        for rel in _tracked("*.md")
        if not rel.startswith(EXCLUDE_PREFIX) and not EXCLUDE_DATED.search(rel)
    ]


def tracked_code() -> list[str]:
    return [rel for rel in _tracked("*.py") if not rel.startswith(EXCLUDE_CODE_PREFIX)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--all", action="store_true", help="every tracked guide")
    ap.add_argument("--code", action="store_true", help="docstrings and comments")
    ap.add_argument("--baseline", action="store_true", help="rewrite the ratchet")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--counts", action="store_true", help="totals per rule")
    args = ap.parse_args()

    if args.files:
        scope, targets = "adhoc", args.files
    elif args.code:
        scope, targets = "code", tracked_code()
    elif args.all:
        scope, targets = "all", tracked_guides()
    else:
        scope, targets = "front", list(FRONT)

    results = {rel: scan_file(rel) for rel in targets}
    results = {k: v for k, v in results.items() if v}
    total = sum(len(v) for v in results.values())

    explain = {name: why for name, _, why, _ in RULES}
    if not args.quiet:
        for rel in sorted(results):
            print(f"\n{rel}")
            for lineno, name, frag in results[rel]:
                print(f"  {lineno:>5}  {name:<22} {frag}")
                print(f"         {'':<22} -> {explain[name]}")

    if args.counts:
        tally: dict[str, int] = {}
        for hits in results.values():
            for _, name, _ in hits:
                tally[name] = tally.get(name, 0) + 1
        print("\nper rule:")
        for name, n in sorted(tally.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>5}  {name}")

    if args.baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
        data[scope] = {"total": total,
                       "per_file": {k: len(v) for k, v in results.items()}}
        BASELINE.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(f"\nbaseline written for scope '{scope}': {total} findings")
        return 0

    print(f"\n{total} findings across {len(targets)} documents")

    if BASELINE.exists() and scope != "adhoc":
        stored = json.loads(BASELINE.read_text())
        if scope not in stored:
            print(f"no baseline for scope '{scope}'; run --baseline to record one")
            return 0
        prior = stored[scope]["total"]
        if total > prior:
            print(f"FAIL: {total} > baseline {prior}. The ratchet only goes down.")
            return 1
        if total < prior:
            print(f"OK: {total} < baseline {prior}. Re-run with --baseline to tighten.")
        else:
            print(f"OK: at baseline {prior}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
