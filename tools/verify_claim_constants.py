#!/usr/bin/env python3
"""Every constant a claim cites must still hold that value in the code.

CLAIMS_MATRIX.md is what an external reviewer reads to learn what this system
does and does not do. Claim 4a said, as current fact, that `DEFAULT_ALPHA` is
5.0 and `_INJECTION_ALPHA_CEILING` clips it to 3.0, and concluded that affect
steering ships below the magnitude at which it changes a token.

c7dcc548a had already fixed that, nine days earlier: alpha became a fraction of
the residual stream, measured on both models, DEFAULT_ALPHA 0.2 with the
ceiling at 0.6 — which is the closure condition claim 4a itself named. The code
moved and the record did not, so an outside reviewer read the matrix, reasoned
correctly from it, and reached a conclusion that had stopped being true.

This repository already refuses claims whose evidence was retracted
(`make evidence-integrity`). It had nothing for a claim whose evidence is a
number in a file that has since changed. Nothing catches a record that is wrong
in the direction of understating, because understating looks like caution.

The extraction is deliberately narrow. It reads an assertion only where the
matrix uses an explicit copula between a backticked identifier and a numeric
literal — "`X` is 5.0", "`X` clips it to 3.0", "`X` derates to 0.35". Prose that
merely mentions a number near a name is not an assertion about that name's
value, and treating it as one would make this gate cry wolf until somebody
turned it off.

Exit code is non-zero when any cited constant no longer holds the cited value,
or when a claim cites a constant that no longer exists.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = ROOT / "CLAIMS_MATRIX.md"
DEFAULT_SCANNED = ("core", "interface", "tools", "training")

#: A claim asserts a constant's value only through an explicit copula. Anything
#: looser matches prose ("measured at magnitude 150, `DEFAULT_ALPHA` was the
#: problem") and produces findings nobody can act on.
_ASSERTION = re.compile(
    r"`(?P<name>_?[A-Za-z][A-Za-z0-9_]*)`\s+"
    r"(?:is\s+now|is|=|clips\s+it\s+to|clips\s+to|derates\s+to|defaults\s+to)\s+"
    r"`?(?P<value>-?\d+(?:\.\d+)?)`?"
)

#: Names common enough in prose that a copula match is more likely to be English
#: than an assertion about a constant. Each is a word the matrix uses as a noun.
_NOT_CONSTANTS = frozenset({"n", "alpha", "p", "d", "k"})


@dataclass(frozen=True)
class Assertion:
    name: str
    value: float
    line: int
    excerpt: str


@dataclass(frozen=True)
class Finding:
    kind: str
    name: str
    asserted: float
    actual: tuple[str, ...]
    line: int
    excerpt: str

    def render(self) -> str:
        if self.kind == "missing":
            return (
                f"CLAIMS_MATRIX.md:{self.line}: claims `{self.name}` is "
                f"{self.asserted:g}, but no such constant is defined anywhere "
                f"under {', '.join(DEFAULT_SCANNED)}\n    {self.excerpt}"
            )
        return (
            f"CLAIMS_MATRIX.md:{self.line}: claims `{self.name}` is "
            f"{self.asserted:g}; the code says {', '.join(self.actual)}\n"
            f"    {self.excerpt}"
        )


def _numeric(node: ast.AST) -> float | None:
    """The value of a numeric literal, including a negated one."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            return None
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -float(node.operand.value)
    return None


def collect_constants(roots: tuple[str, ...]) -> dict[str, list[tuple[str, float]]]:
    """Every module-level and class-level numeric constant, by name.

    A name can be defined in more than one place. All definitions are kept, and
    an assertion is satisfied by ANY of them — the matrix names a constant, not
    a module, so demanding a unique definition would fail on names that are
    legitimately reused.
    """
    found: dict[str, list[tuple[str, float]]] = {}
    for root in roots:
        base = ROOT / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in str(path):
                continue
            try:
                tree = ast.parse(path.read_text("utf-8"))
            except (OSError, SyntaxError, UnicodeError):
                continue
            rel = str(path.relative_to(ROOT))
            for node in ast.walk(tree):
                if isinstance(node, ast.AnnAssign):
                    targets = [node.target]
                    value = node.value
                elif isinstance(node, ast.Assign):
                    targets = list(node.targets)
                    value = node.value
                else:
                    continue
                if value is None:
                    continue
                number = _numeric(value)
                if number is None:
                    continue
                for target in targets:
                    if isinstance(target, ast.Name):
                        found.setdefault(target.id, []).append((rel, number))
    return found


def read_assertions(matrix_path: Path) -> list[Assertion]:
    assertions: list[Assertion] = []
    for lineno, line in enumerate(matrix_path.read_text("utf-8").splitlines(), 1):
        for match in _ASSERTION.finditer(line):
            name = match.group("name")
            if name.lower() in _NOT_CONSTANTS:
                continue
            # A constant name carries an underscore or is fully upper case.
            # Ordinary prose words do neither, and this is what keeps the gate
            # from reading English as an assertion.
            if not ("_" in name or name.isupper()):
                continue
            start = max(0, match.start() - 60)
            assertions.append(
                Assertion(
                    name=name,
                    value=float(match.group("value")),
                    line=lineno,
                    excerpt="…" + line[start : match.end() + 40].strip() + "…",
                )
            )
    return assertions


def verify(matrix_path: Path, roots: tuple[str, ...]) -> list[Finding]:
    constants = collect_constants(roots)
    findings: list[Finding] = []
    for assertion in read_assertions(matrix_path):
        definitions = constants.get(assertion.name)
        if not definitions:
            findings.append(
                Finding(
                    kind="missing",
                    name=assertion.name,
                    asserted=assertion.value,
                    actual=(),
                    line=assertion.line,
                    excerpt=assertion.excerpt,
                )
            )
            continue
        if any(abs(value - assertion.value) < 1e-9 for _, value in definitions):
            continue
        findings.append(
            Finding(
                kind="mismatch",
                name=assertion.name,
                asserted=assertion.value,
                actual=tuple(f"{path}={value:g}" for path, value in definitions),
                line=assertion.line,
                excerpt=assertion.excerpt,
            )
        )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    parser.add_argument("--json", action="store_true", help="emit machine-readable findings")
    args = parser.parse_args(argv)

    matrix_path = Path(args.matrix)
    if not matrix_path.exists():
        print(f"error: no claims matrix at {matrix_path}", file=sys.stderr)
        return 2

    findings = verify(matrix_path, DEFAULT_SCANNED)
    assertions = read_assertions(matrix_path)

    if args.json:
        print(
            json.dumps(
                {
                    "assertions_checked": len(assertions),
                    "findings": [
                        {
                            "kind": f.kind,
                            "name": f.name,
                            "asserted": f.asserted,
                            "actual": list(f.actual),
                            "line": f.line,
                        }
                        for f in findings
                    ],
                    "ok": not findings,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1 if findings else 0

    print(f"claim constants checked: {len(assertions)}")
    if not findings:
        print("every constant a claim cites still holds that value")
        return 0
    print(f"{len(findings)} claim(s) cite a constant the code no longer holds:")
    for finding in findings:
        print("  " + finding.render())
    print(
        "\nThe record is what an outside reviewer reasons from. A claim describing "
        "a constant that has since changed sends them to a conclusion that stopped "
        "being true."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
