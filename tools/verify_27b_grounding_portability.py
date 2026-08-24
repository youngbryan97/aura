#!/usr/bin/env python3
"""Which grounding contracts actually survived the tokenizer change.

The 27B vocabulary is 248,320 entries against the 32B's 152,064, and the first
migration pass concluded from that alone that every tokenizer-bound contract was
dead. Re-deriving them says otherwise, in both directions.

The ten digit ids are **identical** -- 15 through 24 on both checkpoints -- so
literal grounding, which is the contract the typed answer path depends on most,
crosses intact. All seven frontier opcode markers **changed**, because they are
multi-token English phrases and the merges around them moved. And two of the
five files first listed as tokenizer-bound never touch a tokenizer at all: the
action and state schemas bind a typed opcode vocabulary, which is not a
vocabulary in the tokenizer sense.

A guess in either direction is expensive. Calling a portable contract dead
throws away tissue that still works; calling a changed one portable serves
answers assembled from token ids that mean something else now. So this reads
both tokenizers and compares, and reports per contract rather than per file
count.

    python tools/verify_27b_grounding_portability.py
    python tools/verify_27b_grounding_portability.py --json OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

GROUNDING_PORTABILITY_SCHEMA: Final = "aura.rlc.27b_grounding_portability.v1"

#: Contracts that bind a typed vocabulary rather than a tokenizer. Listing them
#: keeps the report honest about what was never at risk.
TYPED_ONLY: Final = (
    "core/learning/recurrent_action_schema.py",
    "core/learning/recurrent_state_schema.py",
)


def _install_root() -> Path:
    if (REPO_ROOT / "training/fused-model/active.json").exists():
        return REPO_ROOT
    import subprocess

    try:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return REPO_ROOT
    return Path(common).parent if common else REPO_ROOT


INSTALL = _install_root()
LEGACY_MODEL = (
    INSTALL / "training/fused-model/Aura-32B-crsm-closeout-jul1-20260701-215118"
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _probe(tokenizer: Any) -> dict[str, Any]:
    """Every token binding the recurrent contracts derive, under one tokenizer."""
    from core.learning.recurrent_literal_grounding import tokenizer_digit_token_ids
    from core.learning.recurrent_opcode_grounding import (
        _FRONTIER_FAMILY_TEXT,
        _tokenizer_patterns,
    )

    digits = list(tokenizer_digit_token_ids(tokenizer))
    opcodes = {
        str(label): list(pattern)
        for label, pattern in _tokenizer_patterns(tokenizer, _FRONTIER_FAMILY_TEXT)
    }

    from core.learning.recurrent_answer_emission import _encode_exact

    # The emission contract's family prefixes are exact JSON fragments, so they
    # are the part of it that a merge-table change can move.
    prefixes = {
        name: list(_encode_exact(tokenizer, text))
        for name, text in (
            ("khop", '{"node":'),
            ("modular", '{"residue":'),
        )
    }
    return {
        "digit_token_ids": digits,
        "opcode_marker_patterns": opcodes,
        "answer_emission_prefixes": prefixes,
    }


def _compare(name: str, old: Any, new: Any, consumer: str) -> dict[str, Any]:
    portable = old == new
    return {
        "binding": name,
        "consumer": consumer,
        "portable": portable,
        "verdict": "portable" if portable else "must_regenerate",
        "legacy_sha256": _digest(old),
        "target_sha256": _digest(new),
    }


def build(tokenizers: dict[str, Any] | None = None) -> dict[str, Any]:
    if tokenizers is None:
        from transformers import AutoTokenizer

        active = json.loads(
            (INSTALL / "training/fused-model/active.json").read_text()
        )["active_model_path"]
        tokenizers = {
            "legacy": AutoTokenizer.from_pretrained(str(LEGACY_MODEL)),
            "target": AutoTokenizer.from_pretrained(str(active)),
        }

    legacy = _probe(tokenizers["legacy"])
    target = _probe(tokenizers["target"])

    findings = [
        _compare(
            "digit_token_ids",
            legacy["digit_token_ids"],
            target["digit_token_ids"],
            "recurrent_literal_grounding.py",
        ),
        _compare(
            "opcode_marker_patterns",
            legacy["opcode_marker_patterns"],
            target["opcode_marker_patterns"],
            "recurrent_opcode_grounding.py",
        ),
        _compare(
            "answer_emission_prefixes",
            legacy["answer_emission_prefixes"],
            target["answer_emission_prefixes"],
            "recurrent_answer_emission.py",
        ),
    ]

    changed_opcodes = {
        label: {
            "legacy": legacy["opcode_marker_patterns"][label],
            "target": target["opcode_marker_patterns"][label],
        }
        for label in sorted(legacy["opcode_marker_patterns"])
        if legacy["opcode_marker_patterns"][label]
        != target["opcode_marker_patterns"].get(label)
    }

    body = {
        "schema": GROUNDING_PORTABILITY_SCHEMA,
        "legacy_checkpoint": str(LEGACY_MODEL),
        "findings": findings,
        "changed_opcode_markers": changed_opcodes,
        "typed_only_contracts": {
            path: "binds a typed opcode vocabulary; no tokenizer involved"
            for path in TYPED_ONLY
        },
        "must_regenerate": sorted(
            f["consumer"] for f in findings if not f["portable"]
        ),
        "portable": sorted(f["consumer"] for f in findings if f["portable"]),
        "legacy_bindings": legacy,
        "target_bindings": target,
    }
    return {**body, "report_sha256": _digest(body)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = build()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1, sort_keys=True))
        print(f"wrote {args.json}")

    for finding in report["findings"]:
        mark = "portable      " if finding["portable"] else "MUST REGENERATE"
        print(f"  {mark}  {finding['binding']:26s} {finding['consumer']}")
    for path in report["typed_only_contracts"]:
        print(f"  not at risk     {'typed vocabulary':26s} {path}")
    if report["changed_opcode_markers"]:
        print(f"\n{len(report['changed_opcode_markers'])} opcode markers moved:")
        for label, pair in report["changed_opcode_markers"].items():
            print(f"  opcode {label}: {pair['legacy']} -> {pair['target']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
