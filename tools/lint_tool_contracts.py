#!/usr/bin/env python3
"""What every tool promises, counted, with a ceiling that only falls.

CrewAI requires structured schemas both ways and normalises them, so a
provider quirk cannot leak upward and a caller knows the shape of a result
without running it. Aura declares the argument side — ``input_model`` or a
``schema_override`` — and left the result to whatever the caller happened to
get.

Four things a tool should say, and each is counted separately because they
fail for different reasons:

* what its arguments are, so a caller can be wrong before it runs;
* what its result is, so a consumer can be wrong before it reads;
* which version of that contract this is, so two tools with one name and
  different arguments are told apart;
* what it does outside the process, so authority has something to admit.

The counts are a ratchet. Declaring is work and the work is not done in one
pass; what matters is that the number of undeclared ones goes down and never
up. `config/tool_contract_baseline.json` is the mark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1]
BASELINE = HERE / "config" / "tool_contract_baseline.json"


def _every_tool() -> list[Any]:
    """Every registered skill's metadata, however the registry is spelled."""

    sys.path.insert(0, str(HERE))
    from core.capability_engine import CapabilityEngine

    engine = CapabilityEngine()
    for name in ("skills", "_skills", "registry", "_registry", "metadata"):
        held = getattr(engine, name, None)
        if isinstance(held, dict) and held:
            return [one for one in held.values() if hasattr(one, "what_it_promises")]
    return []


def measure() -> dict[str, Any]:
    tools = _every_tool()
    without_arguments = sorted(
        one.name for one in tools if not one.declares_its_arguments
    )
    without_a_result = sorted(one.name for one in tools if not one.declares_its_result)
    without_an_effect = sorted(
        one.name for one in tools if str(getattr(one, "effect_scope", "")) in {"", "unknown"}
    )
    without_authority = sorted(
        one.name
        for one in tools
        if str(getattr(one, "authority_class", "")) in {"", "unclassified"}
    )
    return {
        "tools": len(tools),
        "without_arguments": len(without_arguments),
        "without_a_result": len(without_a_result),
        "without_an_effect_class": len(without_an_effect),
        "without_an_authority_class": len(without_authority),
        "which": {
            "without_arguments": without_arguments[:40],
            "without_a_result": without_a_result[:40],
            "without_an_effect_class": without_an_effect[:40],
            "without_an_authority_class": without_authority[:40],
        },
    }


#: The counts that may only fall. `tools` is not one of them — adding a tool
#: is not a regression, and a ratchet that says otherwise stops the work it
#: exists to encourage.
THE_COUNTS = (
    "without_arguments",
    "without_a_result",
    "without_an_effect_class",
    "without_an_authority_class",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args()

    now = measure()
    print(
        f"🔧 {now['tools']} tools: "
        f"{now['without_arguments']} do not say what they take, "
        f"{now['without_a_result']} do not say what they give back, "
        f"{now['without_an_effect_class']} do not say what they do outside, "
        f"{now['without_an_authority_class']} have no authority class"
    )
    for key in THE_COUNTS:
        named = now["which"][key][: args.show]
        if named:
            print(f"   {key}: {', '.join(named)}")

    if args.write_baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(
            json.dumps({key: now[key] for key in THE_COUNTS}, indent=2) + "\n"
        )
        print(f"✍️  wrote {BASELINE}")
        return 0

    if not BASELINE.exists():
        print("no baseline; run with --write-baseline once")
        return 0
    was = json.loads(BASELINE.read_text())
    rose = {
        key: (was.get(key, 0), now[key]) for key in THE_COUNTS if now[key] > was.get(key, 0)
    }
    if rose:
        for key, (before, after) in sorted(rose.items()):
            print(f"❌ {key} rose from {before} to {after}")
        print("\nA tool that does not say what it takes or gives back cannot be")
        print("checked before it runs, and a caller finds out by running it.")
        return 1
    fell = {
        key: (was.get(key, 0), now[key]) for key in THE_COUNTS if now[key] < was.get(key, 0)
    }
    for key, (before, after) in sorted(fell.items()):
        print(f"✅ {key} fell from {before} to {after} (the baseline should shrink)")
    print("✅ no tool contract regressed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
