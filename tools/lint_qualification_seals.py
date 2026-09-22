"""Whether the code under a qualified claim is still the code that was measured.

An activation envelope names the exact symbols and files a qualification ran
against and records their hashes. Change one and the claim describes code that
no longer exists — but the only thing that checked it was a test deep in the
full suite, which is hours away from the commit that broke it. Four symbols had
drifted by the time one reached it, across three commits and eight days.

This hashes the same selectors the envelope recorded and names every one that
moved. It never reseals: a seal is refreshed by re-running the qualification,
not by writing down what the code says today.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _drifted(root: Path) -> list[str]:
    from core.brain.llm.semantic_neural_serving import source_contract_sha256s
    from core.learning.compositional_semantic_qualification import (
        COMPOSITIONAL_SEMANTIC_SOURCE_CONTRACTS,
    )

    envelope = root / "artifacts/rlc/semantic_program_27b_frozen_path_v1/activation.json"
    if not envelope.is_file():
        return [f"MISSING  {envelope.relative_to(root)}"]
    sealed = json.loads(envelope.read_text(encoding="ascii")).get(
        "source_contract_sha256s"
    )
    if not isinstance(sealed, dict) or not sealed:
        return [f"EMPTY    {envelope.relative_to(root)} records no source contracts"]
    current = source_contract_sha256s(root, COMPOSITIONAL_SEMANTIC_SOURCE_CONTRACTS)
    out = []
    for key in sorted(set(sealed) | set(current)):
        was, now = sealed.get(key), current.get(key)
        if was == now:
            continue
        if now is None:
            out.append(f"GONE     {key}")
        elif was is None:
            out.append(f"NEW      {key} is not in the envelope")
        else:
            out.append(f"DRIFTED  {key}  {was[:12]} -> {now[:12]}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    findings = _drifted(Path(args.root).resolve())
    if not findings:
        print("✅ every qualified symbol still hashes to what was measured")
        return 0
    print(f"{len(findings)} qualified symbol(s) no longer match the envelope:")
    for one in findings:
        print("  ", one)
    print(
        "\nA seal is refreshed by re-running the qualification, never by "
        "writing down what the code says today."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
