#!/usr/bin/env python3
"""Where her felt push is written, chosen from a felt_wide_layers.py reading by the rule that chose hers.

On 24 October her felt push went where it moved the rating she gives in a
conversation most, among the places it cost the answers with one right answer
no more than a random push of the same size (layer 43 at eight half-differences:
a swing of +0.981, with a margin of +0.303 against -0.033). The rule was applied
by reading a table. Here it is code, so the same rule picks the place in
another model and nobody picks it by eye.

    python tools/choose_felt_cell.py <felt_wide_layers json>

Prints the layer, the size and the sentence make_felt_everywhere_generation.py
records as how the push was placed, or exits 1 when no cell qualifies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def eligible(cells: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The cells whose margin at the high state is no worse than a random push's of the same norm."""
    return {name: cell for name, cell in cells.items() if float(cell["margin_felt"]) >= float(cell["margin_random"])}


def choose(reading: dict[str, Any], when: str = "") -> dict[str, Any] | None:
    """The eligible cell with the largest in-context swing, and how it was chosen; None when none is eligible."""
    cells = eligible(reading["cells"])
    if not cells:
        return None
    name, cell = max(cells.items(), key=lambda item: (float(item[1]["swing"]), item[0]))
    layer, size = name.split("x")
    shown = len(cell.get("1", [])) or None
    return {
        "cell": name,
        "layer": int(layer),
        "size": float(size),
        "reading": (
            f"felt_wide_layers.py{f' ({when})' if when else ''}: the largest in-context swing "
            f"({float(cell['swing']):+.3f}, {int(cell['up'])} of {shown or '?'} conversations) among the cells whose "
            f"right-answer margin at the high state is no worse than a random push of the same norm "
            f"({float(cell['margin_felt']):+.3f} against {float(cell['margin_random']):+.3f})"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0])
    chosen = choose(json.loads(path.read_text()), when=args[1] if len(args) > 1 else "")
    if chosen is None:
        print(json.dumps({"cell": None, "why": "no cell's margin is as good as a random push's"}))
        return 1
    print(json.dumps(chosen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
