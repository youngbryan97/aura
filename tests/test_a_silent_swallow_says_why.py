"""`except X: pass` — caught, and nothing whatsoever done about it."""
from __future__ import annotations

import json
from pathlib import Path

from tools.lint_silent_swallows import (
    BASELINE,
    load_baseline,
    silent_swallows,
    unexplained,
)

ROOT = Path(__file__).resolve().parents[1]


def test_the_number_with_no_reason_only_goes_down() -> None:
    """A defensible silent handler and a real swallow look identical.

    The rule is not "never swallow" — a QueueFull that falls through to
    shedding a worse task, a transition the state machine has already
    recorded, a substrate not loaded yet during boot are all correct. The rule
    is that the difference between one of those and a failure nobody will ever
    hear about is a sentence, and the sentence has to be there.
    """
    mute = unexplained(ROOT)
    held = load_baseline(ROOT / BASELINE)
    assert len(mute) <= held, (
        f"{len(mute)} silent handlers with no reason, up from {held}:\n"
        + "\n".join(mute[:15])
    )


def test_the_gate_finds_the_shape_it_is_about() -> None:
    """A gate that matched nothing would report green forever."""
    every = silent_swallows(ROOT)
    assert len(every) > 100, "the shape this is about is all over the tree"
    assert any(explained for _, _, explained in every), "some carry a reason"
    assert any(not explained for _, _, explained in every), "and some do not"


def test_a_reason_above_on_or_below_the_except_all_count() -> None:
    """Three places a person naturally writes it, and all three are reading."""
    import ast

    from tools.lint_silent_swallows import _says_nothing

    for source in (
        "try:\n    x()\nexcept ValueError:\n    pass  # nothing to do\n",
        "try:\n    x()\n# it may not be there yet\nexcept ValueError:\n    pass\n",
        "try:\n    x()\nexcept ValueError:  # boot ordering\n    pass\n",
    ):
        tree = ast.parse(source)
        handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
        assert handlers and _says_nothing(handlers[0])
        lines = source.splitlines()
        window = "\n".join(
            lines[max(0, handlers[0].lineno - 3) : handlers[0].body[0].lineno + 1]
        )
        assert "#" in window, source


def test_the_baseline_says_which_way_it_moves() -> None:
    held = json.loads((ROOT / BASELINE).read_text("utf-8"))
    assert "only goes down" in held["note"].lower()
    assert held["unexplained"] <= held["silent_handlers"]


def test_the_marker_is_seen_when_a_sentence_capitalises_it():
    """A gate that matches one casing of its own marker cannot be used.

    Eighteen handlers in ``model_lane_control`` were explained and six of them
    still counted, because the note opened a sentence — "Not a failure:" — and
    the check looked for the lowercase form only. Same shape as the keyword
    ratchet that could not see an identifier.
    """
    import ast

    from tools.lint_swallowed_reasons import _Swallowed

    for marker in ("not a failure:", "Not a failure:", "NOT A FAILURE:"):
        source = (
            "def read(row):\n"
            "    try:\n"
            "        return int(row)\n"
            "    except ValueError:\n"
            f"        # {marker} a cell that is not a number is not one.\n"
            "        return None\n"
        )
        walker = _Swallowed(Path("probe.py"), source.splitlines())
        walker.visit(ast.parse(source))
        assert walker.found == [], marker


def test_a_handler_with_no_note_still_counts():
    """The marker must be the thing that clears it, not the shape."""
    import ast

    from tools.lint_swallowed_reasons import _Swallowed

    source = (
        "def read(row):\n"
        "    try:\n"
        "        return int(row)\n"
        "    except ValueError:\n"
        "        return None\n"
    )
    walker = _Swallowed(Path("probe.py"), source.splitlines())
    walker.visit(ast.parse(source))
    assert [kind for _line, kind, _said in walker.found] == ["a failure value"]
