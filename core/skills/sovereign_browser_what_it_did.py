"""What her last move did to the page, beside what she said it would do.

A person who presses "more" watches four lines open under it. The pursuit showed
her the whole page again each round and nothing of the difference, so on her own
result (LIVE 2026-10-02) she pressed four "more" links, was shown the page with
her four scores open, said she still had to expand it, and pressed "less". The
pair took twenty minutes. Every decision asks her what a move should do; nothing
ever showed her what it did.

Facts only, and nothing about what kind of page it is: which controls she used,
what she expected, and which lines of the page's text and which controls came or
went. What that means is hers.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _lines(observation: Mapping[str, Any]) -> list[str]:
    """The page's text, a line at a time, each line once and in order."""
    seen: set[str] = set()
    lines: list[str] = []
    for raw in str(observation.get("text") or "").splitlines():
        line = " ".join(raw.split())
        if line and line not in seen:
            seen.add(line)
            lines.append(line)
    return lines


def _controls(observation: Mapping[str, Any]) -> dict[str, str]:
    """Each control by its selector, as she would read it in the list."""
    held: dict[str, str] = {}
    for element in observation.get("elements") or []:
        if not isinstance(element, Mapping) or not element.get("selector"):
            continue
        name = str(element.get("name") or "").strip() or str(element.get("role") or "")
        held[str(element["selector"])] = (
            f"{name} (checked)" if element.get("checked") is True else name
        )
    return held


def names_of_the_moves(moves: Iterable[tuple[Any, Any]], elements: Sequence[Any]) -> list[str]:
    """What she did, named as the controls were named to her."""
    named = {
        str(element.get("selector") or ""): str(element.get("name") or "").strip()
        for element in elements
        if isinstance(element, Mapping)
    }
    done: list[str] = []
    for action, _said in moves:
        kind = str(getattr(action, "type", "") or "")
        target = named.get(str(getattr(action, "selector", "") or "")) or "a control"
        if kind == "type":
            done.append(f'typed "{getattr(action, "value", "")}" into {target}')
        elif kind == "scroll":
            done.append(f"scrolled {getattr(action, 'value', '') or 'down'}")
        else:
            done.append(f"pressed {target}")
    return done


def what_the_move_did(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    done: Sequence[str],
    expected: str = "",
) -> str:
    """Her last move, what she expected of it, and what it changed."""
    if not before or not after or not done:
        return ""
    said = f"YOUR LAST MOVE: you {', '.join(done)}"
    expected = " ".join(str(expected or "").split())
    said = f'{said}, expecting: "{expected}".' if expected else f"{said}."
    if str(before.get("url") or "") != str(after.get("url") or ""):
        return f"{said}\nIt took you from {before.get('url')} to {after.get('url')}."

    old, new = _lines(before), _lines(after)
    old_set, new_set = set(old), set(new)
    appeared = [line for line in new if line not in old_set]
    gone = [line for line in old if line not in new_set]
    was, now = _controls(before), _controls(after)
    controls = [
        f"{now[key]} (was {was[key]})" if key in was else f"{now[key]} (new)"
        for key in now
        if was.get(key) != now[key]
    ] + [f"{was[key]} (gone)" for key in was if key not in now]

    if not appeared and not gone and not controls:
        return f"{said}\nNothing on the page changed."
    # No line survived: the page was replaced, and the page below already says
    # what is there now. Listing both would be the old page and the new one again.
    if old and new and not old_set & new_set:
        return (
            f"{said}\n{len(appeared)} line(s) of the page's text appeared and "
            f"{len(gone)} went away; the page below is what is there now."
        )
    told = [said]
    if appeared:
        told.append("These lines appeared:\n" + "\n".join(f"+ {line}" for line in appeared))
    if gone:
        told.append("These lines went away:\n" + "\n".join(f"- {line}" for line in gone))
    if controls:
        told.append("Controls that changed: " + "; ".join(controls))
    if not appeared and not gone:
        told.append("The page's text did not change.")
    return "\n".join(told)


def every_way_on_led_back(observation: Mapping[str, Any], went_nowhere: set[str]) -> bool:
    """Whether every control on the page has already led back to a state she has seen.

    Then nothing here leads anywhere new, and a round spent on it is a round
    without progress, whatever the page does when pressed. The stall counter
    compares each page with the one before, so a pair of controls that undo
    each other changes the page every round and never trips it.
    """
    selectors = [
        str((element or {}).get("selector") or "")
        for element in observation.get("elements") or []
    ]
    return bool(went_nowhere) and bool(selectors) and all(
        selector in went_nowhere for selector in selectors
    )


def the_ones_tried_here(tried_from: set[tuple[str, str]], signature: str) -> set[str]:
    """The controls already used from this exact page state.

    A page answers the same press the same way, so pressing one again from a
    state it was pressed in can only lead where it led before. LIVE 2026-10-02:
    with her scores open she pressed "less", was back on the closed page, and
    pressed "more" again from there; the opened page was one she had already
    read, and the round, two and a half minutes of her model, showed her
    nothing new. Retiring a control only once it led back took four rounds to
    close a pair that two rounds had already explored.
    """
    return {selector for state, selector in tried_from if state == signature}


def remember_what_was_seen(seen: dict[str, list[str]], observation: Mapping[str, Any]) -> None:
    """Keep every line of text each page has shown in this run."""
    url = str(observation.get("url") or "")
    kept = seen.setdefault(url, [])
    known = set(kept)
    kept.extend(line for line in _lines(observation) if line not in known)


def with_what_was_seen_here(
    observation: Mapping[str, Any], seen: Mapping[str, list[str]]
) -> Mapping[str, Any]:
    """The page as it is, and what it showed earlier in this run that it hides now.

    The verdict reads the page as the run left it. Where her own moves opened a
    section and closed it again, what she read there was missing from the page
    she was asked to judge.
    """
    now = set(_lines(observation))
    hidden = [
        line for line in seen.get(str(observation.get("url") or ""), []) if line not in now
    ]
    if not hidden:
        return observation
    shown = dict(observation)
    shown["text"] = (
        f"{observation.get('text') or ''}\n\n"
        "(Shown on this page earlier in this run, hidden now:)\n" + "\n".join(hidden)
    )
    return shown
