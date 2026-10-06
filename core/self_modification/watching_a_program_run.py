"""What a program that draws does when it runs, watched the way she watches any game.

Reading code says where something looks wrong. Only running it says what is
wrong, and a repair is judged the same way: by whether the program now does
what it should. "What it should" comes from the program's own words (its
title screen says "up and down arrows move your paddle") and from what she
knows about the thing it is a version of, and it is measured with her own
eyes and hands (core/agency/playing_as_it_happens.py and the measurements
under it), not by reading the program's variables.

A run has three parts, each a small experiment:

1. Try the keys the words name. Which thing is hers, and does each key move
   it the way its name says?
2. Meet the things that move. A thing she gets in front of should turn back,
   not go through her.
3. Hold still and let them go by. When one gets past her, which counter
   changes? The words and what she knows say who a miss should count for.

Throughout: a thing that leaves the screen and leaves the game standing still,
with no counter changed and nothing moving, has escaped; a thing shaped like
hers on the other side that never moves while the play goes on is an actor
that does not act.

The page's randomness is fixed for the run, so two versions of a program are
watched through the same game.
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Behaviour", "what_it_does"]

#: Math.random replaced by a seeded generator before the page's own scripts run.
_SAME_DICE = """
(() => {
  let seed = 1234567;
  Math.random = function () {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
})();
"""

_DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


@dataclass
class Behaviour:
    """What was seen, as plain findings with their evidence."""

    started_by: str = ""
    hers: str = ""
    controls: dict[str, tuple[float, float]] = field(default_factory=dict)
    findings: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0
    #: The raw record the findings were read from, for whoever wants to check one.
    evidence: dict[str, Any] = field(default_factory=dict)

    #: What was checked and found as it should be. A finding that is absent is
    #: only good news when its check is here: unmeasured is never fine.
    right: set[str] = field(default_factory=set)

    #: The checks this kind of watch makes: a game's, or a page's.
    checks: set[str] = field(default_factory=set)

    @property
    def wrong(self) -> set[str]:
        return set(self.findings)


@dataclass
class _Watch:
    moves: Any
    hers: Any
    readouts: Any
    met: list[dict[str, Any]] = field(default_factory=list)
    went: list[dict[str, Any]] = field(default_factory=list)
    counter_changes: list[tuple[float, str, int]] = field(default_factory=list)
    last_moving: float = 0.0
    last_others_moving: float = 0.0
    screen_at: float = 0.0
    still_others: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    restarts: int = 0
    bounces: int = 0
    bounced_off: dict[str, int] = field(default_factory=dict)
    changes: Any = None
    were_hers: set[int] = field(default_factory=set)
    _heading: dict[int, tuple[float, float]] = field(default_factory=dict)
    frames: Any = None


async def _look(page: Any, clip: dict[str, float], frames: Any = None) -> tuple[Any, float] | None:
    from core.perception.picture_arithmetic import decode

    if frames is not None:
        seen = await frames.look(clip)
        if seen is not None:
            return seen
    try:
        data = await page.screenshot(clip=clip, type="jpeg", quality=80, scale="css")
    except Exception:  # noqa: BLE001 - a page that cannot be photographed ends the watch
        return None
    picture = decode(data)
    return (picture, time.monotonic()) if picture is not None else None


async def _start(page: Any, clip: dict[str, float], words: str) -> str:
    """Begin the program the way its screen asks: a key it names, or a click in its middle."""
    from core.agency.what_i_can_do_here import keys_a_screen_asks_for

    asked = keys_a_screen_asks_for(words)
    if asked:
        name = {"space": "Space", "return": "Enter"}.get(asked[0], asked[0])
        await page.keyboard.press(name)
        return f"pressed {asked[0]}, as the screen asked"
    await page.mouse.click(clip["x"] + clip["width"] / 2, clip["y"] + clip["height"] / 2)
    return "clicked in the middle"


async def _see(page: Any, clip: dict[str, float], watch: _Watch, keys_held: str, *, trying: bool) -> list[dict[str, Any]]:
    seen = await _look(page, clip, watch.frames)
    if seen is None:
        return []
    picture, at = seen
    happened = watch.moves.see(picture, at)
    watch.hers.holding(keys_held, at, trying=trying)
    watch.hers.saw(watch.moves, happened, at)
    moving = watch.moves.moving(faster_than=8.0)
    if moving:
        watch.last_moving = at
    if any(thing.number != watch.hers.number for thing in moving):
        watch.last_others_moving = at
    return happened


async def _read_counters(page: Any, clip: dict[str, float], watch: _Watch) -> tuple[str, ...]:
    """Read the screen's numbers; returns the keys its words ask for, if any."""
    from core.agency.what_i_can_do_here import keys_a_screen_asks_for
    from core.perception.what_the_pixels_show import recognize_text

    seen = await _look(page, clip, watch.frames)
    if seen is None:
        return ()
    picture, at = seen
    from core.perception.the_drawing_as_objects import words_in

    regions = words_in(picture)
    if regions is None:
        regions = await asyncio.to_thread(recognize_text, picture[:, :, ::-1].copy())
    before = dict(watch.readouts.values)
    watch.readouts.read(regions, at, None)
    for name, value in watch.readouts.values.items():
        if name in before and before[name] != value:
            watch.counter_changes.append((at, name, value - before[name]))
    return keys_a_screen_asks_for(" ".join(str(r.get("text") or "") for r in regions))


async def _try_keys(page: Any, clip: dict[str, float], watch: _Watch, keys: list[str]) -> None:
    """Hold each key and rest between, twice through, and again while nothing has answered.

    Each hold and each rest long enough to be measured once its first
    pictures, which still show the key before, are left out
    (core/agency/which_one_answers_to_her.py). Held for a third of a second
    on a fresh browser's slow first pictures, nothing was measured and a
    whole first watch said nothing was wrong (offline 2026-10-04).
    """
    for passes in range(TRY_KEYS_PASSES):
        if passes >= 2 and watch.hers.number is not None:
            return
        for key in keys:
            for held, how_long in ((key, HOLD_S), ("", HOLD_S)):
                if held:
                    await page.keyboard.down(_KEY.get(held, held))
                began = time.monotonic()
                while time.monotonic() - began < how_long:
                    await _see(page, clip, watch, held, trying=True)
                if held:
                    await page.keyboard.up(_KEY.get(held, held))


#: How long each key is held, and each rest between, while finding what answers.
HOLD_S = 0.5

#: The most times through the keys before watching without knowing which is hers.
TRY_KEYS_PASSES = 4

#: The least time between one trial of her keys to find her again and the next.
FIND_AGAIN_EVERY_S = 3.0


_KEY = {"up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight", "space": "Space"}


def _note_contacts(watch: _Watch, at: float) -> None:
    mine = watch.hers.thing(watch.moves)
    if mine is None:
        return
    for thing in watch.moves.things.values():
        if thing.number == mine.number or not thing.moved:
            continue
        # Met: touching, with its middle within her length. A ball that clips
        # her corner and goes by was never in front of her.
        long_way_y = mine.h >= mine.w
        within = abs(thing.y - mine.y) < mine.h / 2 if long_way_y else abs(thing.x - mine.x) < mine.w / 2
        near = within and abs(thing.x - mine.x) < (thing.w + mine.w) / 2 + 1 and abs(thing.y - mine.y) < (thing.h + mine.h) / 2 + 1
        if near and not any(m["thing"] == thing.number and at - m["at"] < 0.5 for m in watch.met):
            # Which third of her thing it met, along the way her thing is long.
            if mine.h >= mine.w:
                along = (thing.y - mine.y) / max(1.0, mine.h / 2)
            else:
                along = (thing.x - mine.x) / max(1.0, mine.w / 2)
            part = -1 if along < -0.33 else 1 if along > 0.33 else 0
            watch.met.append({"thing": thing.number, "at": at, "vx": thing.vx, "vy": thing.vy, "part": part})


def _note_departures(watch: _Watch, happened: list[dict[str, Any]], at: float) -> None:
    if any(event.get("what") == "new screen" for event in happened):
        return
    from core.agency.when_motion_breaks_a_rule import departure_edge
    tall, wide = watch.moves.shape
    mine = watch.hers.thing(watch.moves)
    for event in happened:
        if event.get("what") != "gone" or event.get("thing") == watch.hers.number:
            continue
        if math.hypot(event.get("vx", 0), event.get("vy", 0)) < 8:
            continue
        x, y = event["x"], event["y"]
        edge = departure_edge(event, watch.moves.shape)
        if edge:
            behind = mine is not None and ((edge == "left" and mine.x < wide / 2) or (edge == "right" and mine.x > wide / 2))
            watch.went.append({"at": at, "edge": edge, "behind_her": behind, "her_side": mine.x / max(1, wide) if mine is not None else None})


async def _play(page: Any, clip: dict[str, float], watch: _Watch, keys: list[str], seconds: float, *, still: bool) -> None:
    from core.agency.playing_as_it_happens import _Choosing
    from core.agency.what_meeting_things_does import WhatMeetingDoes

    meeting = WhatMeetingDoes()
    held = ""
    began = time.monotonic()
    counted = found_again_at = 0.0
    while time.monotonic() - began < seconds:
        happened = await _see(page, clip, watch, held, trying=False)
        at = time.monotonic()
        _note_contacts(watch, at)
        _note_departures(watch, happened, at)
        _note_still_others(watch, at)
        _note_bounces(watch)
        if any(h.get("what") == "new screen" for h in happened):
            watch.screen_at = at
        _note_changes_in_place(watch, happened, at)
        if at - counted > 0.5:
            counted = at
            asks = await _read_counters(page, clip, watch)
            # A screen standing still that asks for a key is a game over or a
            # pause: pressed, the watch goes on watching play.
            if asks and at - watch.last_others_moving > 1.5:
                await page.keyboard.press({"space": "Space", "return": "Enter"}.get(asks[0], asks[0]))
                watch.restarts += 1
        if watch.hers.lost() and at - watch.last_others_moving < 0.5 and at - found_again_at >= FIND_AGAIN_EVERY_S:
            # Hidden and drawn afresh (an end screen, a game begun again), her
            # thing is found as it was at the start: by trying her keys. Every
            # check but one needs to know which thing is hers.
            if held:
                await page.keyboard.up(_KEY.get(held, held))
                held = ""
            await _try_keys(page, clip, watch, keys)
            found_again_at = time.monotonic()
            continue
        if still:
            continue
        choosing = _Choosing(watch.moves, watch.hers, meeting, keys)
        if choosing.mine is None or not choosing.ways:
            continue
        # A tester meets things with the ends as well as the middle: a ball
        # sent off at a steep angle is what tests the walls.
        end = (len(watch.met) % 3 - 1) * 0.4 * (choosing.mine.h if choosing.line() == 0 else choosing.mine.w)
        key, _why, _aim = choosing.key(held, offset=(0.0, end) if choosing.line() == 0 else (end, 0.0))
        if key != held:
            if held:
                await page.keyboard.up(_KEY.get(held, held))
            if key:
                await page.keyboard.down(_KEY.get(key, key))
            held = key
    if held:
        await page.keyboard.up(_KEY.get(held, held))


def _note_changes_in_place(watch: _Watch, happened: list[dict[str, Any]], at: float) -> None:
    """Somewhere still that changed and then stayed changed is writing that changed: a counter.

    Text recognition misses a lone digit (a "0" on a black court is read as
    nothing at any size), so a counter cannot always be read. That it changed,
    and where, is seen without reading it, by the same finder her play uses
    (core/perception/what_changed_and_stayed.py). This had its own copy, which
    kept out the place of anything that had ever moved, and a score digit
    going from 1 to 2 moves its own middle: the score was kept out of its own
    counting (offline 2026-10-04, three misses and one change seen).
    """
    from core.perception.what_changed_and_stayed import WhatChangedAndStayed

    if any(h.get("what") == "new screen" for h in happened):
        watch.changes = None
        return
    if watch.changes is None:
        watch.changes = WhatChangedAndStayed()
    picture = getattr(watch.moves, "_last", None)
    for when, x, y in watch.changes.see(picture, watch.moves.things, at, never=set(watch.were_hers)):
        name = f"changed at {x:.2f},{y:.1f}"
        watch.readouts.where[name] = (x, y)
        watch.counter_changes.append((when, name, 1))


def _note_bounces(watch: _Watch) -> None:
    """A moving thing turned back by the top or bottom of the picture itself: that wall holds.

    Turned back with nothing else near it, and still going the same way
    across. A ball sent back by a paddle that happens to be near the top is
    the paddle's doing, and taking it for the wall's once let a missing wall
    be believed present (offline 2026-10-04). Nothing else means anything,
    still or not, and the thing at its own size: LIVE 2026-10-04 a ball
    passing the top dash of the net drew as one blob with it, the blob's
    middle jumped down, and a court with no top wall was said to have one.
    """
    import statistics

    tall, _wide = watch.moves.shape
    others = list(watch.moves.things.values())
    for thing in watch.moves.things.values():
        if not thing.moved or thing.number in watch.were_hers:
            continue
        before = watch._heading.get(thing.number)
        watch._heading[thing.number] = (thing.vx, thing.vy)
        if before is None:
            continue
        bvx, bvy = before
        wall = "top" if thing.y < tall * 0.15 else "bottom" if thing.y > tall * 0.85 else ""
        flipped = bvy * thing.vy < 0 and abs(bvy) > 20 and abs(thing.vy) > 20
        same_across = bvx * thing.vx > 0 or (abs(bvx) < 5 and abs(thing.vx) < 5)
        alone = all(
            other.number == thing.number
            or abs(other.x - thing.x) > (other.w + thing.w) / 2 + 8
            or abs(other.y - thing.y) > (other.h + thing.h) / 2 + 8
            for other in others
        )
        usual = statistics.median(thing.sizes) if thing.sizes else thing.size
        its_own_size = max(thing.size, usual) / max(1.0, min(thing.size, usual)) < 1.5
        if wall and flipped and same_across and alone and its_own_size:
            watch.bounces += 1
            watch.bounced_off[wall] = watch.bounced_off.get(wall, 0) + 1


def _note_still_others(watch: _Watch, at: float) -> None:
    """Things shaped like hers, by width and height, and how fast each is going."""
    mine = watch.hers.thing(watch.moves)
    if mine is None:
        return
    watch.were_hers.add(mine.number)
    for thing in watch.moves.things.values():
        if thing.number in watch.were_hers:
            continue
        alike = abs(math.log(max(1.0, thing.w) / max(1.0, mine.w))) < 0.4 and abs(math.log(max(1.0, thing.h) / max(1.0, mine.h))) < 0.4
        # On the other half from her: where the other side's player would stand.
        across = (thing.x < watch.moves.shape[1] / 2) != (mine.x < watch.moves.shape[1] / 2)
        if alike and across:
            watch.still_others.setdefault(thing.number, []).append((at, thing.x, thing.y))


#: A verdict on one check: what it found, and whether it was wrong, right or not measured.
WRONG, RIGHT, UNMEASURED = "wrong", "right", "unmeasured"


def _controls(watch: _Watch, keys: list[str]) -> tuple[dict[str, tuple[float, float]], tuple[str, str]]:
    """Every direction key the words name moves her thing that way, or the controls are wrong."""
    ways: dict[str, tuple[float, float]] = {}
    named = [key for key in keys if key in _DIRECTIONS]
    if watch.hers.kind is None:
        # Nothing found answering is not a finding that nothing answers: on a
        # loaded machine a watch at ten pictures a second misses her thing
        # now and then, and read as "no key moves anything" that turned down
        # right edits as harm (LIVE 2026-10-04).
        return ways, (UNMEASURED, "")
    for key in named:
        if watch.hers.tried(key) < 4:
            return ways, (UNMEASURED, "")
        way = watch.hers.way_of(key) or (0.0, 0.0)
        ways[key] = way
        meant = _DIRECTIONS[key]
        along = way[0] * meant[0] + way[1] * meant[1]
        if along < -10:
            return ways, (WRONG, f"the {key} key moves {_mine(watch)} the other way")
    moved = [key for key in named if math.hypot(*ways.get(key, (0.0, 0.0))) > 10]
    if named and not moved:
        return ways, (UNMEASURED, "")
    for key in named:
        if key not in moved and any(_DIRECTIONS[k][0] == -_DIRECTIONS[key][0] and _DIRECTIONS[k][1] == -_DIRECTIONS[key][1] for k in moved):
            return ways, (WRONG, f"the {key} key does nothing")
    return ways, (RIGHT, "")


def _went_through(watch: _Watch) -> tuple[str, str]:
    """Things she got in front of turned back, all along her, or some part of her let them through.

    Right only where every third of her was met and none let a thing through:
    a paddle that only its top tenth stops is right every time it happens to
    be met there (offline 2026-10-04, and that belief turned down the right
    repair of the controls).
    """
    if len(watch.met) < 2:
        return UNMEASURED, ""
    by_part: dict[int, list[int]] = {}
    for meeting in watch.met:
        through = any(0 <= m["at"] - meeting["at"] < 1.5 and m["behind_her"] for m in watch.went)
        counts = by_part.setdefault(meeting.get("part", 0), [0, 0])
        counts[0] += 1
        counts[1] += int(through)
    through_all = sum(c[1] for c in by_part.values())
    if through_all >= 2 and any(c[1] and c[1] * 2 >= c[0] for c in by_part.values()):
        return WRONG, f"of {len(watch.met)} things I got in front of, {through_all} went straight through me"
    if set(by_part) == {-1, 0, 1} and through_all == 0:
        return RIGHT, ""
    return UNMEASURED, ""


def _mine(watch: _Watch) -> str:
    """What she controls, as she sees it: "my white bar", or "what I control" before she knows."""
    from core.agency.playing_as_it_happens import describe

    mine = watch.hers.thing(watch.moves)
    return f"my {describe(watch.moves, mine.kind, mine)}" if mine is not None else "what I control"


def _escaped(watch: _Watch) -> tuple[str, str]:
    """Things stay in play: a thing left through the top or bottom and the game stood still after it."""
    for departure in watch.went:
        if departure["edge"] not in ("top", "bottom"):
            continue
        changed = any(0 <= at - departure["at"] < 2.0 for at, _n, _d in watch.counter_changes)
        if changed:
            return WRONG, f"something left through the {departure['edge']} and a counter changed instead of it turning back"
        if watch.last_others_moving - departure["at"] < 1.0:
            return WRONG, f"something left through the {departure['edge']} and the game stood still after it"
    # Both walls seen to turn things back: one wall that works says nothing
    # about the other, and believing it did kept a missing wall from being
    # looked for (offline 2026-10-04).
    if watch.bounced_off.get("top") and watch.bounced_off.get("bottom"):
        return RIGHT, ""
    return UNMEASURED, ""


def _credited(watch: _Watch) -> tuple[str, str]:
    """When a thing got past her, the counter that went up was on her half, or on the other.

    Each time something got past her, the counter change nearest it in time
    is the one it caused; and a verdict needs two such times that agree, more
    than disagree. One reading decided it before, and a score misread once
    turned the right edit of two mirrored score lines down and kept the wrong
    one (LIVE 2026-10-04).
    """
    # A score is written again and again in one place. A place that changed
    # once is as likely the end screen's writing appearing, or a stray: a
    # miss credited to a one-off change in mid-court made a game whose
    # misses scored for her read as right (offline 2026-10-04).
    times: dict[str, int] = {}
    for _at, name, _delta in watch.counter_changes:
        times[name] = times.get(name, 0) + 1
    counters = {name for name, n in times.items() if n >= 2 or name.startswith("number")}
    hers_went_up = others_went_up = 0
    for departure in watch.went:
        her_side = departure.get("her_side")
        if not departure["behind_her"] or her_side is None:
            continue
        # The score is redrawn as the ball leaves, and the leaving is only
        # called a quarter second later, so the change can come first.
        near = [
            (abs(at - departure["at"]), where)
            for at, name, delta in watch.counter_changes
            if delta > 0 and name in counters and (where := watch.readouts.where.get(name)) is not None
            and -0.5 <= at - departure["at"] < 2.0
        ]
        if not near:
            continue
        where = min(near, key=lambda pair: pair[0])[1]
        if (where[0] < 0.5) == (her_side < 0.5):
            hers_went_up += 1
        else:
            others_went_up += 1
    if hers_went_up >= 2 and hers_went_up > others_went_up:
        return WRONG, "when something got past me, the score on my side went up"
    if others_went_up >= 2 and others_went_up > hers_went_up:
        return RIGHT, ""
    return UNMEASURED, ""


def _idle_actor(watch: _Watch) -> tuple[str, str]:
    """Another thing shaped like hers moves while play goes on, or it never does."""
    seen = [samples for number, samples in watch.still_others.items() if len(samples) >= 60 and number not in watch.were_hers]
    if not seen:
        return UNMEASURED, ""

    def went_nowhere(samples: list[tuple[float, float, float]]) -> bool:
        # Where it was, not how fast it seemed: a ball passing over it merges
        # with it for a picture and its centre jumps, though it never moved.
        for axis in (1, 2):
            places = sorted(sample[axis] for sample in samples)
            low, high = places[len(places) // 20], places[-1 - len(places) // 20]
            if high - low > 4.0:
                return False
        return True

    if all(went_nowhere(samples) for samples in seen):
        return WRONG, "the other one shaped like mine never moved"
    return RIGHT, ""


async def _used(page: Any, behaviour: Behaviour, failed: list[str]) -> Behaviour:
    """A page of controls, used once each (what_a_page_does.py), read as the same three-valued checks."""
    from core.self_modification.what_a_page_does import what_a_page_does

    began = time.monotonic()
    behaviour.checks = {"errors", "dead", "failed"}
    use = await what_a_page_does(page, before=behaviour.errors, failed=failed)
    if use.errors:
        behaviour.findings["errors"] = f"the page threw an error: {use.errors[0]}"
    elif use.used:
        behaviour.right.add("errors")
    if use.dead:
        behaviour.findings["dead"] = f"{len(use.dead)} of {len(use.used)} controls changed nothing when used: {', '.join(use.dead[:4])}"
    elif use.used:
        behaviour.right.add("dead")
    if use.failed:
        behaviour.findings["failed"] = f"it could not load {use.failed[0].rsplit('/', 1)[-1]}"
    else:
        behaviour.right.add("failed")
    behaviour.seconds = round(time.monotonic() - began, 1)
    behaviour.evidence = {"used": use.used, "dead": use.dead, "errors": use.errors[:5], "failed": use.failed[:5],
                          "right": sorted(behaviour.right), "pictures": 0, "pictures_a_second": 0, "hers": ""}
    return behaviour


async def what_it_does(page: Any, address: str, *, words: str, keys: list[str], seconds: float = 12.0) -> Behaviour:
    """Load ``address`` in ``page`` with fixed randomness and watch what it does."""
    from core.agency.what_meeting_things_does import Readouts
    from core.agency.which_one_answers_to_her import WhichIsHers
    from core.perception.what_moves_in_the_picture import WhatMoves
    from core.perception.frames_as_they_are_drawn import CanvasFrames
    from core.perception.the_drawing_as_objects import BOOTSTRAP

    behaviour = Behaviour()
    page.on("pageerror", lambda error: behaviour.errors.append(str(error)[:200]))
    failed: list[str] = []
    folder = address.rsplit("/", 1)[0]
    page.on("requestfailed", lambda request: failed.append(request.url) if request.url.startswith(folder) else None)
    page.on("response", lambda response: failed.append(response.url) if response.status >= 400 and response.url.startswith(folder) else None)
    await page.add_init_script(_SAME_DICE)
    await page.add_init_script(BOOTSTRAP)
    await page.goto(address)
    await page.wait_for_timeout(300)
    clip = await page.evaluate(
        "(() => { const c = document.querySelector('canvas'); if (!c) return null;"
        " const r = c.getBoundingClientRect(); return {x: r.left, y: r.top, width: r.width, height: r.height}; })()"
    )
    if not clip:
        # Not a moving picture: a page of controls, judged by using them.
        return await _used(page, behaviour, failed)
    began = time.monotonic()
    behaviour.started_by = await _start(page, clip, words)
    watch = _Watch(moves=WhatMoves(), hers=WhichIsHers(), readouts=Readouts(), frames=CanvasFrames(page))
    try:
        await _try_keys(page, clip, watch, keys)
        await _play(page, clip, watch, keys, seconds, still=False)
        await _play(page, clip, watch, keys, seconds, still=True)
    finally:
        await watch.frames.close()
    behaviour.controls, controls = _controls(watch, keys)
    behaviour.checks = {"controls", "went through", "escaped", "credited", "idle"}
    for name, (verdict, finding) in (("controls", controls), ("went through", _went_through(watch)),
                                     ("escaped", _escaped(watch)), ("credited", _credited(watch)),
                                     ("idle", _idle_actor(watch))):
        if verdict == WRONG:
            behaviour.findings[name] = finding
        elif verdict == RIGHT:
            behaviour.right.add(name)
    if behaviour.errors:
        behaviour.findings["errors"] = behaviour.errors[0]
    mine = watch.hers.thing(watch.moves)
    if mine is not None:
        from core.agency.playing_as_it_happens import describe

        behaviour.hers = describe(watch.moves, mine.kind, mine)
    behaviour.seconds = round(time.monotonic() - began, 1)
    behaviour.evidence = {
        "met": len(watch.met),
        "went": [(round(d["at"] - began, 1), d["edge"], d["behind_her"]) for d in watch.went],
        "counters": [(round(at - began, 1), name, delta) for at, name, delta in watch.counter_changes],
        "where": dict(watch.readouts.where),
        "values": dict(watch.readouts.values),
        "others_last_moved": round(watch.last_others_moving - began, 1),
        "restarts": watch.restarts,
        "bounces": dict(watch.bounced_off),
        "alike_across": {n: len(v) for n, v in watch.still_others.items()},
        "were_hers": sorted(watch.were_hers),
        "right": sorted(behaviour.right),
        "pictures": watch.moves.pictures,
        "pictures_a_second": round(watch.moves.pictures / max(0.1, behaviour.seconds), 1),
        "hers": behaviour.hers,
        "observation": "canvas and pixels",
    }
    return behaviour
