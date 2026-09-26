#!/usr/bin/env python3
"""A small first-person room in a real window, for proving her camera loop live.

Not a game and not a world she learns anything general from: a proving ground
for the plumbing. Her eyes read it through the window server and Vision OCR,
her hands reach it through Quartz events, and nothing in it talks to her any
other way. The walls are a textured panorama that slides when the camera turns
and grows as she walks; labelled things stand at bearings and distances; a
prompt appears when she is close to one and facing it, and pressing what it
says answers it.

Keys: w or up walks, s or down backs away. The mouse turns the camera by how
far it moved, read from the event's delta the way a game reads it.

    python tools/a_room_to_walk_in.py --things "Door:25:9,Chest:-40:6"

Each thing is name:bearing-in-degrees:distance. The window's title is
"A room to walk in"; the log of what happened goes to stdout as JSON lines.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys

import AppKit
import objc
from Foundation import NSMakeRect, NSObject

FIELD = 90.0
PACE = 2.5          # distance a second of walking covers
TURN = 0.15         # degrees of turn per point of mouse delta
REACH = 1.8


def _log(**what: object) -> None:
    print(json.dumps(what), flush=True)


class Room:
    def __init__(self, things: list[tuple[str, float, float]]) -> None:
        self.facing = 0.0
        self.x = self.y = 0.0
        self.things = {
            name: (far * math.sin(math.radians(bearing)), far * math.cos(math.radians(bearing)))
            for name, bearing, far in things
        }
        self.opened: list[str] = []
        roll = random.Random(7)
        self.wall = [[roll.random() for _ in range(360)] for _ in range(12)]
        self.walking = 0.0

    def seen(self) -> list[tuple[str, float, float]]:
        """Each thing in view: name, bearing from the middle, distance."""
        found = []
        for name, (x, y) in self.things.items():
            dx, dy = x - self.x, y - self.y
            off = (math.degrees(math.atan2(dx, dy)) - self.facing + 180.0) % 360.0 - 180.0
            if abs(off) < FIELD / 2:
                found.append((name, off, math.hypot(dx, dy)))
        return found

    def prompt(self) -> str:
        for name, off, far in self.seen():
            if far < REACH and abs(off) < 12 and name not in self.opened:
                return f"Press E to open the {name}"
        return ""

    def step(self, seconds: float) -> None:
        if self.walking:
            pace = PACE * seconds * self.walking
            self.x += pace * math.sin(math.radians(self.facing))
            self.y += pace * math.cos(math.radians(self.facing))


class View(AppKit.NSView):
    def initWithFrame_room_(self, frame, room):
        self = objc.super(View, self).initWithFrame_(frame)
        if self is None:
            return None
        self.room = room
        return self

    def acceptsFirstResponder(self):
        return True

    def isFlipped(self):
        return True

    def drawRect_(self, rect):
        try:
            self._draw()
        except Exception as why:  # noqa: BLE001 - AppKit swallows what a draw raises; say it
            _log(event="could not draw", why=repr(why))

    @objc.python_method
    def _draw(self):
        room = self.room
        wide, high = self.bounds().size.width, self.bounds().size.height
        AppKit.NSColor.blackColor().set()
        AppKit.NSRectFill(self.bounds())
        # The walls: one column a degree, magnified by how far she has walked.
        nearest = min((far for _n, _o, far in room.seen()), default=10.0)
        grow = max(0.6, min(3.0, 8.0 / max(nearest, 1.0)))
        per_degree = wide / FIELD
        for column in range(int(-FIELD / 2) - 1, int(FIELD / 2) + 2):
            angle = int(room.facing + column) % 360
            x = wide / 2 + column * per_degree * grow
            if x < -per_degree * grow or x > wide:
                continue
            for row, band in enumerate(room.wall):
                shade = band[angle]
                AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.2 + 0.6 * shade, 1.0).set()
                top = high / 2 + (row - 6) * 18 * grow
                AppKit.NSRectFill(NSMakeRect(x, top, per_degree * grow + 1, 18 * grow + 1))
        # The things, as signs: a panel and the name, sized by distance.
        for name, off, far in room.seen():
            size = max(14.0, min(120.0, 180.0 / far))
            x = wide / 2 + off * per_degree
            AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.9, 0.9, 0.8, 1.0).set()
            AppKit.NSRectFill(NSMakeRect(x - size * 2, high / 2 - size, size * 4, size * 1.6))
            self._say(name, x, high / 2 - size * 0.6, size, centered=True)
        said = room.prompt()
        if said:
            AppKit.NSColor.whiteColor().set()
            AppKit.NSRectFill(NSMakeRect(wide / 2 - 220, high - 70, 440, 40))
            self._say(said, wide / 2, high - 64, 22, centered=True)
        for index, name in enumerate(room.opened):
            AppKit.NSColor.whiteColor().set()
            AppKit.NSRectFill(NSMakeRect(20, 20 + 34 * index, 300, 30))
            self._say(f"The {name} is open", 30, 24 + 34 * index, 18)

    @objc.python_method
    def _say(self, text, x, y, size, centered=False):
        font = AppKit.NSFont.boldSystemFontOfSize_(size)
        attrs = {AppKit.NSFontAttributeName: font, AppKit.NSForegroundColorAttributeName: AppKit.NSColor.blackColor()}
        said = AppKit.NSString.stringWithString_(text)
        wide = said.sizeWithAttributes_(attrs).width
        said.drawAtPoint_withAttributes_((x - wide / 2 if centered else x, y), attrs)

    def keyDown_(self, event):
        key = (event.charactersIgnoringModifiers() or "").lower()
        code = event.keyCode()
        if key == "w" or code == 126:
            self.room.walking = 1.0
        elif key == "s" or code == 125:
            self.room.walking = -1.0
        elif key == "e":
            said = self.room.prompt()
            if said:
                name = said.rsplit(" ", 1)[-1]
                self.room.opened.append(name)
                _log(event="opened", thing=name)
        _log(event="key", key=key or str(code), down=True)
        self.setNeedsDisplay_(True)

    def keyUp_(self, event):
        key = (event.charactersIgnoringModifiers() or "").lower()
        code = event.keyCode()
        if key in ("w", "s") or code in (125, 126):
            self.room.walking = 0.0
        self.setNeedsDisplay_(True)

    def mouseMoved_(self, event):
        self.room.facing += TURN * float(event.deltaX())
        self.setNeedsDisplay_(True)


class Ticker(NSObject):
    def initWithView_(self, view):
        self = objc.super(Ticker, self).init()
        self.view = view
        return self

    def tick_(self, _timer):
        before = (self.view.room.x, self.view.room.y)
        self.view.room.step(1.0 / 30.0)
        if (self.view.room.x, self.view.room.y) != before:
            self.view.setNeedsDisplay_(True)


def main() -> int:
    ask = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ask.add_argument("--things", default="Door:25:9,Chest:-40:6")
    said = ask.parse_args()
    things = []
    for part in said.things.split(","):
        name, bearing, far = part.split(":")
        things.append((name, float(bearing), float(far)))
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)
    window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(200, 200, 960, 600),
        AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable,
        AppKit.NSBackingStoreBuffered,
        False,
    )
    window.setTitle_("A room to walk in")
    window.setAcceptsMouseMovedEvents_(True)
    view = View.alloc().initWithFrame_room_(window.contentView().frame(), Room(things))
    window.setContentView_(view)
    window.makeFirstResponder_(view)
    window.makeKeyAndOrderFront_(None)
    ticker = Ticker.alloc().initWithView_(view)
    AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        1.0 / 30.0, ticker, "tick:", None, True
    )
    app.activateIgnoringOtherApps_(True)
    _log(event="ready", things=[name for name, _b, _f in things])
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
