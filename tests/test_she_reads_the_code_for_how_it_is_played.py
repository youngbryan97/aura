"""She reads the program a thing runs for how it is played: its lesson whole, and a player's manual from its code.

LIVE 2026-10-10 a game's program held its whole lesson as twenty-nine lines ("click here to open", "THE device
Library", "choose the", "type of device"), and each line was judged alone: six were kept. Its code, its screens by name
(pickaroom, build, settrap, success, fail) and its parts (gridHighlight, testPoint, deviceMenu) were read for key codes
and nothing else.
"""
from __future__ import annotations

import asyncio

import pytest

from core.cognition.a_guide_to_a_place import Guide
from core.cognition.reading_the_code import read_the_code_beside, take_in_the_code, the_telling_part
from core.perception.reading_a_program import ProgramRead, _runs

pytestmark = pytest.mark.unit

LINES = ["SWITCH ROOMS", "SWITCH ROOMS", "TEST TRAP", "P", "i", "c", "k", "the trap-o-matic", "starts with this",
         "mouse trap...", "...AND toM NEEDS", "YOUR HELP TO buILD a", "Trap to the cage that", "WiLL catch Jerry!",
         "click here to open", "THE device Library", "choose the", "type of device", "you wish to use",
         "CONTINUE ADDING", "Additional DEVICES...", "...UNTIL YOU", "are ready to", "test the trap!"]


def test_a_programs_lines_are_read_as_its_user_reads_them():
    runs = _runs(LINES)
    assert "click here to open THE device Library choose the type of device you wish to use CONTINUE ADDING " \
           "Additional DEVICES … UNTIL YOU are ready to test the trap!" in runs
    assert any(r.endswith("mouse trap … AND toM NEEDS YOUR HELP TO buILD a Trap to the cage that WiLL catch Jerry!")
               for r in runs)
    assert not any(" P i c k " in r for r in runs)                              # a title drawn a letter at a time


def _a_program() -> ProgramRead:
    code = ["// frame script in sprite 7", "stop()", "// frame script in the main timeline",
            "if !(this.counter > 10) goto +20", "trap1.gotoAndPlay(2)", "function addDevice(id, xOffset, yOffset) {",
            "if (cheatMode) gotoLabel('success')", "stop()"]
    return ProgramRead(words=["click here to open THE device Library"], labels=["pickaroom", "build", "success", "fail"],
                       names=["gridHighlight", "testPoint", "i_0", "i_1"], code=code)


def test_the_telling_part_holds_its_screens_its_parts_and_what_decides_things_and_no_cheat():
    told = the_telling_part(_a_program())
    assert "Its screens, in order: pickaroom, build, success, fail" in told
    assert "gridHighlight" in told and "i_0" not in told
    assert "if !(this.counter > 10) goto +20" in told and "function addDevice" in told
    assert "cheat" not in told.lower() and "stop()" not in told
    assert the_telling_part(ProgramRead(words=["x"])) == ""                     # a read kept without its code


def test_a_manual_from_its_code_goes_into_the_guide_and_is_kept_and_spoils_nothing():
    asked: list[str] = []

    async def ask(prompt, schema, most):
        asked.append(prompt)
        return schema.model_validate({
            "goal": "Build a chain of devices from the mouse trap to the cage that catches Jerry.",
            "win": "the chain's last device reaches the cage", "lose": "a device's path ends short of the cage",
            "controls": [{"control": "the pointer", "does": "drag a device from the library onto a lit square"},
                         {"control": "the debug key", "does": "skip to the success screen"}],
            "steps": ["Open the device library", "Drag a device onto a lit square", "Turn it so its arrow points on",
                      "Test the trap"],
            "progression": ["the kitchen, then the living room"],
            "tools": [{"name": "anvil", "does": "falls straight down"}],
            "hidden": ["a device only connects at the square its neighbour's arrow points to"],
            "watch_for": ["where each device's arrow points"]})

    guide = Guide(place="a place of devices")

    async def run():
        assert read_the_code_beside(guide, _a_program(), ask)
        await guide.reading_the_code

    asyncio.run(run())
    assert "pickaroom" in asked[0] and "Leave out cheats" in asked[0]
    assert guide.goals and guide.win == ["the chain's last device reaches the cage"]
    thinking = guide.for_thinking()
    assert "How to play it: Open the device library → Drag a device onto a lit square" in thinking
    assert "What its screens don't say: a device only connects" in thinking and "debug" not in thinking
    again = Guide(place="a place of devices")                                    # understood once, at once again
    assert read_the_code_beside(again, ProgramRead(), None)
    assert again.from_its_code == guide.from_its_code


def test_what_is_said_of_a_manual_names_how_it_is_won_and_what_the_screens_do_not_say():
    said = take_in_the_code(Guide(place="p"), {"win": "Every crate is on a marked square.",
                                               "hidden": ["A crate pushed into a corner cannot be moved again"]})
    assert said == ["From its code: it's won when every crate is on a marked square; what its screens don't say: "
                    "a crate pushed into a corner cannot be moved again."]
