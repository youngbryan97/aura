"""The OS handed her the layout and it was thrown away one line before use.

macOS Vision computes the position of every recognized text run: each
VNRecognizedTextObservation carries a boundingBox. ``_ocr_image_text`` read
``.string()`` off the top candidate and discarded the geometry, returning
``"\\n".join(lines)``.

So the flat text a caller received was not what the OS produced — it was what
survived. Anything laid out in two dimensions arrived as a column of strings in
reading order: a table lost its columns, a form lost which label belonged to
which field, a grid lost the grid. Any task that needs to know WHERE something
is was unreachable, and the cause was invisible because OCR appeared to be
working perfectly.

This was never a missing capability. It was a dropped field.
"""
from __future__ import annotations

import pytest

from core.capabilities.host_automation import AutomationReceipt


def test_a_perception_receipt_can_carry_layout():
    """`result` is the words; `layout` is the same reading with geometry."""
    receipt = AutomationReceipt(
        action="get_screen_text",
        target="",
        adapter="ocr",
        success=True,
        result="Score 2048",
        layout=[{"text": "Score", "x": 0.1, "y": 0.2}],
    )

    assert receipt.layout[0]["text"] == "Score"


def test_layout_defaults_to_empty_rather_than_none():
    """Callers iterate it; None would make every reader defensive."""
    receipt = AutomationReceipt(
        action="x", target="", adapter="ocr", success=True, result=""
    )

    assert receipt.layout == []


def test_two_receipts_do_not_share_one_layout_list():
    """A mutable default would leak one screen's geometry into the next."""
    first = AutomationReceipt(action="a", target="", adapter="ocr", success=True)
    second = AutomationReceipt(action="b", target="", adapter="ocr", success=True)

    first.layout.append({"text": "only mine"})

    assert second.layout == []


def test_the_reader_returns_a_list_for_an_unreadable_image(tmp_path):
    """Losing layout must never mean losing the words, or raising."""
    from core.capabilities.host_automation import HostAutomationProvider as HostAutomation

    missing = tmp_path / "not_an_image.png"

    assert HostAutomation._ocr_image_regions(str(missing)) == []


def test_regions_are_read_in_the_same_pass_as_the_text():
    """A second screenshot plus a second OCR is ~2s, which kills a watch loop."""
    import inspect

    from core.capabilities.host_automation import HostAutomationProvider as HostAutomation

    source = inspect.getsource(HostAutomation.get_screen_text)

    assert "_ocr_image_regions" in source
    assert source.count("take_screenshot") == 1


@pytest.mark.parametrize(
    "field_name", ["x", "y", "width", "height", "center_x", "center_y", "confidence"]
)
def test_a_region_carries_everything_a_click_needs(field_name):
    """Enough to point at the thing, not merely to know it exists."""
    import inspect

    from core.capabilities.host_automation import HostAutomationProvider as HostAutomation

    source = inspect.getsource(HostAutomation._ocr_image_regions)

    assert f'"{field_name}"' in source


def test_the_origin_is_converted_to_top_left():
    """Vision measures up from the bottom; everything else measures down.

    Leaving that mismatch for each caller to remember is how a click lands at
    the wrong end of the screen.
    """
    import inspect

    from core.capabilities.host_automation import HostAutomationProvider as HostAutomation

    source = inspect.getsource(HostAutomation._ocr_image_regions)

    assert "1.0 - (y + h)" in source
