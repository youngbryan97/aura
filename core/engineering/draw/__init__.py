"""Drawing a design: the sheet, the views, the symbols and the schematics.

Everything here reads a :class:`~core.engineering.model.Design` and a set of
:class:`~core.engineering.analysis.Finding` objects and returns SVG. Nothing
here computes an engineering result, and nothing here may put a number on a
drawing that did not come from a finding.
"""

from core.engineering.draw.canvas import Canvas, LINE_TYPES, THEMES, Theme
from core.engineering.draw.project import VIEWS, View, view_named
from core.engineering.draw.views import (
    Region,
    draw_assembly,
    draw_exploded,
    draw_orthographic,
    draw_section,
)

__all__ = [
    "Canvas",
    "LINE_TYPES",
    "THEMES",
    "Theme",
    "VIEWS",
    "View",
    "view_named",
    "Region",
    "draw_assembly",
    "draw_exploded",
    "draw_orthographic",
    "draw_section",
]
