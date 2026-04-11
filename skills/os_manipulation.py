"""Compatibility shim for legacy imports.

The canonical implementation lives in ``core.skills.os_manipulation``.
"""

from core.skills.os_manipulation import DesktopControlSkill, OSManipulationInput

__all__ = ["DesktopControlSkill", "OSManipulationInput"]
