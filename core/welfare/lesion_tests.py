"""core/welfare/lesion_tests.py
Lesion test suite for behavioral analysis.
Simulates disabling or forcing welfare parameters to verify dependency structures.
"""


class WelfareLesionSuite:
    """Simulates lesions in welfare variables for analysis checks."""

    def __init__(self):
        self._active_lesions: dict[str, float] = {}

    def apply_lesion(self, parameter: str, fixed_value: float) -> None:
        self._active_lesions[parameter] = fixed_value

    def clear_lesions(self) -> None:
        self._active_lesions.clear()

    def get_lesion_value(self, parameter: str, default: float) -> float:
        return self._active_lesions.get(parameter, default)
