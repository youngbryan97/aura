"""Every number in the package carries the reason it holds that value.

A coefficient with no stated origin is an opinion wearing a decimal
point, and the reviewed prototypes are full of them: a sanctity prior of
10.0 that makes conscientious objection return True for every input in
range, a resolvability of 0.85 that makes the resolution half of
incongruity-resolution a constant.

A guess is allowed here on one condition, and this is where the
condition is enforced: a calibration parameter must not change the
ordering the faculty produces, anywhere in the range it declares.
"""

from __future__ import annotations

import pytest

from core.interiority.faculties import load_all
from core.interiority.params import ParamKind, registry as param_registry


@pytest.fixture(scope="module", autouse=True)
def _loaded() -> None:
    load_all()


def test_every_parameter_states_its_basis() -> None:
    params = param_registry().all()
    assert params, "no parameters declared"
    for param in params:
        assert len(param.basis) > 40, f"{param.name}: basis is too thin to check"
        assert param.sensitivity.strip(), f"{param.name}: no stated sensitivity"
        assert param.lower <= param.value <= param.upper, param.name


def test_calibration_parameters_declare_the_range_they_could_take() -> None:
    for param in param_registry().calibration():
        assert param.sweep_range is not None, param.name
        low, high = param.sweep_range
        assert low < high, param.name
        assert len(param.sweep()) >= 2, param.name


def test_no_parameter_is_a_bare_guess() -> None:
    """A cited or derived parameter must say what it is derived from."""
    for param in param_registry().all():
        if param.kind in (ParamKind.CITED, ParamKind.DERIVED):
            lowered = param.basis.lower()
            assert any(
                marker in lowered
                for marker in (
                    "because", "so ", "set ", "matches", "follows", "derived",
                    "reported", "measured", "which is", "the same",
                )
            ), f"{param.name}: basis states no derivation"
