"""The developmental reservoir's body channel, and whose body it is.

`_advance_lifetime` hands the ontogenetic reservoir one feature per domain.
Its interoception feature read `soma.hardware.cpu_usage` — the load on the
machine the run happens to be on — so the channel moved with whatever else was
compiling, and a lesion of interoception could not reach it, because a lesion
does not change the CPU.

The subject schema names the difference where it records both: the host
readings are the environment, and `soma.exertion` is "what she spent thinking,
the one body channel an experiment can hold the host still without also
holding still".
"""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "core" / "phases" / "affect_update.py"


def _lifetime_features() -> ast.Dict:
    """The dict literal `_advance_lifetime` builds, read from the source.

    Read structurally rather than by running the phase: the reservoir needs a
    live state and a service, and the claim here is about which field the
    feature is taken from.
    """
    tree = ast.parse(SOURCE.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "_advance_lifetime":
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Dict) and any(
                isinstance(key, ast.Constant) and key.value == "interoception"
                for key in inner.keys
            ):
                return inner
    raise AssertionError("_advance_lifetime no longer builds a feature dict")


def _feature_source(name: str) -> str:
    features = _lifetime_features()
    for key, value in zip(features.keys, features.values, strict=True):
        if isinstance(key, ast.Constant) and key.value == name:
            return ast.unparse(value)
    raise AssertionError(f"the reservoir no longer has a {name!r} feature")


def test_the_body_channel_is_her_exertion():
    assert "exertion" in _feature_source("interoception")


def test_the_body_channel_is_not_the_host():
    body = _feature_source("interoception")
    assert "cpu_usage" not in body
    assert "hardware" not in body


def test_every_domain_still_has_exactly_one_feature():
    """Ten domains, ten channels. A missing one is a reservoir sensing into a
    void on that axis, which is how this channel went unnoticed."""
    features = _lifetime_features()
    names = {key.value for key in features.keys if isinstance(key, ast.Constant)}
    assert names == {
        "perception",
        "interoception",
        "affect_valence",
        "affect_arousal",
        "workspace",
        "cognition",
        "self_state",
        "memory",
        "world",
        "deliberation",
    }
