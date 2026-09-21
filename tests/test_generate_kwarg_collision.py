"""A duplicate keyword argument fails the whole engine closed.

Every `_generate_with_client` call passes `max_tokens=`, `temperature=`,
`messages=`, `origin=`, `is_background=` and `foreground_request=` BY NAME, and
also splats `**morpho_kwargs` beside them. Any overlap raises

    TypeError: _generate_with_client() got multiple values for keyword
    argument 'max_tokens'

which fails the inference_gate closed under its fail-closed policy and reaches
the person as `user_cycle_no_response` — the engine returning nothing at all, in
two seconds. Measured live: one added key broke EVERY desktop turn while
ordinary conversation through the same engine kept working, which is what made
it look like a desktop-routing problem instead of a TypeError.
"""

from __future__ import annotations

import re
from pathlib import Path

from core.brain import inference_gate
from tests.source_contract import family_text


def test_the_explicit_kwarg_set_matches_the_call_sites():
    """The scrub is only correct if it lists what the calls actually pass."""
    source = family_text(inference_gate)
    call = source.index("text = await self._generate_with_client(")
    body = source[call : source.index("**morpho_kwargs", call)]
    passed = set(re.findall(r"^\s+([a-z_]+)=", body, flags=re.MULTILINE))

    # Positional arguments and the deadline/label are not keyword collisions.
    keyword_args = passed - {"enabled", "priority", "worker", "timeout_s"}
    missing = keyword_args - inference_gate._GENERATE_EXPLICIT_KWARGS
    assert not missing, (
        f"these are passed explicitly but not scrubbed from morpho_kwargs: {sorted(missing)}"
    )


def test_the_scrub_runs_before_dispatch():
    from source_support import inlined_function_source

    # the turn as it runs: the scrub sits in a block the size sweep moved
    # into a helper, and the order only shows once it is read back in place
    source = inlined_function_source(
        Path("core/brain/inference_gate.py"), "InferenceGate._generate_with_metadata_sink"
    )
    assert "for _reserved in _GENERATE_EXPLICIT_KWARGS:" in source
    assert "morpho_kwargs.pop(_reserved, None)" in source
    scrub_at = source.index("for _reserved in _GENERATE_EXPLICIT_KWARGS:")
    dispatch_at = source.index("text = await self._generate_with_client(")
    assert scrub_at < dispatch_at, "the scrub must run before the call it protects"


def test_scrubbing_removes_every_reserved_name():
    kwargs = {
        "max_tokens": 1024,
        "temperature": 0.7,
        "messages": [],
        "origin": "x",
        "is_background": False,
        "foreground_request": True,
        "top_p": 0.9,
        "clean_user_surface_contract": True,
    }
    for reserved in inference_gate._GENERATE_EXPLICIT_KWARGS:
        kwargs.pop(reserved, None)
    assert kwargs == {"top_p": 0.9, "clean_user_surface_contract": True}, (
        "the scrub must drop exactly the colliding names and keep the rest"
    )
