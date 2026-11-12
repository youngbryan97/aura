"""What every launch of her runtime starts from, whichever launcher started it.

Two launchers start her. launch_aura.sh set its defaults with `: "${VAR:=value}"`, and the app's own launcher
(scripts/AuraLauncher.swift) spawns aura_main.py directly with a copy of some of them in baseAuraEnvironment(). The
copies drifted. LIVE 12 November 2026, her app relaunched by Aura.app was asked to check a claim and read nothing in
1.3 s: Bryan's standing yes to the reputable sources (10 November, AURA_REACH_READ_HOSTS=@reputable) lived only in
the shell script, and so did her epistemic reach being on at all.

These are applied by aura_main.py right after her .env is read, only where nothing has set them, so a value from
.env, the shell or either launcher still wins. One owner, read by every path.
"""
from __future__ import annotations

import os
from collections.abc import MutableMapping

__all__ = ["STANDING_DEFAULTS", "apply_standing_defaults"]

#: Name -> value, each with the decision it carries.
STANDING_DEFAULTS: dict[str, str] = {
    # Feel the decode: the per-token surprisal and entropy tap.
    "AURA_INTEROCEPTION": "1",
    # Felt doubt, and a person's request to check, may verify a claim outside, through the governed gateway.
    "AURA_EPISTEMIC_REACH": "1",
    # Bryan's standing yes, 10 November 2026: her ordinary app reads the reputable sources
    # (config/reputable_sources.json, its search engines among them).
    "AURA_REACH_READ_HOSTS": "@reputable",
    # The hard hourly budget for her external verifications.
    "AURA_EPISTEMIC_REACH_PER_HOUR": "6",
}


def apply_standing_defaults(env: MutableMapping[str, str] | None = None) -> list[str]:
    """Set each standing default nothing else has set; the names it set, in order."""
    target = os.environ if env is None else env
    set_here = []
    for name, value in STANDING_DEFAULTS.items():
        if not str(target.get(name, "") or "").strip():
            target[name] = value
            set_here.append(name)
    return set_here
