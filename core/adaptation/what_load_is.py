"""What counts as load for her immune system to answer: what the resource governor, which owns load, names.

Moved out of adaptive_immunity.py whole, which is over the size line (tools/lint_module_size.py).
"""
from __future__ import annotations

import logging


def the_governor_names_load() -> bool:
    """Whether the resource governor, which owns load and sheds it on its own cadence, names an eviction tier now.

    Resource pressure over 0.78 is her resident 27B's ordinary footprint. Read as an antigen, it asked autopoiesis to
    reduce_load/global every fifteen seconds: 536 ineffective repairs in two hours on 13 November 2026, each a failure
    fed back here. Where the governor names no tier, the load is what her running model needs.
    """
    try:
        from core.resource.resource_governor import EvictionTier, get_resource_governor

        return get_resource_governor().sample().eviction_tier is not EvictionTier.NONE
    except (ImportError, AttributeError, RuntimeError, OSError) as exc:
        logging.getLogger("Aura.AdaptiveImmunity").debug("No governor reading, so the antigen's own pressure stands: %s", exc)
        return True
