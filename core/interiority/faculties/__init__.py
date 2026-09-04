"""core/interiority/faculties — the forty-three, one module each.

They are separate modules because they are separate mechanisms. Reading
another animal's state is inverse inference over channels; grief is
extinction over a distributed prediction; anger is a bargaining
threshold; guilt is a counterfactual over an endorsed norm. Putting them
behind one scoring function with different weights, which is what most
of the reviewed prototypes do, makes them the same mechanism wearing
forty-three names, and the way you can tell is that no intervention
distinguishes them.

Importing this package registers every faculty. Nothing else in the
package imports a faculty by name.
"""

from __future__ import annotations

import importlib
import pkgutil

from core.interiority.faculty import registry

__all__ = ["load_all", "registry"]


def load_all() -> int:
    """Import every faculty module. Returns how many are registered."""
    package = __name__
    for module in pkgutil.iter_modules(__path__):
        if module.name.startswith("f") and not module.name.startswith("_"):
            importlib.import_module(f"{package}.{module.name}")
    return len(registry())
