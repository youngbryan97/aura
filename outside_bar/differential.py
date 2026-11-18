"""Call the named functions on the given inputs in this checkout, and write down what each call came to.

    python differential.py INPUTS.json OUT.json

INPUTS.json maps ``path::function`` to a list of argument objects. Run once in the code before a change and once in the
code after it, by outside_bar/verify_change.py, inside a boundary that refuses the network and writes outside a scratch
folder.
"""
from __future__ import annotations

import importlib
import json
import signal
import sys
from typing import Any

#: How long one call may take, in seconds. A call that takes longer says nothing about the value it would have returned.
_CALL_SECONDS = 2


def _too_long(*_: Any) -> None:
    raise TimeoutError


def main() -> int:
    inputs, out = json.load(open(sys.argv[1], encoding="utf-8")), {}
    sys.path.insert(0, ".")
    signal.signal(signal.SIGALRM, _too_long)
    for name, calls in inputs.items():
        path, function = name.split("::")
        try:
            target = getattr(importlib.import_module(path.removesuffix(".py").replace("/", ".")), function)
        except Exception as exc:  # noqa: BLE001 - a module that cannot be imported has no values to compare
            out[name] = {"unavailable": type(exc).__name__}
            continue
        results = {}
        for arguments in calls:
            key = json.dumps(arguments, sort_keys=True)
            signal.alarm(_CALL_SECONDS)
            try:
                results[key] = repr(target(**json.loads(key)))
            except TimeoutError:
                results[key] = "took too long"
            except Exception as exc:  # noqa: BLE001 - what it raised is what the call came to
                results[key] = f"raised {type(exc).__name__}"
            finally:
                signal.alarm(0)
        out[name] = results
    json.dump(out, open(sys.argv[2], "w", encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
