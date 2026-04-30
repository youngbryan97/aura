from __future__ import annotations
#!/usr/bin/env python3
"""Compatibility entrypoint for Aura cleanup.

The desktop launchers still invoke `aura_cleanup.py` from the repo root.
Keep this thin shim in place so launcher flows remain stable even if the
cleanup implementation lives under `scripts/one_off/`.
"""


import runpy
from pathlib import Path


def main() -> None:
    cleanup_script = Path(__file__).resolve().parent / "scripts" / "one_off" / "aura_cleanup.py"
    runpy.run_path(str(cleanup_script), run_name="__main__")


if __name__ == "__main__":
    main()
