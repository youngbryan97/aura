"""The environment her whole self runs in, set up before anything of hers is imported.

A campaign run with `--whole` has to serve the same models the desktop serves.
Two things stood between it and them.

The desktop reads the primary checkout's `.env` in `aura_main.py` before the
model registry reads its flags, and that file is what makes the Ternary Bonsai
her brainstem. A campaign never read it, so it would have served the 9B the
flag defaults to. A worktree has no `.env` of its own, so the file is found the
way `launch_aura.sh` finds it: `AURA_ENV_FILE`, else the primary checkout.

And under a campaign's own state root the registry cannot verify the fused
model's pointer, so it falls back to the base weights rather than her persona
model. So the registry is asked, in a child process that has the desktop's
environment and none of the campaign's isolation, which paths it would serve
for each lane, and those paths are pinned for the run through the flags the
registry itself reads first.

Standard library only: this runs before any module of hers is imported,
because the registry reads its flags when it is imported.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

__all__ = ["enter_whole_environment", "env_file_for"]

#: Where each lane's path is pinned, and the registry call that resolves it.
_LANES: dict[str, str] = {
    "AURA_LLM__MLX_MODEL_PATH": "str(m._current_cortex_path())",
    "AURA_LLM__MLX_DEEP_MODEL_PATH": "str(m.get_deep_model_path())",
    "AURA_LLM__MLX_BRAINSTEM_PATH": "str(m.get_brainstem_path())",
}

#: The key that signs the promotion evidence behind the active cortex, pinned the
#: same way. Under a campaign's own state root the registry looked for it there,
#: found nothing, and could not confirm the model it had loaded, so her affective
#: steering stayed detached for the whole run of 23 September and the substrate
#: modulated none of what she said.
_CUSTODY: dict[str, str] = {
    "AURA_CORTEX_AUTHORITY_KEY_FILE": "str(a.default_authority_key_path())",
}

#: Variables that make a process a campaign rather than the desktop.
_ISOLATION = ("AURA_STATE_ROOT", "AURA_TESTING", "AURA_PROOF_RUN", "AURA_LOG_DIR")


def env_file_for(repo: Path) -> Path | None:
    """The `.env` the desktop would read for this checkout, or None."""
    override = str(os.environ.get("AURA_ENV_FILE", "") or "").strip()
    candidates = [Path(override).expanduser()] if override else []
    candidates.append(repo / ".env")
    marker = repo / ".git"
    if marker.is_file():
        # A worktree: `gitdir: <primary>/.git/worktrees/<name>`.
        text = marker.read_text("utf-8", errors="ignore").strip()
        if text.startswith("gitdir:"):
            gitdir = Path(text.split(":", 1)[1].strip())
            candidates.append(gitdir.parents[1].parent / ".env")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text("utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def enter_whole_environment(repo: Path, python: str | None = None) -> dict[str, str]:
    """Load the desktop's `.env` and pin the lanes it would serve. Returns what was pinned.

    Existing variables win, as `load_dotenv(override=False)` lets them in the
    desktop, so a pin set by hand is kept.
    """
    env_file = env_file_for(repo)
    if env_file is not None:
        for key, value in _read_env_file(env_file).items():
            os.environ.setdefault(key, value)
    pins = {**_LANES, **_CUSTODY}
    wanted = {name: call for name, call in pins.items() if not os.environ.get(name)}
    pinned: dict[str, str] = {name: os.environ[name] for name in pins if os.environ.get(name)}
    if wanted:
        child_env = {key: value for key, value in os.environ.items() if key not in _ISOLATION}
        script = (
            "import json\n"
            "from core.brain.llm import model_registry as m\n"
            "from core.learning import cortex_migration_authority as a\n"
            f"print(json.dumps({{{', '.join(f'{name!r}: {call}' for name, call in wanted.items())}}}))\n"
        )
        result = subprocess.run(
            [python or sys.executable, "-c", script],
            cwd=str(repo),
            env={**child_env, "PYTHONPATH": str(repo)},
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        resolved = json.loads(result.stdout.strip().splitlines()[-1])
        for name, value in resolved.items():
            os.environ[name] = value
            pinned[name] = value
    if env_file is not None:
        pinned["env_file"] = str(env_file)
    return pinned
