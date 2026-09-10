"""Every readiness check threw the conversation's cached prefix away.

`disable_prompt_cache` means "this request neither reads nor writes the
cache". `clear_prompt_cache` meant something much larger: wipe the whole
model+scope trie. Every caller that asked for the second also asked for the
first, so the wipe was redundant for the caller and destructive for everyone
else — and the scope is `user_surface`, which is the conversation.

The visible readiness probe asked for it, and that probe runs BETWEEN user
turns.

LIVE, 2026-09-08:

    🔥 [MLX] Verifying conversation readiness ... with a visible probe
    🧊 [PROMPT CACHE] cleared everything under key=(5026061904, 'user_surface')

then three consecutive turns of one conversation, each `matched 0 (0.0%)`
against 663 tokens retained from the turn before.

The same reasoning was already written in the worker beside the bypass flag —
"health probes fire between user turns, and clearing on every probe would
evict the conversation's cached prefix before the next turn could reuse it" —
and the explicit flag went on doing it.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_probe_does_not_ask_for_the_scope_to_be_wiped():
    from core.brain.llm import mlx_client

    source = inspect.getsource(mlx_client.MLXLocalClient.prove_visible_readiness)
    assert "disable_prompt_cache=True" in source, (
        "the probe must still refuse to read or write the cache"
    )
    assert "clear_prompt_cache=True" not in source


def test_the_flag_no_longer_wipes_the_trie():
    from core.brain.llm import mlx_worker

    source = inspect.getsource(mlx_worker)
    at = source.index('clear_prompt_cache = bool(job.get("clear_prompt_cache", False))')
    window = source[at : at + 300]
    assert "disable_prompt_cache = True" in window
    assert "clear_model_key" not in window


def test_every_caller_that_asks_to_clear_also_asks_to_disable():
    """The premise of the change: the wipe was redundant at the caller.

    If a future caller asks to clear WITHOUT disabling, it wanted something
    this flag no longer does, and it should say so out loud rather than
    silently getting a no-op.
    """
    lonely: list[str] = []
    for path in sorted(ROOT.glob("core/**/*.py")) + sorted(ROOT.glob("interface/**/*.py")):
        if path.name in {"mlx_worker.py", "mlx_client.py"}:
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        # The flag as PASSED, not as discussed. Several places explain in prose
        # why they no longer use it, and a scanner that reads comments would
        # report those as callers.
        for match in re.finditer(r"^[^#\n]*\bclear_prompt_cache\b", text, re.M):
            start = text.rfind("\n", 0, max(0, match.start() - 400))
            end = text.find("\n", match.end() + 400)
            near = text[max(0, start) : end if end > 0 else len(text)]
            if "disable_prompt_cache" not in near:
                lonely.append(f"{path.relative_to(ROOT)}:{text[:match.start()].count(chr(10)) + 1}")
    assert not lonely, (
        "these ask to clear the prompt cache without disabling it for their "
        f"own request, which is now a no-op: {lonely}"
    )
