from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace


def test_third_party_import_initialization_is_process_serialized(monkeypatch) -> None:
    from core.runtime import third_party_imports

    state_lock = threading.Lock()
    active = 0
    max_active = 0

    def _import(name: str):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with state_lock:
            active -= 1
        return SimpleNamespace(name=name)

    monkeypatch.setattr(third_party_imports.importlib, "import_module", _import)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(
            pool.map(
                third_party_imports.import_module_serialized,
                ("sentence_transformers", "faster_whisper"),
            )
        )

    assert tuple(result.name for result in results) == (
        "sentence_transformers",
        "faster_whisper",
    )
    assert max_active == 1


def test_third_party_attribute_resolves_inside_same_import_fence(monkeypatch) -> None:
    from core.runtime import third_party_imports

    expected = object()
    monkeypatch.setattr(
        third_party_imports.importlib,
        "import_module",
        lambda _name: SimpleNamespace(WhisperModel=expected),
    )

    assert (
        third_party_imports.import_attribute_serialized(
            "faster_whisper",
            "WhisperModel",
        )
        is expected
    )
