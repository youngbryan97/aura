from __future__ import annotations

import json

import pytest

from core.language.substrate_store import LanguageSubstrateStore


def test_matcher_identity_cannot_escape_the_owned_namespace(tmp_path) -> None:
    store = LanguageSubstrateStore(data_root=tmp_path, project_root=tmp_path)

    for unsafe in ("", "../outside", "a/b", ".hidden", "name with spaces"):
        with pytest.raises(ValueError):
            store.matcher_path(unsafe)


def test_store_derives_and_schema_wraps_matcher_state(tmp_path) -> None:
    store = LanguageSubstrateStore(data_root=tmp_path, project_root=tmp_path)

    target = store.write_matcher("action_claim", {"name": "action_claim"})

    assert target == tmp_path / "language" / "action_claim.json"
    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["schema"] == "aura.language.learned_matcher"
    assert document["payload"] == {"name": "action_claim"}


def test_measurement_has_one_fixed_artifact_path(tmp_path) -> None:
    store = LanguageSubstrateStore(data_root=tmp_path, project_root=tmp_path)

    target = store.write_measurement({"results": []})

    assert target == tmp_path / "artifacts" / "language_substrate" / "measurement.json"
    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["schema"] == "aura.language.substrate_measurement"
    assert document["payload"] == {"results": []}
