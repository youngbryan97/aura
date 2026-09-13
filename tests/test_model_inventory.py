"""Every live role resolves to a checkpoint that is here, with the facts its consumers read.

Q01. The inventory is `tools/model_inventory.py`; this holds what it must
report and that nothing it names is missing. What sits on disk unnamed by any
code is listed with its size and left where it is — deleting a checkpoint is
the owner's decision, and the tool's job is to make that decision possible.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import model_inventory  # noqa: E402

pytestmark = [
    pytest.mark.model,
    pytest.mark.slow,
    pytest.mark.skipif(
        not Path(os.environ.get("AURA_MODELS_DIR", "~/.aura/models")).expanduser().is_dir(),
        reason="no models directory on this machine",
    ),
]


@pytest.fixture(scope="module")
def record():
    return model_inventory.inventory()


def test_every_required_role_resolves_to_a_checkpoint_on_disk(record) -> None:
    assert record["live_roles_missing"] == [], record["live_roles_missing"]
    for role in record["roles"]:
        if role.get("optional"):
            continue
        assert role["exists"], role
        assert role["size_gb"] > 0, role


def test_a_language_model_role_carries_what_its_consumers_read(record) -> None:
    by_role = {r["role"]: r for r in record["roles"]}
    for name in ("cortex", "brainstem", "reflex"):
        row = by_role[name]
        assert row["context_window"]["tokens"] > 0, row
        assert row["geometry"].get("num_hidden_layers"), row
        assert row["tokenizer"].get("chat_template_sha256"), row
        # A declared footprint smaller than the checkpoint cannot admit it.
        assert float(row["declared_footprint_gb"]) >= float(row["size_gb"]), row


def test_a_role_that_is_not_a_language_model_is_not_asked_for_a_context_window(record) -> None:
    by_role = {r["role"]: r for r in record["roles"]}
    for name in ("embedding", "asr_partial", "asr_final"):
        if name in by_role and by_role[name].get("exists"):
            assert "context_window" not in by_role[name], by_role[name]


def test_unreferenced_checkpoints_are_listed_and_not_touched(record, tmp_path) -> None:
    for row in record["unreferenced_on_disk"]:
        assert row["code_mentions"] == 0
        assert Path(row["path"]).exists(), "the inventory reports; it does not delete"
    assert record["unreferenced_gb"] == pytest.approx(
        sum(r["size_gb"] for r in record["unreferenced_on_disk"]), abs=0.1
    )
