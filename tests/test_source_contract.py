from pathlib import Path

import pytest

from core.runtime.source_contract import (
    source_contract_sha256,
    source_contract_sha256s,
)


def test_symbol_contract_ignores_unrelated_source_edits(tmp_path: Path):
    source = tmp_path / "sample.py"
    source.write_text(
        "def kept(value):\n    return value + 1\n\ndef other():\n    return 1\n",
        encoding="utf-8",
    )
    before = source_contract_sha256(source, "symbol:kept")
    source.write_text(
        "def kept(value):\n    return value + 1\n\ndef other():\n    return 2\n",
        encoding="utf-8",
    )
    assert source_contract_sha256(source, "symbol:kept") == before


def test_symbol_contract_detects_load_bearing_edit(tmp_path: Path):
    source = tmp_path / "sample.py"
    source.write_text("def kept(value):\n    return value + 1\n", encoding="utf-8")
    before = source_contract_sha256(source, "symbol:kept")
    source.write_text("def kept(value):\n    return value + 2\n", encoding="utf-8")
    assert source_contract_sha256(source, "symbol:kept") != before


def test_contract_inventory_names_each_selector(tmp_path: Path):
    source = tmp_path / "sample.py"
    source.write_text(
        "def kept(value):\n    return helper(value)\n\ndef helper(value):\n    return value\n",
        encoding="utf-8",
    )
    assert set(
        source_contract_sha256s(
            tmp_path,
            {"sample.py": ("symbol:kept", "call:helper")},
        )
    ) == {"sample.py::symbol:kept", "sample.py::call:helper"}


def test_contract_rejects_missing_or_unknown_selector(tmp_path: Path):
    source = tmp_path / "sample.py"
    source.write_text("def kept():\n    return 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="symbol is missing"):
        source_contract_sha256(source, "symbol:absent")
    with pytest.raises(RuntimeError, match="selector kind is invalid"):
        source_contract_sha256(source, "line:kept")
