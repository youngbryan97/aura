"""Governed, extensible import policy for deterministic AST repair."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.state_ownership import state_root

_DEFAULT_IMPORTS = {
    "asyncio": "import asyncio",
    "json": "import json",
    "logging": "import logging",
    "time": "import time",
    "math": "import math",
    "re": "import re",
    "Path": "from pathlib import Path",
    "Any": "from typing import Any",
    "Dict": "from typing import Dict",
    "List": "from typing import List",
    "Optional": "from typing import Optional",
}

_FORBIDDEN_ROOTS = frozenset(
    {
        "ctypes",
        "ftplib",
        "http",
        "importlib",
        "multiprocessing",
        "os",
        "shutil",
        "smtplib",
        "socket",
        "subprocess",
        "sys",
        "urllib",
    }
)


@dataclass(frozen=True)
class ApprovedRepairImport:
    symbol: str
    statement: str
    governance_receipt_id: str


class RepairImportPolicy:
    """Allows safe policy growth without turning unknown imports into execution."""

    def __init__(self, policy_path: str | Path | None = None) -> None:
        self.policy_path = Path(
            policy_path or state_root() / "config" / "repair_import_policy.json"
        )
        self._approved = dict(_DEFAULT_IMPORTS)
        self._receipts: dict[str, str] = {key: "builtin-policy" for key in self._approved}
        self._load()

    def resolve(self, symbol: str) -> str | None:
        return self._approved.get(str(symbol or "").strip())

    def approve(
        self,
        symbol: str,
        statement: str,
        *,
        governance_receipt_id: str,
        persist: bool = True,
    ) -> ApprovedRepairImport:
        symbol = str(symbol or "").strip()
        statement = str(statement or "").strip()
        if not symbol or not governance_receipt_id:
            raise ValueError("symbol and governance receipt are required")
        self._validate_statement(symbol, statement)
        self._approved[symbol] = statement
        self._receipts[symbol] = governance_receipt_id
        if persist:
            self._persist()
        return ApprovedRepairImport(symbol, statement, governance_receipt_id)

    @staticmethod
    def _validate_statement(symbol: str, statement: str) -> None:
        tree = ast.parse(statement)
        if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.Import, ast.ImportFrom)):
            raise ValueError("repair import policy accepts exactly one import statement")
        node = tree.body[0]
        roots: set[str] = set()
        exposed: set[str] = set()
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
                exposed.add(alias.asname or alias.name.split(".", 1)[0])
        else:
            roots.add(str(node.module or "").split(".", 1)[0])
            exposed.update(alias.asname or alias.name for alias in node.names)
        if roots & _FORBIDDEN_ROOTS:
            raise ValueError(f"forbidden repair import root: {sorted(roots & _FORBIDDEN_ROOTS)[0]}")
        if symbol not in exposed:
            raise ValueError(f"import statement does not expose requested symbol {symbol}")

    def _load(self) -> None:
        if not self.policy_path.is_file():
            return
        try:
            payload = json.loads(self.policy_path.read_text(encoding="utf-8"))
        # not a failure: text that is not the JSON this expects is not a record it can
        # read back.
        except (OSError, json.JSONDecodeError):
            return
        for raw in payload.get("imports", []):
            if not isinstance(raw, dict):
                continue
            try:
                self.approve(
                    str(raw.get("symbol", "")),
                    str(raw.get("statement", "")),
                    governance_receipt_id=str(raw.get("governance_receipt_id", "")),
                    persist=False,
                )
            except (SyntaxError, ValueError):
                continue

    def _persist(self) -> None:
        records = [
            {
                "symbol": symbol,
                "statement": statement,
                "governance_receipt_id": self._receipts[symbol],
            }
            for symbol, statement in sorted(self._approved.items())
            if self._receipts.get(symbol) != "builtin-policy"
        ]
        atomic_write_text(
            self.policy_path,
            json.dumps({"version": 1, "imports": records}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


__all__ = ["ApprovedRepairImport", "RepairImportPolicy"]
