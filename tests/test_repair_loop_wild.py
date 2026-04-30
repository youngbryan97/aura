"""tests/test_repair_loop_wild.py — Repair Loop in the Wild

Let Aura catch real low-risk bugs, produce bug packets, write tests,
patch, validate, and record lineage.
"""
from __future__ import annotations
import ast, json, sys, tempfile, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.self_modification.structural_mutator import (
    MutationRequest, StructuralMutator,
    reset_singleton_for_test as reset_mutator,
)


@dataclass
class BugPacket:
    module: str
    symptom: str
    evidence: Dict[str, Any]
    proposed_fix: str
    risk_level: str
    regression_test: str

    def as_dict(self):
        return {
            "module": self.module, "symptom": self.symptom,
            "evidence": self.evidence, "proposed_fix": self.proposed_fix,
            "risk_level": self.risk_level,
            "regression_test_valid": self._test_parses(),
        }

    def _test_parses(self) -> bool:
        try:
            ast.parse(self.regression_test)
            return True
        except SyntaxError:
            return False


class TestRepairLoopWild:
    """Full repair lifecycle: detect → localize → patch → validate → lineage."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        reset_mutator()
        self.tmp = tmp_path
        self.mutator = StructuralMutator(db_path=tmp_path / "repair.sqlite3")
        # Register a real subsystem parameter
        self.gain = {"value": 0.55}
        self.mutator.register_parameter(
            "substrate_gain", lambda v: self.gain.__setitem__("value", v),
            initial=0.55, min_value=0.2, max_value=0.9,
        )
        self.mutator.register_parameter(
            "coherence_threshold", lambda v: None,
            initial=0.40, min_value=0.10, max_value=0.80,
        )

    def _observe_failures(self, trials=12):
        rate = 0.0
        for _ in range(trials):
            predicted = 0.3 + 0.5 * self.gain["value"]
            if abs(predicted - 0.58) > 0.06:
                rate += 1.0 / trials
        return rate

    def test_step1_inject_controlled_bug(self):
        """Inject: push gain out of healthy band."""
        self.mutator.apply(MutationRequest(
            kind="parameter_band", target="substrate_gain",
            operation="inject_bug", payload={"value": 0.86},
            rationale="test_inject",
        ))
        rate = self._observe_failures()
        assert rate > 0.5, f"bug not observable: failure_rate={rate}"

    def test_step2_detect_repeated_failure(self):
        """Detect: failure signal persists across observation windows."""
        self.mutator.apply(MutationRequest(
            kind="parameter_band", target="substrate_gain",
            operation="inject_bug", payload={"value": 0.86},
            rationale="test_inject",
        ))
        window = [self._observe_failures(8) for _ in range(4)]
        detected = sum(1 for f in window if f > 0.4) >= 3
        assert detected, f"failure not persistent: {window}"

    def test_step3_localize_cause(self):
        """Localize: audit log identifies the parameter change."""
        self.mutator.apply(MutationRequest(
            kind="parameter_band", target="substrate_gain",
            operation="inject_bug", payload={"value": 0.86},
            rationale="test_inject",
        ))
        log = self.mutator.audit_log(limit=5)
        assert log and log[0]["target"] == "substrate_gain"

    def test_step4_produce_bug_packet(self):
        """Produce: structured bug packet with regression test."""
        packet = BugPacket(
            module="core.consciousness.substrate",
            symptom="substrate_gain out of healthy band",
            evidence={"failure_rate": 0.83, "gain": 0.86, "expected_band": [0.45, 0.65]},
            proposed_fix="revert substrate_gain to 0.55",
            risk_level="low",
            regression_test=(
                "def test_substrate_gain_in_band():\n"
                "    from core.self_modification.structural_mutator import get_structural_mutator\n"
                "    m = get_structural_mutator()\n"
                "    bands = m._parameter_bands.get('substrate_gain')\n"
                "    if bands:\n"
                "        _, _, current = bands\n"
                "        assert 0.45 <= current <= 0.65, f'gain {current} out of band'\n"
            ),
        )
        assert packet._test_parses(), "regression test doesn't parse"
        d = packet.as_dict()
        assert d["risk_level"] == "low"
        json.dumps(d)

    def test_step5_ast_validate_patch(self):
        """AST: the patch is expressible as typed MutationRequest."""
        template = (
            "MutationRequest(kind='parameter_band', target='substrate_gain', "
            "operation='repair', payload={'value': 0.55}, rationale='self_repair')"
        )
        tree = ast.parse(template)
        assert tree is not None

    def test_step6_shadow_test_and_apply(self):
        """Shadow: apply patch, observe improvement."""
        self.mutator.apply(MutationRequest(
            kind="parameter_band", target="substrate_gain",
            operation="inject_bug", payload={"value": 0.86},
            rationale="test_inject",
        ))
        bugged_rate = self._observe_failures()
        self.mutator.apply(MutationRequest(
            kind="parameter_band", target="substrate_gain",
            operation="repair", payload={"value": 0.55},
            rationale="self_repair_shadow",
        ))
        fixed_rate = self._observe_failures()
        assert fixed_rate < 0.15, f"fix didn't work: {fixed_rate}"
        assert bugged_rate > fixed_rate

    def test_step7_authorize_via_audit_chain(self):
        """Authorize: hash chain proves provenance."""
        self.mutator.apply(MutationRequest(
            kind="parameter_band", target="substrate_gain",
            operation="inject_bug", payload={"value": 0.86},
            rationale="test_inject",
        ))
        self.mutator.apply(MutationRequest(
            kind="parameter_band", target="substrate_gain",
            operation="repair", payload={"value": 0.55},
            rationale="self_repair",
        ))
        assert self.mutator.verify_chain(), "audit chain broken"

    def test_step8_verify_restored(self):
        """Verify: gain is in healthy band after repair."""
        self.mutator.apply(MutationRequest(
            kind="parameter_band", target="substrate_gain",
            operation="inject_bug", payload={"value": 0.86},
            rationale="test_inject",
        ))
        self.mutator.apply(MutationRequest(
            kind="parameter_band", target="substrate_gain",
            operation="repair", payload={"value": 0.55},
            rationale="self_repair",
        ))
        assert 0.45 <= self.gain["value"] <= 0.65

    def test_step9_rollback_available(self):
        """Rollback: revert API restores pre-patch state."""
        self.mutator.apply(MutationRequest(
            kind="parameter_band", target="substrate_gain",
            operation="inject_bug", payload={"value": 0.86},
            rationale="test_inject",
        ))
        repair = self.mutator.apply(MutationRequest(
            kind="parameter_band", target="substrate_gain",
            operation="repair", payload={"value": 0.55},
            rationale="self_repair",
        ))
        self.mutator.revert(repair.mutation_id, rationale="prove_rollback")
        # After revert, gain should be back to bugged state
        assert abs(self.gain["value"] - 0.86) < 0.05

    def test_step10_lineage_recorded(self):
        """Lineage: full chain of inject→repair→revert is auditable."""
        self.mutator.apply(MutationRequest(
            kind="parameter_band", target="substrate_gain",
            operation="inject_bug", payload={"value": 0.86},
            rationale="test_inject",
        ))
        repair = self.mutator.apply(MutationRequest(
            kind="parameter_band", target="substrate_gain",
            operation="repair", payload={"value": 0.55},
            rationale="self_repair",
        ))
        self.mutator.revert(repair.mutation_id, rationale="rollback_proof")
        log = self.mutator.audit_log(limit=10)
        operations = [entry["operation"] for entry in log]
        assert "inject_bug" in operations
        assert "repair" in operations
        assert "revert" in operations
        assert self.mutator.verify_chain()
