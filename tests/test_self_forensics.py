"""Self-forensics grounding — she answers about her own failures from
her black boxes, never from invention.

Live regression (July 4): asked about her overnight death she
confabulated electromagnetic interference three drafts in a row while
the true cause sat in her own records. The honesty gate rejected every
draft; nothing supplied the evidence a truthful draft needed.
"""
from __future__ import annotations

import inspect
import json
import time

from core.introspection.self_forensics import (
    build_self_forensics_context,
    is_self_forensics_question,
)


class TestDetector:
    def test_matches_the_live_session_questions(self):
        for question in (
            "did you crash last night",
            "What was the cause",
            "what was the root cause of the crash?",
            "why did you shut down",
            "what happened last night",
            "why did you disappear",
            "you restarted overnight?",
        ):
            assert is_self_forensics_question(question), question

    def test_ordinary_questions_stay_out(self):
        for question in (
            "what is the weather",
            "tell me about the moon landing",
            "how does your memory work",
            "can you hear me?",
        ):
            assert not is_self_forensics_question(question), question


class TestEvidenceBlock:
    def test_reads_real_grace_flag(self, tmp_path, monkeypatch):
        run_dir = tmp_path / ".aura" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "grace_exit.flag").write_text(json.dumps({
            "schema": "aura.shutdown_grace.v1",
            "pid": 1234,
            "reason": "coordinator",
            "created_at_unix": time.time() - 1800,
        }))
        monkeypatch.setenv("HOME", str(tmp_path))
        # Path.home() honors HOME on posix.
        block = build_self_forensics_context()
        assert "GROUNDED SELF-FORENSICS" in block
        assert "reason='coordinator'" in block
        assert "0.5h ago" in block

    def test_no_evidence_yields_honest_unavailability(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        # AURA_LOG_DIR, not chdir. This used to rely on the forensics readers
        # being cwd-relative, which is precisely the defect that had crash
        # correlation watching an empty directory on the real machine. The
        # override is the supported way to say "the record lives here", and it
        # is exclusive — so pointing it at an empty directory is what actually
        # produces a no-evidence run.
        monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "forensics"))
        monkeypatch.chdir(tmp_path)
        # The block also reads process-global incident/fault registries that
        # OTHER tests in the same process legitimately populate — neutralize
        # them so this test asserts the true no-evidence branch.
        import core.introspection.self_forensics as sf

        monkeypatch.setattr(sf, "_live_incidents", lambda: "")
        monkeypatch.setattr(sf, "_recent_faults", lambda: "")
        block = build_self_forensics_context()
        assert "do " in block and "not invent a cause" in block

    @staticmethod
    def _no_black_boxes(tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "forensics"))
        monkeypatch.chdir(tmp_path)
        import core.introspection.self_forensics as sf

        monkeypatch.setattr(sf, "_live_incidents", lambda: "")
        monkeypatch.setattr(sf, "_recent_faults", lambda: "")

    def test_an_empty_outcome_record_is_not_evidence(self, tmp_path, monkeypatch):
        """Absence put into words is still absence.

        The outcome record's reader answers an empty record with a sentence,
        and from 2026-09-05 the block counted that sentence as evidence. A
        machine with no black boxes then got the grounded instruction with one
        line under it — "there is no record of anything having been tried" —
        and lost the one that says the records are unavailable.
        """
        import core.self.what_has_ever_worked as record

        self._no_black_boxes(tmp_path, monkeypatch)
        monkeypatch.setattr(record, "what_has_ever_worked", lambda learner=None: {})
        block = build_self_forensics_context()
        assert "not invent a cause" in block
        assert "WHAT HAS EVER WORKED" not in block

    def test_an_outcome_record_with_rows_is_still_evidence(self, tmp_path, monkeypatch):
        import core.self.what_has_ever_worked as record

        self._no_black_boxes(tmp_path, monkeypatch)
        monkeypatch.setattr(
            record,
            "what_has_ever_worked",
            lambda learner=None: {"search": record.HowItHasGone("search", 4, 3)},
        )
        block = build_self_forensics_context()
        assert "- WHAT HAS EVER WORKED: 1 worked" in block
        assert "not invent a cause" not in block

    def test_instruction_forbids_invention(self):
        block = build_self_forensics_context()
        lowered = block.lower()
        assert "never invent" in lowered or "not invent" in lowered


class TestEngineWiring:
    def test_grounding_block_is_wired_into_think(self):
        from core.brain import cognitive_engine

        src = inspect.getsource(cognitive_engine)
        assert "SELF-FORENSICS EVIDENCE" in src
        assert "is_self_forensics_question" in src


class TestCapabilityMap:
    def test_matches_the_live_declined_task(self):
        from core.introspection.capability_map import is_actionable_request

        assert is_actionable_request(
            "can you write a note about dinosaurs using my notes app, "
            "create a folder on my desktop titled Notes and export it there?"
        )
        assert is_actionable_request("make a file called ideas.txt in my documents")
        assert not is_actionable_request("what do you think about dinosaurs")

    def test_map_names_lanes_and_decomposition_rule(self):
        from core.introspection.capability_map import build_capability_map_context

        block = build_capability_map_context()
        for token in ("FILESYSTEM", "SCRIPTING", "GUI-CONTROL", "never decline the whole task"):
            assert token in block

    def test_engine_wiring(self):
        import inspect

        from core.brain import cognitive_engine

        assert "CAPABILITY MAP" in inspect.getsource(cognitive_engine)


class TestUngroundedSelfCauseGate:
    def test_live_fabrications_flag(self):
        from core.conversation.response_reliability import (
            _has_ungrounded_self_cause_claim,
        )

        assert _has_ungrounded_self_cause_claim(
            "what was the root cause of the crash?",
            "The crash was caused by a memory corruption issue. A module "
            "overwrote critical system pointers.",
        )

    def test_grounded_and_honest_replies_pass(self):
        from core.conversation.response_reliability import (
            _has_ungrounded_self_cause_claim,
        )

        assert not _has_ungrounded_self_cause_claim(
            "what was the root cause of the crash?",
            "The records show my generation gate wedged during a cold load; "
            "the launcher then sent SIGKILL while I was recovering.",
        )
        assert not _has_ungrounded_self_cause_claim(
            "why did you crash",
            "Honestly, the exact cause is unknown — my sentinel shows memory "
            "stayed calm throughout.",
        )
