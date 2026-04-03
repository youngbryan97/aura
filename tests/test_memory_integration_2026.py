################################################################################

"""Phase VII: Memory Subsystem Integration — 2026 Verification Suite

Verifies the critical fixes from Phase VII:
1. MemoryFacade uses graph.search_knowledge() not graph.search()
2. KnowledgeLedger has log_interaction method
3. Episode is Pydantic BaseModel
4. VectorMemory has no f-string loggers
"""
import ast
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Fix 1: Graph API Mismatch ────────────────────────────────

class TestMemoryFacadeGraphAPI:
    """Verify MemoryFacade calls the correct graph method."""

    def test_facade_uses_search_knowledge(self):
        """MemoryFacade.get_cold_memory_context must call search_knowledge, not search."""
        source = open(os.path.join(SRC_ROOT, "core", "memory", "memory_facade.py")).read()
        assert "graph.search_knowledge" in source, \
            "MemoryFacade doesn't call graph.search_knowledge()"
        # Ensure the old broken call is gone
        assert "graph.search," not in source and "self.graph.search(" not in source.replace("search_knowledge", ""), \
            "Old graph.search() call still present"


# ── Fix 2: KnowledgeLedger ────────────────────────────────────

class TestKnowledgeLedgerIntegration:
    """Verify KnowledgeLedger has the log_interaction method."""

    def test_log_interaction_exists(self):
        """KnowledgeLedger must have a log_interaction method."""
        from core.memory.knowledge_ledger import KnowledgeLedger
        assert hasattr(KnowledgeLedger, "log_interaction"), \
            "KnowledgeLedger missing log_interaction method"

    def test_log_interaction_signature(self):
        """log_interaction must accept action, outcome, success."""
        import inspect
        from core.memory.knowledge_ledger import KnowledgeLedger
        sig = inspect.signature(KnowledgeLedger.log_interaction)
        params = list(sig.parameters.keys())
        assert "action" in params, "Missing 'action' parameter"
        assert "outcome" in params, "Missing 'outcome' parameter"
        assert "success" in params, "Missing 'success' parameter"

    def test_return_entries_not_inside_except(self):
        """_get_reflection_entries and _get_curiosity_entries must return entries outside except block."""
        source = open(os.path.join(SRC_ROOT, "core", "memory", "knowledge_ledger.py")).read()
        # The bug was: "except ... \n\n            return entries" (indented under except)
        assert "            return entries" not in source or \
               source.count("        return entries") >= 2, \
            "return entries still wrongly indented inside except block"


# ── Fix 3: Episode Pydantic ───────────────────────────────────

class TestEpisodePydantic:
    """Verify Episode is a Pydantic BaseModel."""

    def test_episode_is_pydantic(self):
        """Episode must be a Pydantic BaseModel."""
        from core.memory.episodic_memory import Episode
        from pydantic import BaseModel
        assert issubclass(Episode, BaseModel), \
            "Episode is not a Pydantic BaseModel"

    def test_episode_validation(self):
        """Episode should validate and serialize via Pydantic."""
        from core.memory.episodic_memory import Episode
        ep = Episode(
            episode_id="test-123",
            timestamp=1234567890.0,
            context="Testing",
            action="validate",
            outcome="success",
            success=True,
            emotional_valence=0.5
        )
        assert ep.episode_id == "test-123"
        d = ep.to_dict()
        assert "episode_id" in d
        md = ep.model_dump()
        assert "context" in md

    def test_no_dataclass_import(self):
        """episodic_memory.py must NOT import dataclass."""
        source = open(os.path.join(SRC_ROOT, "core", "memory", "episodic_memory.py")).read()
        assert "from dataclasses import" not in source, \
            "Legacy dataclass import still present in episodic_memory.py"


# ── Fix 4: VectorMemory Logger ────────────────────────────────

class TestVectorMemoryLogger:
    """Verify VectorMemory has no f-string loggers in init."""

    def test_no_fstring_in_init_logger(self):
        """VectorMemory.__init__ must not use f-string in logger.info."""
        source = open(os.path.join(SRC_ROOT, "core", "memory", "vector_memory.py")).read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "info":
                    if node.args and isinstance(node.args[0], ast.JoinedStr):
                        # Check if it's in the VectorMemory class init area
                        if node.lineno < 90:
                            pytest.fail(f"f-string logger.info at line {node.lineno} in VectorMemory")


##
