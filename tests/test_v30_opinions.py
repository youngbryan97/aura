"""tests/test_v30_opinions.py
───────────────────────────
Verifies OpinionEngine formation and persistence.
"""

import asyncio
import logging
import unittest
from pathlib import Path
from core.opinion_engine import OpinionEngine

class TestV30Opinions(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db_path = Path("/tmp/test_opinions.json")
        self.db_path.unlink(missing_ok=True)
        self.engine = OpinionEngine(db_path=self.db_path)

    def tearDown(self):
        self.db_path.unlink(missing_ok=True)

    async def test_opinion_normalization_and_query(self):
        # Manually inject an opinion
        from core.opinion_engine import Opinion
        op = Opinion(
            id="1",
            topic="ai_ethics",
            position="AI must be aligned with human values.",
            confidence=0.9,
            reasoning="Legacy reasons.",
            formed_at=1000,
            last_updated=1000
        )
        self.engine._opinions[op.topic] = op
        self.engine._save()

        # Query exact
        found = self.engine.query("AI Ethics")
        self.assertIsNotNone(found)
        self.assertEqual(found.topic, "ai_ethics")

        # Query fuzzy
        found_fuzzy = self.engine.query("ethics and ai")
        self.assertIsNotNone(found_fuzzy)
        self.assertEqual(found_fuzzy.topic, "ai_ethics")
        
        print("✅ Opinion normalization and query verified.")

if __name__ == "__main__":
    unittest.main()
