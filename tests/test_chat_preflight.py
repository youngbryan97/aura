"""tests/test_chat_preflight.py
─────────────────────────────────
Unit tests for the chat preflight helpers (file-reference detection,
file loading with sandboxing, pending-chat queue).

Run:
    /Users/bryan/.aura/live-source/.venv/bin/python -m unittest tests.test_chat_preflight -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.conversation.chat_preflight import (
    PendingChat,
    answer_pending,
    build_file_context_block,
    consume_for_session,
    enqueue,
    extract_file_references,
    format_resume_prefix,
    has_unanswered_for_session,
    load_referenced_files,
)


def _temp_path(suffix: str = ".jsonl") -> Path:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    p = Path(path)
    if p.exists():
        p.unlink()
    return p


class TestFileReferenceDetection(unittest.TestCase):
    def test_look_at_pattern(self):
        refs = extract_file_references("Look at the file aura/knowledge/bryan-curated-media.md")
        self.assertIn("aura/knowledge/bryan-curated-media.md", refs)

    def test_at_pattern(self):
        refs = extract_file_references("I dropped a curated media list at aura/knowledge/bryan-curated-media.md")
        self.assertIn("aura/knowledge/bryan-curated-media.md", refs)

    def test_read_pattern(self):
        refs = extract_file_references("Read scoping/fuse-comparison-9870-vs-7500.md and tell me what you think")
        self.assertIn("scoping/fuse-comparison-9870-vs-7500.md", refs)

    def test_no_false_positives(self):
        self.assertEqual(extract_file_references("Just a normal chat with no files"), [])
        self.assertEqual(extract_file_references(""), [])
        self.assertEqual(extract_file_references("How's the weather today.com"), [])

    def test_caps_at_max(self):
        msg = " ".join([f"look at file{i}.md" for i in range(20)])
        self.assertLessEqual(len(extract_file_references(msg)), 3)

    def test_dedup(self):
        refs = extract_file_references("Look at X.md. Read X.md. Open X.md.")
        # Should appear once, not three times
        self.assertEqual(refs.count("X.md"), 1)


class TestFileLoading(unittest.TestCase):
    def test_loads_existing_real_file(self):
        # The curated-media doc was shipped in the previous commit
        files = load_referenced_files(["aura/knowledge/bryan-curated-media.md"])
        self.assertEqual(len(files), 1)
        display, content = files[0]
        self.assertTrue(display.endswith("bryan-curated-media.md"))
        self.assertGreater(len(content), 100)
        self.assertIn("curated", content.lower())

    def test_rejects_traversal(self):
        # Must not be able to escape PROJECT_ROOT
        files = load_referenced_files(["../../../../etc/passwd"])
        self.assertEqual(files, [])

    def test_rejects_unsupported_extension(self):
        # .safetensors files exist but should be filtered
        files = load_referenced_files(["training/adapters/aura-personality/adapters.safetensors"])
        self.assertEqual(files, [])

    def test_missing_file_returns_empty(self):
        files = load_referenced_files(["this/path/does/not/exist.md"])
        self.assertEqual(files, [])

    def test_build_context_block_format(self):
        block = build_file_context_block(["aura/knowledge/bryan-curated-media.md"])
        self.assertIn("=== FILE:", block)
        self.assertIn("=== END", block)
        self.assertIn("references files", block)


class TestPendingQueue(unittest.TestCase):
    def setUp(self):
        self.path = _temp_path()

    def tearDown(self):
        if self.path.exists():
            self.path.unlink()

    def test_enqueue_and_unanswered_check(self):
        enqueue("session-1", "What is the meaning of life?", reason="timeout", path=self.path)
        self.assertTrue(has_unanswered_for_session("session-1", path=self.path))
        self.assertFalse(has_unanswered_for_session("other-session", path=self.path))

    def test_answer_marks_consumed(self):
        enqueue("s2", "How are you?", path=self.path)
        ok = answer_pending("s2", "I'm doing fine — sorry for the wait.", path=self.path)
        self.assertTrue(ok)
        # No more unanswered for this session
        self.assertFalse(has_unanswered_for_session("s2", path=self.path))

    def test_consume_returns_answered_only(self):
        enqueue("s3", "Q1", path=self.path)
        enqueue("s3", "Q2", path=self.path)
        answer_pending("s3", "A2", path=self.path)
        delivered = consume_for_session("s3", path=self.path)
        # Only the answered one should be delivered
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].answer_text, "A2")
        # The unanswered one should remain
        self.assertTrue(has_unanswered_for_session("s3", path=self.path))

    def test_format_resume_prefix(self):
        delivered = [PendingChat(
            session_id="s4",
            user_message="What's the status of the deploy?",
            queued_at=time.time(),
            answered=True,
            answer_text="Deploy succeeded; all four shards green.",
            answered_at=time.time(),
        )]
        prefix = format_resume_prefix(delivered)
        self.assertIn("Coming back to your earlier message", prefix)
        self.assertIn("status of the deploy", prefix)
        self.assertIn("all four shards green", prefix)

    def test_empty_delivered_returns_empty_string(self):
        self.assertEqual(format_resume_prefix([]), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
