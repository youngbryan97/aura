"""The account's name is system metadata only where it is part of a path or a login (26 November 2026).

Bryan: "I think her email broke again." Past the gate that had held it for days, her reply to him was refused by her
executive as "epistemic_safety_violation: HARD BLOCK. Your message contained sensitive system paths, credentials, or
metadata": the computer's account bore his name, and the reply named him three times. Had it gone, the scrubber her
mail runs would have turned his name, and the part of his address that holds it, into "user".
"""
from __future__ import annotations

import pytest

from core.utils.privacy_hygiene import MetadataScrubber

pytestmark = pytest.mark.unit


@pytest.fixture
def scrubber() -> MetadataScrubber:
    scrub = MetadataScrubber()
    scrub.real_username, scrub.real_home = "alex", "/Users/alex"
    scrub.real_cwd, scrub.real_hostname = "/Users/alex/work/project", "alex-laptop.local"
    return scrub


@pytest.mark.parametrize("text", [
    "Hi alex, thanks for asking how I am.",
    "I wrote back to youngalex42@example.com this morning.",
    "alex said the build was green.",
])
def test_a_reply_that_names_the_person_is_sent_as_written(scrubber, text):
    assert scrubber.scrub_text(text) == text


@pytest.mark.parametrize("text, kept_out", [
    ("the log is in /Users/alex/logs/run.log", "/Users/alex"),
    ("look in /tmp/alex/cache", "/alex/"),
    ("I ran it as alex@alex-laptop.local", "alex@"),
    ("my working folder is /Users/alex/work/project", "/Users/alex"),
    ("token=abcdef123456 and 192.168.1.20", "abcdef123456"),
])
def test_a_path_a_login_or_a_secret_is_still_taken_out(scrubber, text, kept_out):
    assert kept_out not in scrubber.scrub_text(text)
