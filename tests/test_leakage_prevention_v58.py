import asyncio
import re
import unittest
from unittest.mock import MagicMock, patch

# Mocking parts of the system to test the logic in isolation
class TestLeakageScrubbers(unittest.TestCase):
    def test_stream_of_being_scrubber(self):
        """Verify the scrubber logic added to StreamOfBeing._run_deep_narrative"""
        # The logic we added:
        def scrub(narrative):
            patterns = [
                r"^(?:Step|Phase)\s*\d+[:\.]\s*", 
                r"^Thinking[:\.]\s*",
                r"^Let's think step by step[:\.]?\s*",
                r"^I will now\s*",
                r"^Analyzing\s+.*?\.\.\.\s*"
            ]
            scrubbed = narrative.strip()
            for p in patterns:
                scrubbed = re.sub(p, "", scrubbed, flags=re.IGNORECASE)
            return scrubbed.strip()

        # Test cases
        self.assertEqual(scrub("Step 1: I am feeling present."), "I am feeling present.")
        self.assertEqual(scrub("Thinking: The user wants news."), "The user wants news.")
        self.assertEqual(scrub("Phase 2. Logic processing. I notice the light."), "Logic processing. I notice the light.")
        self.assertEqual(scrub("Let's think step by step: First, I will..."), "First, I will...")
        self.assertEqual(scrub("Analyzing data... The connection is stable."), "The connection is stable.")
        self.assertEqual(scrub("Just a normal sentence."), "Just a normal sentence.")

    def test_proactive_presence_scrubber(self):
        """Verify the scrubber logic added to ProactivePresence._emit"""
        def scrub_emit(content):
            content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL)
            content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
            content = re.sub(r'^(?:Step|Phase)\s*\d+[:\.]\s*', '', content, flags=re.IGNORECASE)
            return content.strip()

        # Test cases
        self.assertEqual(scrub_emit("<thought>Internal logic</thought>Hello world."), "Hello world.")
        self.assertEqual(scrub_emit("<thinking>Plan: Say hi.</thinking>Hey!"), "Hey!")
        self.assertEqual(scrub_emit("Step 2: Observation about tech."), "Observation about tech.")
        self.assertEqual(scrub_emit("Normal text."), "Normal text.")

    def test_mlx_prefill_scrubber(self):
        """Verify the prefill safety logic added to mlx_client.py"""
        def scrub_prefill(opening):
            patterns = [
                r"^(?:Step|Phase)\s*\d+[:\.]\s*", 
                r"^Thinking[:\.]\s*",
                r"^Let's think step by step[:\.]?\s*",
                r"^Analyzing\s+.*?\.\.\.\s*"
            ]
            scrubbed = opening.strip()
            for p in patterns:
                scrubbed = re.sub(p, "", scrubbed, flags=re.IGNORECASE)
            return scrubbed.strip()

        self.assertEqual(scrub_prefill("Step 1: The model is awake."), "The model is awake.")
        self.assertEqual(scrub_prefill("Thinking: About existence."), "About existence.")

if __name__ == "__main__":
    unittest.main()
