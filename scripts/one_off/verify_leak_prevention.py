import asyncio
import re
from typing import Optional

# Mocking the necessary parts for testing
class MockThoughtPacket:
    def __init__(self, stance: str, primary_points: list, tone: str = "direct"):
        self.stance = stance
        self.primary_points = primary_points
        self.tone = tone
        self.model_tier = "local"
        self.length_target = "medium"
        self.ask_followup = False
        self.followup_question = None
    
    def to_system_prompt(self):
        return f"STANCE: {self.stance}\nPOINTS: {self.primary_points}"

# Testing the scrubbing logic directly
def test_scrubbing():
    from core.synthesis import strip_meta_commentary
    
    test_cases = [
        (
            "[Integrated Neural Thought Environment] INTERNAL STATE: I feel clear-headed.\nHello there!",
            "Hello there!"
        ),
        (
            "DOMAIN: relational\nSTRATEGY: converse\nCOMPLEXITY: moderate\nI think so too.",
            "I think so too."
        ),
        (
            "INTERNAL STATE: contemplative\n[Mood: active]\nHey Bryan.",
            "Hey Bryan."
        ),
        (
            "### INTERNAL STATE\nSome internal thoughts.\nActual response here.",
            "Actual response here."
        ),
        (
            "Goal: Investigating emergent behaviors.\n* Next Steps: Fix the bug.\nI'm ready.",
            "I'm ready."
        )
    ]
    
    print("--- Testing Scrubbing Logic ---")
    for input_text, expected in test_cases:
        result = strip_meta_commentary(input_text)
        if result == expected:
            print(f"PASSED: {input_text[:30]}... -> {result}")
        else:
            print(f"FAILED: {input_text[:30]}...\n  Got: {result}\n  Expected: {expected}")

# Testing the fallback logic
async def test_fallbacks():
    from core.language_center import LanguageCenter
    from core.inner_monologue import ThoughtPacket
    
    lc = LanguageCenter()
    # Mocking router as None to trigger fallback
    lc._router = None 
    lc._fallback_mode = True
    
    thought = ThoughtPacket(
        stance="DOMAIN: relational STRATEGY: converse",
        primary_points=["Goal: Investigating something."],
        tone="direct"
    )
    
    print("\n--- Testing LanguageCenter Fallback ---")
    response = lc._fallback_response(thought, "Hello")
    if "DOMAIN" in response or "Goal" in response:
        print(f"FAILED: Fallback leaked metadata: {response}")
    else:
        print(f"PASSED: Fallback is clean: {response}")

if __name__ == "__main__":
    test_scrubbing()
    asyncio.run(test_fallbacks())
