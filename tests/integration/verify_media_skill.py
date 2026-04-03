################################################################################

import logging
import unittest
from unittest.mock import MagicMock, patch
from skills.media_generation import MediaGenerationSkill

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestMedia")

class TestMediaSkill(unittest.TestCase):
    
    def setUp(self):
        self.skill = MediaGenerationSkill()
        # Mock client to avoid real API calls
        self.skill.client = MagicMock()
        self.skill.client.check_health.return_value = True
        self.skill.client.generate_image.return_value = "https://mock.openai.com/image.png"

    def test_image_generation(self):
        goal = {"objective": "Generate a futuristic city"}
        context = {}
        
        result = self.skill.execute(goal, context)
        
        self.assertTrue(result["ok"])
        self.assertEqual(result["url"], "https://mock.openai.com/image.png")
        self.assertEqual(result["type"], "image")
        
        # Verify client call
        self.skill.client.generate_image.assert_called_with("Generate a futuristic city", quality="standard")
        print("✅ TEST PASSED: Image generation skill execution.")

    def test_missing_prompt(self):
        goal = {"params": {}}
        result = self.skill.execute(goal, {})
        self.assertFalse(result["ok"])
        print("✅ TEST PASSED: Missing prompt handling.")

if __name__ == "__main__":
    unittest.main()


##
