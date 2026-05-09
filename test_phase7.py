import asyncio
import json
import os
import shutil
from pathlib import Path

# Test JSON Regex
import re
def test_json_regex():
    bad_json = """
    {
      "project_name": "Test",
      "tasks": [
        { "description": "Step 1", "priority": 1 },
        { "description": "Step 2", "priority": 2 },
      ],
    }
    """
    clean = re.sub(r',\s*([\]}])', r'\1', bad_json)
    try:
        data = json.loads(clean)
        assert data["project_name"] == "Test", "JSON parsed incorrectly"
        print("✅ JSON Trailing Comma test passed!")
    except Exception as e:
        print("❌ JSON Regex failed:", e)

# Test Abstraction Engine Increment
async def test_abstraction_engine():
    from core.adaptation.abstraction_engine import AbstractionEngine
    
    # Create temporary file
    test_path = "data/test_principles.json"
    if os.path.exists(test_path):
        os.remove(test_path)
        
    engine = AbstractionEngine(storage_path=test_path)
    
    # Commit some principles
    await engine._commit_principle("Principle A")
    await engine._commit_principle("Principle B")
    
    # Increment Principle A
    await engine.increment_application_counts(["Principle A"])
    
    # Verify
    content = Path(test_path).read_text()
    parsed = json.loads(content)
    payload = parsed.get("payload") if isinstance(parsed, dict) else parsed
    
    for p in payload:
        if p["principle"] == "Principle A":
            assert p["application_count"] == 1, "Principle A count should be 1"
        elif p["principle"] == "Principle B":
            assert p["application_count"] == 0, "Principle B count should be 0"
            
    print("✅ Abstraction Engine Increment test passed!")
    if os.path.exists(test_path):
        os.remove(test_path)

if __name__ == "__main__":
    test_json_regex()
    asyncio.run(test_abstraction_engine())
