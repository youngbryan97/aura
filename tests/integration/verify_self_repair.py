################################################################################


import logging
import sys
import os

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skills.self_repair import SelfRepairSkill

def test_missing_args():
    print("🧪 Testing Self-Repair with missing args...")
    skill = SelfRepairSkill()
    
    # 1. Empty goal
    result = skill.execute(goal={}, context={})
    print(f"Result (empty): {result}")
    
    # 2. Params without component
    result = skill.execute(goal={"params": {}}, context={})
    print(f"Result (empty params): {result}")
    
    # 3. Valid args failure (should return error, not crash)
    result = skill.execute(goal={"component": "non_existent_file.py"}, context={})
    print(f"Result (bad component): {result}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_missing_args()


##
