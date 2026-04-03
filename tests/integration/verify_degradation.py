################################################################################


import sys
import os
import time

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from core.resilience.resilience.resilience.degradation import degradation_manager, FailureEvent, FailureType, SystemState

def test_degradation_logic():
    print("🧪 Testing Degradation Manager...")
    
    # 1. Initial State
    assert degradation_manager.current_state == SystemState.HEALTHY
    print("✅ Initial State: HEALTHY")
    
    # 2. Report Minor Failure (Should stay Healthy/Degraded)
    print("💥 Simulating Skill Failure...")
    degradation_manager.report_failure(FailureEvent(
        type=FailureType.SKILL_FAILURE,
        component="web_search",
        error_msg="Timeout",
        severity=0.3
    ))
    
    # Should be DEGRADED
    if degradation_manager.current_state == SystemState.DEGRADED:
        print("✅ State transitioned to DEGRADED (Correct)")
    else:
        print(f"❌ State is {degradation_manager.current_state} (Expected DEGRADED)")

    # 3. Report Critical Failure (Should go to SAFE_MODE)
    print("💥 Simulating Critical LLM Failure...")
    degradation_manager.report_failure(FailureEvent(
        type=FailureType.LLM_API_ERROR,
        component="CognitiveEngine",
        error_msg="Connection Refused",
        severity=0.9
    ))
    
    if degradation_manager.current_state == SystemState.SAFE_MODE:
        print("✅ State transitioned to SAFE_MODE (Correct)")
    else:
        print(f"❌ State is {degradation_manager.current_state} (Expected SAFE_MODE)")
        
    # 4. Verify Capability Lock
    print("🔒 Checking Capabilities in Safe Mode...")
    can_plan = degradation_manager.can_perform("complex_planning")
    if not can_plan:
        print("✅ Complex Planning Disabled (Correct)")
    else:
        print("❌ Complex Planning Enabled (Incorrect)")

if __name__ == "__main__":
    test_degradation_logic()


##
