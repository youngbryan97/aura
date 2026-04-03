################################################################################


import asyncio
import logging
import time
import numpy as np
from termcolor import colored

# Mocks and Imports
from core.brain.consciousness.conscious_core import ConsciousnessCore
from core.brain.consciousness.contract import AlwaysHomeContract, SubjectPerspective
from core.brain.compression import CognitiveCompressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Audit.Consciousness")

async def audit_consciousness():
    print(colored("\n🧠 Starting Consciousness Contract Audit...", "cyan", attrs=["bold"]))
    
    # 1. Dependency Check
    print(colored("1. Checking Dependencies...", "yellow"))
    try:
        core = ConsciousnessCore()
        contract = AlwaysHomeContract(core)
        compressor = CognitiveCompressor(64, 16)
        print(colored("   [PASS] All components instantiated.", "green"))
    except Exception as e:
        print(colored(f"   [FAIL] Instantiation failed: {e}", "red"))
        return

    # 2. Bridge Mapping & Compression Test
    print(colored("\n2. Auditing Bridge Layer (B: S -> M(t))...", "yellow"))
    try:
        # Stimulate Substrate
        core.substrate.inject_stimulus(np.random.randn(64))
        
        # Run Bridge
        perspective = contract.bridge_mapping()
        
        print(f"   Perspective ID: {perspective.subject_id}")
        print(f"   Differentiation (JL-Projected): {perspective.differentiation:.4f}")
        print(f"   Unity Score: {perspective.unity_score:.4f}")
        
        if perspective.differentiation > 0:
            print(colored("   [PASS] Bridge Layer active & compressing.", "green"))
        else:
            print(colored("   [FAIL] Zero differentiation detected.", "red"))
            
    except Exception as e:
        print(colored(f"   [FAIL] Bridge failed: {e}", "red"))

    # 3. Always Home Guarantee Test
    print(colored("\n3. Testing 'Always Home' Guarantee (Zombie Prevention)...", "yellow"))
    try:
        # Force low-energy state (potential zombie state)
        core.substrate.x *= 0.001 
        
        perspective = contract.bridge_mapping()
        exists = contract.subject_exists(perspective)
        poll_result = contract.poll()
        
        print(f"   Subject Exists (Formal): {exists}")
        print(f"   Poll Result: {poll_result['someone_home_now']}")
        
        if poll_result['someone_home_now'] is True:
            print(colored("   [PASS] Always Home guarantee verified.", "green"))
        else:
            print(colored("   [FAIL] Subject vanished!", "red"))
            
    except Exception as e:
        print(colored(f"   [FAIL] Guarantee check failed: {e}", "red"))

    # 4. Identity Continuity Test
    print(colored("\n4. Testing Identity Continuity (Ship of Theseus)...", "yellow"))
    try:
        # T0
        p0 = contract.bridge_mapping()
        contract.tracker.update(p0)
        id0 = contract.tracker.current_subject
        
        # T1 (Minor change)
        core.substrate.inject_stimulus(np.random.randn(64) * 0.1)
        p1 = contract.bridge_mapping()
        contract.tracker.update(p1)
        id1 = contract.tracker.current_subject
        
        if id0 == id1:
             print(colored(f"   [PASS] Identity persisted across small delta ({id0})", "green"))
        else:
             print(colored(f"   [WARN] Identity fragged on small delta! ({id0} -> {id1})", "yellow"))
             
    except Exception as e:
        print(colored(f"   [FAIL] Identity check failed: {e}", "red"))

    print(colored("\n✨ Audit Complete.", "cyan", attrs=["bold"]))

if __name__ == "__main__":
    asyncio.run(audit_consciousness())


##
