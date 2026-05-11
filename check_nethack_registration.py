import asyncio
from core.capability_engine import CapabilityEngine

def main():
    engine = CapabilityEngine()
    engine.reload_skills()
    skill_names = list(engine.skills.keys())
    if "execute_nethack_action" in skill_names:
        print("SUCCESS: execute_nethack_action is registered!")
    else:
        print("FAILED: execute_nethack_action is missing.")
        print(f"Registered skills: {skill_names}")

if __name__ == "__main__":
    main()
