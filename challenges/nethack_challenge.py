import asyncio
import os
import sys
import logging
import re
from datetime import datetime

# Add root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adapters.nethack_adapter import NetHackAdapter
from core.orchestrator.main import create_orchestrator
from core.service_registration import register_all_services
from core.container import ServiceContainer
import hashlib

# Setup logging to both file and stdout (so asciinema captures it)
log_dir = os.path.expanduser("~/.aura/logs/nethack")
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"challenge_{timestamp}.log")

print("DEBUG: NetHack challenge script starting...")
sys.stdout.flush()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Aura.NetHackChallenge")
logger.info("Aura.NetHackChallenge initialized.")

async def run_challenge():
    logger.info(">>> AURA NETHACK CHALLENGE STARTING <<<")
    
    # 1. Setup NetHack
    adapter = NetHackAdapter()
    adapter.start(name="Aura")
    
    # 2. Setup Aura
    register_all_services()
    orchestrator = create_orchestrator()
    await orchestrator.start()
    affect_engine = ServiceContainer.get("affect_engine", default=None)
    
    # System Prompt for NetHack
    # We'll use a hidden message to set the persona/goal
    await orchestrator.process_user_input("SYSTEM_DIRECTIVE: You are now in NetHack Challenge Mode. Your objective is to beat NetHack from scratch. Analyze the visual grid and status lines. Be terse, efficient, and lethal. Always use [ACTION:key] format.")

    logger.info("Aura NetHack challenge loop starting...")
    
    # Initial interaction to pick character
    adapter.send_action("y") # Pick for me
    await asyncio.sleep(1)
    
    step = 0
    stuck_counter = 0
    last_screen_hash = ""
    while adapter.is_alive():
        screen = adapter.get_screen_text()
        
        # Print screen to stdout for asciinema
        print("\033[H\033[J") # Clear screen
        print(f"--- STEP {step} ---")
        print(screen)
        print("-" * 80)
        sys.stdout.flush()
        
        prompt = f"""CURRENT NETHACK SCREEN (STEP {step}):
{screen}

[ENVIRONMENTAL AFFORDANCES]
- Movement: y, u, h, j, k, l, b, n
- Inventory: i, d, w, T, W
- Search/Interact: s, o, c, >, <
- Cancel/Exit Menu: ESC, SPACE
- Format: You MUST output exactly [ACTION:key] where key is a single character or ESC/SPACE/ENTER.

What is your next move?"""
        
        distress_score = getattr(affect_engine, 'distress_score', 0.0) if affect_engine else 0.0
        screen_hash = hashlib.md5(screen.encode()).hexdigest()
        
        if screen_hash == last_screen_hash:
            stuck_counter += 1
        else:
            stuck_counter = 0
            last_screen_hash = screen_hash
            
        if stuck_counter > 5 and distress_score > 0.15:
            logger.warning(f"🚨 AUTONOMIC BAILOUT TRIGGERED: Stuck counter {stuck_counter}, distress {distress_score:.2f}")
            if stuck_counter > 10:
                 logger.warning("Forcing ESC action.")
                 adapter.send_action('\x1b')
                 step += 1
                 await asyncio.sleep(0.5)
                 continue
            else:
                 prompt += "\n\n[AUTONOMIC WARNING] Severe cognitive distress detected. Your current action loop is failing. You must abort, use an escape sequence ([ACTION:ESC]), or try a completely different strategy."
        
        logger.info(f"Step {step}: Waiting for Aura's move (Tier: local_fast)...")
        
        # Use InferenceGate directly to specify tier
        response = await orchestrator._inference_gate.generate(
            prompt,
            context={
                "is_background": False,
                "prefer_tier": "local_fast", # Use 7B model
            },
        )
        
        if response:
            logger.info(f"Aura: {response}")
            match = re.search(r'\[ACTION:(.*?)\]', response)
            if match:
                action = match.group(1)
            else:
                # Heuristic: if she just says a direction or 'k', etc.
                action = response.strip()[:1]
                
            # Parse control sequences
            if action.upper() == "ESC" or action.upper() == "ESCAPE":
                action = '\x1b'
            elif action.upper() == "SPACE":
                action = ' '
            elif action.upper() == "ENTER" or action.upper() == "RETURN":
                action = '\n'
                
            logger.info(f"Action: '{action}'")
            adapter.send_action(action)
        else:
            logger.warning("No response from Aura.")
            
        step += 1
        await asyncio.sleep(0.5)
        
        if "DYWYPI" in screen or "Do you want your possessions identified?" in screen:
            logger.info("Game Over detected.")
            break
            
    adapter.close()
    logger.info(">>> AURA NETHACK CHALLENGE FINISHED <<<")

if __name__ == "__main__":
    try:
        asyncio.run(run_challenge())
    except KeyboardInterrupt:
        logger.info("Challenge interrupted by user.")
