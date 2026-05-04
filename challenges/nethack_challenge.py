import asyncio
import os
import sys
import logging
import time
import re
import hashlib
from datetime import datetime

# Add root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adapters.nethack_adapter import NetHackAdapter
from core.orchestrator.main import create_orchestrator
from core.service_registration import register_all_services
from core.container import ServiceContainer

# Setup logging
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

# ── Motor Grounding ──────────────────────────────────────────
# Aura's cognitive pipeline emits [ACTION:x] markers in her response text.
# This function extracts them and grounds them into physical keystrokes
# via the adapter. This is analogous to action_grounding.py but for the
# NetHack embodiment — it closes the perception-cognition-action loop.
ACTION_RE = re.compile(r'\[ACTION:(.*?)\]', re.IGNORECASE)

SPECIAL_KEYS = {
    'ESC': '\x1b', 'ESCAPE': '\x1b',
    'SPACE': ' ',
    'ENTER': '\n', 'RETURN': '\n',
}

def extract_actions(response_text):
    """Extract all [ACTION:x] markers from Aura's cognitive response."""
    if not response_text:
        return []
    return ACTION_RE.findall(response_text)

def resolve_key(action_str):
    """Convert an action string to the physical key to send."""
    upper = action_str.strip().upper()
    if upper in SPECIAL_KEYS:
        return SPECIAL_KEYS[upper]
    # Single character actions
    raw = action_str.strip()
    if len(raw) == 1:
        return raw
    # Multi-char that isn't a special key — take first char as heuristic
    if raw:
        return raw[0]
    return None


async def run_challenge():
    logger.info(">>> AURA NETHACK CHALLENGE STARTING (EMBODIED DAEMON v2) <<<")
    
    # 1. Setup NetHack
    adapter = NetHackAdapter()
    adapter.start(name="Aura")
    ServiceContainer.register_instance("nethack_adapter", adapter)
    
    # 2. Setup Aura
    register_all_services()
    orchestrator = create_orchestrator()
    await orchestrator.start()
    
    # System directive — routed through her full cognitive pipeline
    await orchestrator.process_user_input_priority(
        "SYSTEM_DIRECTIVE: You are now embodied in a NetHack terminal session. "
        "You will receive periodic [NETHACK SENSORY UPDATE]s showing the current screen. "
        "To take actions, output [ACTION:key] markers where key is a single character. "
        "MOVEMENT: h=left, j=down, k=up, l=right, y=upleft, u=upright, b=downleft, n=downright. "
        "IMPORTANT: DO NOT use w/a/s/d for movement — those are game commands (w=wield, d=drop). "
        "OTHER: i=inventory, s=search, o=open, c=close, >=descend stairs, <=ascend stairs, .=wait. "
        "MENUS: SPACE or ENTER to advance --More-- prompts, ESC to cancel/exit menus. "
        "You may output multiple [ACTION:x] markers in one response. "
        "Focus on survival and exploration. Be terse.",
        origin="admin"
    )

    logger.info("Aura NetHack embodied daemon loop starting...")
    
    # Initial interaction to pick character
    adapter.send_action("y")  # Pick for me
    await asyncio.sleep(1)
    
    last_screen_hash = ""
    last_prompt_time = 0
    step = 0
    consecutive_none = 0
    
    while adapter.is_alive():
        screen = adapter.get_screen_text()
        
        # Print screen to stdout for asciinema
        print("\033[H\033[J")  # Clear screen
        print(f"--- STEP {step} ---")
        print(screen)
        print("-" * 80)
        sys.stdout.flush()
        
        screen_hash = hashlib.md5(screen.encode()).hexdigest()
        time_since_last = time.time() - last_prompt_time
        
        # Only ping her cognitive pipeline when screen changes or stagnation detected
        if screen_hash != last_screen_hash or time_since_last > 15:
            stagnation = ""
            if screen_hash == last_screen_hash and step > 0:
                stagnation = (
                    "\n[METACOGNITIVE ALARM] Screen unchanged. Your last action had no effect. "
                    "You may be stuck in a modal prompt. Try [ACTION:ESC] or [ACTION:SPACE]."
                )
            
            prompt = f"[NETHACK SENSORY UPDATE T={step}]\nCURRENT SCREEN:\n{screen}{stagnation}"
            
            last_screen_hash = screen_hash
            last_prompt_time = time.time()
            
            logger.info(f"Step {step}: Sending sensory update to cognitive pipeline...")
            response = await orchestrator.process_user_input_priority(prompt, origin="external")
            logger.info(f"Step {step}: Aura responded: {response}")
            
            # ── MOTOR GROUNDING ──────────────────────────────────────
            # Extract [ACTION:x] markers and execute them physically
            if response:
                consecutive_none = 0
                actions = extract_actions(response)
                if actions:
                    for i, action_str in enumerate(actions):
                        key = resolve_key(action_str)
                        if key:
                            display = action_str.strip().upper() if action_str.strip().upper() in SPECIAL_KEYS else action_str.strip()
                            logger.info(f"  MOTOR: Executing action {i+1}/{len(actions)}: '{display}'")
                            adapter.send_action(key)
                            await asyncio.sleep(0.3)  # Small delay between actions
                    consecutive_no_action = 0
                else:
                    # She responded but didn't emit action markers
                    consecutive_no_action = getattr(run_challenge, '_no_action_count', 0) + 1
                    run_challenge._no_action_count = consecutive_no_action
                    logger.warning(f"Step {step}: Response had no [ACTION:x] markers ({consecutive_no_action} consecutive).")
                    
                    if consecutive_no_action >= 3:
                        # Force a default exploratory action to prevent total paralysis
                        logger.warning(f"Step {step}: Forcing exploratory move (l=right) after {consecutive_no_action} non-actionable responses.")
                        adapter.send_action('l')
                        run_challenge._no_action_count = 0
                    else:
                        # Re-prompt immediately with an action reminder
                        reminder = (
                            f"[MOTOR FEEDBACK] Your response had no [ACTION:x] markers. "
                            "You MUST output at least one [ACTION:key] to interact with the game. "
                            "Example: [ACTION:l] to move right, [ACTION:k] to move up. What is your next action?"
                        )
                        retry = await orchestrator.process_user_input_priority(reminder, origin="external")
                        if retry:
                            retry_actions = extract_actions(retry)
                            for i, action_str in enumerate(retry_actions):
                                key = resolve_key(action_str)
                                if key:
                                    display = action_str.strip().upper() if action_str.strip().upper() in SPECIAL_KEYS else action_str.strip()
                                    logger.info(f"  MOTOR (retry): Executing action {i+1}/{len(retry_actions)}: '{display}'")
                                    adapter.send_action(key)
                                    await asyncio.sleep(0.3)
                            if retry_actions:
                                run_challenge._no_action_count = 0
            else:
                consecutive_none += 1
                if consecutive_none > 2:
                    # Pipeline returned None repeatedly (dedup or will gate) — force dedup reset
                    logger.warning(f"Step {step}: {consecutive_none} consecutive None responses. Injecting tick marker.")
                    last_screen_hash = ""  # Force re-send with fresh hash comparison
            
            # Update prompt time after processing (processing can take 10-30s)
            last_prompt_time = time.time()
        
        step += 1
        await asyncio.sleep(1.0)
        
        # Game over detection
        if "DYWYPI" in screen or "Do you want your possessions identified?" in screen:
            logger.info("=== GAME OVER DETECTED ===")
            break
    
    adapter.close()
    logger.info(">>> AURA NETHACK CHALLENGE FINISHED <<<")

if __name__ == "__main__":
    try:
        asyncio.run(run_challenge())
    except KeyboardInterrupt:
        logger.info("Challenge interrupted by user.")
