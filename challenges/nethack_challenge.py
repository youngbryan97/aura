"""challenges/nethack_challenge.py — Pure Sensory Daemon.

This script is NOT her mind. It is her body — the environmental daemon
that bridges the NetHack terminal to her cognitive pipeline.

Architecture:
  1. Launch NetHack via the adapter
  2. Register the adapter in ServiceContainer so her NetHackSkill can reach it
  3. Inject screen updates into process_user_input_priority as sensory data
  4. She processes the screen through her full cognitive pipeline:
     - ProprioceptiveLoop → affect → metacognition → response generation
     - Her response naturally calls [ACTION:execute_nethack_action] markers
     - action_grounding.py dispatches them through the capability engine
     - The skill executes the keystroke and returns proprioceptive feedback
     - FeedbackProcessor tracks success/failure → body schema → affect
  5. This script does NOT parse her output at all. It just feeds her senses.

What this script does NOT do (and why):
  - Parse [ACTION:x] — that's action_grounding.py's job
  - Route feedback — that's FeedbackProcessor + ProprioceptiveLoop's job
  - Detect stagnation — that's detect_action_stagnation()'s job
  - Re-prompt on failure — that's metacognition's job
"""

import asyncio
import os
import sys
import logging
import time
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

print("DEBUG: NetHack sensory daemon starting...")
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


async def run_challenge():
    logger.info(">>> AURA NETHACK CHALLENGE STARTING (SENSORY DAEMON v3) <<<")

    # ── 1. BODY: Launch NetHack terminal ──────────────────────
    adapter = NetHackAdapter()
    adapter.start(name="Aura")
    # Register so her NetHackSkill can reach the adapter
    ServiceContainer.register_instance("nethack_adapter", adapter)

    # ── 2. MIND: Boot her full cognitive architecture ─────────
    register_all_services()
    orchestrator = create_orchestrator()
    await orchestrator.start()

    # ── 3. EMBODIMENT DIRECTIVE ───────────────────────────────
    # Tell her she's embodied. She should use her standard tool-calling
    # mechanism — [ACTION:execute_nethack_action] — which action_grounding.py
    # will dispatch through the capability engine to the NetHackSkill.
    await orchestrator.process_user_input_priority(
        "SYSTEM_DIRECTIVE: You are now embodied in a NetHack terminal session. "
        "You will receive periodic sensory updates showing the current screen. "
        "To interact with the game, use the execute_nethack_action tool. "
        "Call it with a single key: h=left, j=down, k=up, l=right, "
        "y=upleft, u=upright, b=downleft, n=downright, "
        "i=inventory, s=search, o=open, c=close, >=descend, <=ascend, .=wait, "
        "ESC=cancel/exit menu, SPACE=advance --More-- prompt, ENTER=confirm. "
        "CRITICAL: DO NOT use w/a/s/d for movement (w=wield, a=redo, s=search, d=drop). "
        "The tool returns proprioceptive feedback: whether your action changed "
        "the screen, the game message, and your status. Use this feedback to "
        "decide your next action. "
        "Goal: Survive, explore, descend deeper. Be decisive and act.",
        origin="admin"
    )

    logger.info("Cognitive architecture booted. Starting sensory loop...")

    # Initial interaction to pick character
    adapter.send_action("y")  # Pick for me
    await asyncio.sleep(1)

    last_screen_hash = ""
    last_prompt_time = 0
    step = 0
    consecutive_none = 0

    while adapter.is_alive():
        screen = adapter.get_screen_text()

        # Print screen to stdout for asciinema recording
        print("\033[H\033[J")  # Clear terminal
        print(f"--- STEP {step} ---")
        print(screen)
        print("-" * 80)
        sys.stdout.flush()

        screen_hash = hashlib.md5(screen.encode()).hexdigest()
        time_since_last = time.time() - last_prompt_time

        # ── SENSORY GATING ────────────────────────────────────
        # Only send a sensory update when:
        #   a) The screen has changed (new visual information)
        #   b) It's been >12s since last update (time-pressure)
        # This prevents flooding her cognitive pipeline with
        # duplicate information.
        if screen_hash != last_screen_hash or time_since_last > 12:
            # Extract key environmental signals for her
            lines = screen.split('\n')
            msg_line = lines[0].strip() if lines else ""
            non_empty = [l.strip() for l in lines if l.strip()]
            status = ""
            if len(non_empty) >= 2:
                status = non_empty[-2] + " | " + non_empty[-1]

            prompt = (
                f"[SENSORY UPDATE T={step}]\n"
                f"MESSAGE: {msg_line}\n"
                f"STATUS: {status}\n"
                f"SCREEN:\n{screen}"
            )

            last_screen_hash = screen_hash
            last_prompt_time = time.time()

            logger.info(f"Step {step}: Injecting sensory update...")

            # This goes through her FULL cognitive pipeline:
            # ProprioceptiveLoop → Affect → Metacognition →
            # Response Generation → Action Grounding → Tool Execution
            response = await orchestrator.process_user_input_priority(
                prompt, origin="external"
            )

            if response:
                consecutive_none = 0
                # We do NOT parse her response for actions.
                # action_grounding.py in _finalize_response handles that.
                # The tool result flows back through FeedbackProcessor.
                logger.info(
                    f"Step {step}: Cognitive cycle complete. "
                    f"Response length: {len(response)}"
                )
            else:
                consecutive_none += 1
                if consecutive_none > 3:
                    # Pipeline returning None = dedup blocking or will gate
                    # Force a fresh hash to re-trigger next cycle
                    logger.warning(
                        f"Step {step}: {consecutive_none} consecutive None "
                        "responses. Resetting screen hash."
                    )
                    last_screen_hash = ""

            # Update after processing (can take 10-30s)
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
