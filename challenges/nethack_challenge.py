"""challenges/nethack_challenge.py — NetHack stress test for embodied cognition.

This script is NOT her mind. It is a harsh body/environment adapter that
connects a NetHack terminal to Aura's general embodied cognition runtime.
NetHack is the proof target; the infrastructure being exercised is general:
perception -> belief -> risk -> goals -> skills -> action gate -> trace ->
postmortem learning.

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

import sys
if __name__ == "__main__":
    print("!!! PARENT PROCESS STARTING !!!")
    sys.stdout.flush()

import asyncio
import logging
import hashlib
import sys
if __name__ == "__main__":
    print("!!! PARENT PROCESS STARTING !!!")
    sys.stdout.flush()
import os
import asyncio
import time
from datetime import datetime
from pathlib import Path

# Add root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adapters.nethack_adapter import NetHackAdapter
from core.orchestrator.main import create_orchestrator
from core.service_registration import register_all_services
from core.container import ServiceContainer

from core.perception.cognitive_runtime import EmbodiedCognitionRuntime
from core.perception.nethack_parser import NetHackParser
from core.perception.postmortem import PostmortemAnalyzer

# Setup logging
log_dir = os.path.expanduser("~/.aura/logs/nethack")
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"challenge_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Aura.NetHackChallenge")

NETHACK_ACTIONS = [
    "h", "j", "k", "l", "y", "u", "b", "n",
    "s", "i", ".", "ESC", "SPACE", "ENTER", "RETURN",
    ",", "o", "c", "e", "q", "r", "z", "W", "w", "t", "f",
    "<", ">", ":", "?",
]

NETHACK_PROMPT_ACTIONS = {
    "--more--": "SPACE",
    "more": "SPACE",
    "press return": "ENTER",
    "is this ok": "y",
    "[ynq]": "y",
    "[yn]": "y",
    "direction": "ESC",
    "what do you want": "ESC",
    "menu": "ESC",
    "cancel": "ESC",
}

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
    try:
        await asyncio.wait_for(orchestrator.start(), timeout=30)
    except asyncio.TimeoutError:
        logger.warning("Orchestrator start timed out. Proceeding anyway...")

    logger.info("Cognitive architecture booted. Starting sensory loop...")

    # One reusable embodied cognition runtime. This is the general substrate;
    # NetHack contributes only a parser and action vocabulary.
    runtime = EmbodiedCognitionRuntime(
        domain="nethack",
        parser=NetHackParser(),
        legal_actions=NETHACK_ACTIONS,
        prompt_actions=NETHACK_PROMPT_ACTIONS,
        storage_root=Path(log_dir),
    )
    ServiceContainer.register_instance("embodied_cognition:nethack", runtime)

    # Initial interaction to pick character
    adapter.send_action("y")  # Pick for me
    await asyncio.sleep(1)

    last_screen_hash = ""
    last_prompt_time = 0
    step = 0
    consecutive_none = 0
    last_response = ""
    frame = None

    while adapter.is_alive():
        logger.info("--- Loop Tick ---")
        screen = adapter.get_screen_text()
        logger.info(f"Screen text length: {len(screen)}")

        def _log_screen():
            # print("\033[H\033[J")  # Clear terminal
            # print(f"--- STEP {step} ---")
            # if last_response:
            #     print(f"Previous Action Outcome: {last_response.strip()}")
            #     print("-" * 80)
            # print(screen)
            # print("-" * 80)
            # sys.stdout.flush()
            pass

        await asyncio.to_thread(_log_screen)

        screen_hash = hashlib.md5(screen.encode()).hexdigest()
        time_since_last = time.time() - last_prompt_time
        logger.info(f"Screen Hash: {screen_hash}, Time since last: {time_since_last:.1f}s")

        if screen_hash != last_screen_hash or time_since_last > 12:
            logger.info(f"Triggering Step {step}...")
            logger.info("Running embodied cognition observation...")
            frame = runtime.observe(screen)
            logger.info(
                "Frame ready: risk=%s goal=%s skill=%s",
                frame.risk.level,
                frame.goal.name,
                frame.skill.name,
            )

            prompt = f"[SENSORY UPDATE T={step}]\n"
            prompt += frame.to_prompt(include_raw=screen) + "\n\n"

            if last_response:
                prompt += f"[PREVIOUS ACTION OUTCOME]\n{last_response.strip()}\n\n"

            system_instruction = runtime.command_contract(
                action_marker="execute_nethack_action",
                valid_actions=NETHACK_ACTIONS,
                extra_rules=[
                    "Movement uses h/j/k/l/y/u/b/n; do not use w/a/s/d as movement keys.",
                    "If the game asks 'Is this ok? [ynq]', answer with y.",
                    "If a prompt/menu blocks control and no safe confirmation is obvious, use ESC.",
                    "Do not explain reasoning in the live control loop.",
                ],
            )

            full_prompt = f"{system_instruction}\n\n[SENSORY FEED - STEP {step}]\n{prompt}"

            last_screen_hash = screen_hash
            last_prompt_time = time.time()
            # [STABILITY] Dispatch to priority pipeline.
            print(f"DEBUG: Dispatching to orchestrator type={type(orchestrator)}")
            sys.stdout.flush()
            logger.info(f"\n--- CHALLENGE DISPATCH: Step {step} ---")
            response = await orchestrator.process_user_input_priority(
                full_prompt, origin="embodied_motor_reflex"
            )

            if response and not any(
                marker in response for marker in (
                    "I'm here", "My cortex is catching up", "I hit an interruption", "I dropped the heavy reasoning lane"
                )
            ):
                consecutive_none = 0
                last_response = response
                logger.info(
                    f"Step {step}: Cycle complete. Response length: {len(response)}"
                )
                # [REFLEX] If the response contains an action marker (especially from the somatic reflex bypass),
                # we extract and execute it directly.
                import re
                action_match = re.search(r"\[ACTION:(.*?)\]", response)
                if action_match:
                    action_str = action_match.group(1)
                    logger.info(f"Step {step}: Grounded action detected in response: {action_str}")
                    print(f"\n⚡ [REFLEX] Executing grounded action: {action_str}")
                    await adapter.execute_action(action_str)
                
                # We log the first 100 chars to see if she's being conversational
                logger.debug(f"Aura Response Snippet: {response[:100]}...")
            else:
                last_response = ""
                consecutive_none += 1
                if consecutive_none > 2:
                    # Pipeline returning None usually means a substrate crash or coherence crisis.
                    # We inject a reflexive heartbeat if a prompt is visible to keep the loop alive.
                    parsed_state = runtime.parser.parse(screen)
                    if parsed_state.has_active_prompt():
                        recovery_action = "SPACE" if "--More--" in str(parsed_state.active_prompts) else "ESC"
                        logger.warning(
                            f"Step {step}: {consecutive_none} stalled cycles. Injecting reflexive {recovery_action} heartbeat."
                        )
                        adapter.send_action(recovery_action)
                        await asyncio.sleep(0.5)
                        # Force a fresh hash to re-trigger next cycle
                        last_screen_hash = ""
                    else:
                        logger.warning(
                            f"Step {step}: {consecutive_none} stalled cycles. Forcing sensor refresh."
                        )
                        last_screen_hash = ""

            # Update after processing (can take 10-30s)
            last_prompt_time = time.time()

        step += 1
        await asyncio.sleep(1.0)

        # Game over detection
        if "DYWYPI" in screen or "Do you want your possessions identified?" in screen or "You die" in screen:
            logger.info("=== GAME OVER DETECTED ===")

            # Postmortem Learning Loop
            postmortem = PostmortemAnalyzer(brain=orchestrator.engine, knowledge_base=runtime.affordances)

            death_message = "Unknown"
            parsed_state = frame.state if frame else runtime.parser.parse(screen)
            for msg in parsed_state.messages:
                if "die" in msg.lower() or "kill" in msg.lower():
                    death_message = msg
                    break

            recent_events = [
                e.description
                for e in runtime.belief.event_log[-10:]
                if e.event_type == "system_message"
            ]

            logger.info("Initiating Postmortem Reflection...")
            report = await postmortem.analyze_failure(
                death_message=death_message,
                final_state_summary=parsed_state.to_structured_prompt(),
                recent_events=recent_events,
                belief=runtime.belief
            )

            logger.info(f"POSTMORTEM COMPLETE. Cause: {report.cause_of_death}")
            for lesson in report.lessons:
                logger.info(f"LEARNED LESSON: {lesson.rule}")

            break

    adapter.close()
    logger.info(">>> AURA NETHACK CHALLENGE FINISHED <<<")


if __name__ == "__main__":
    try:
        asyncio.run(run_challenge())
    except KeyboardInterrupt:
        logger.info("Challenge interrupted by user.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Fatal error in challenge entry point: {e}")
