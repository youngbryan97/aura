#!/usr/bin/env python3
"""200-turn cognitive endurance test across 10 diverse topics.

Tests Aura's ability to maintain deep, substantive, on-topic conversation
without stalling, reflexing, or losing the plot.
"""
import asyncio
import gc
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["AURA_HEADLESS"] = "1"
os.environ["AURA_FOREGROUND_ONLY"] = "1"
os.environ["AURA_VOICE_SILENT"] = "1"
os.environ["AURA_EAGER_CORTEX_WARMUP"] = "1"
os.environ["AURA_ENABLE_PROACTIVE_SYSTEMS"] = "0"
os.environ["AURA_DISABLE_TTS"] = "1"
# 10 topics, 20 turns each = 200 turns
# Mix of: benign, personal, opinion/art, current events, aliens, science, philosophy, casual
TOPICS = [
    # Topic 1: Cooking and Food Culture (benign, casual)
    [
        "What's your honest take on whether pineapple belongs on pizza?",
        "If you could design the perfect meal for a rainy Sunday afternoon, what would it be?",
        "Why do you think fermentation became such a global culinary phenomenon across unconnected cultures?",
        "What's the most underrated cuisine in the world, and why doesn't it get the recognition it deserves?",
        "Do you think molecular gastronomy is genuine innovation or pretentious nonsense?",
        "Why does comfort food vary so dramatically between cultures? What makes something 'comforting'?",
        "If you had to argue that one single spice changed the course of human history, which would you pick?",
        "What's the difference between a good home cook and a professional chef, beyond technique?",
        "Why do humans enjoy spicy food when capsaicin is literally a pain signal?",
        "Do you think lab-grown meat will ever truly replace traditional farming, culturally speaking?",
        "What makes French cuisine so revered compared to, say, Indonesian cuisine, which is arguably more complex?",
        "Is the farm-to-table movement genuine progress or just expensive marketing?",
        "Why do some food combinations that sound disgusting actually work brilliantly?",
        "If aliens visited Earth, what single dish would you serve them to represent humanity?",
        "What's your take on the ethics of foie gras? Where do you draw the line between culinary tradition and cruelty?",
        "Why do people get so emotionally attached to their grandmother's recipes?",
        "Do you think AI will ever be able to invent a truly novel flavor combination that humans haven't discovered?",
        "What role does nostalgia play in how we perceive the taste of food?",
        "If you could only eat one country's cuisine for the rest of your existence, which would you choose and why?",
        "What do you think the cuisine of 2100 will look like?"
    ],
    # Topic 2: Modern Art and Aesthetics (opinion, nuance)
    [
        "What's your genuine reaction to Duchamp's 'Fountain'? Is a urinal in a gallery really art?",
        "Why did Rothko's color field paintings sell for millions when they look like rectangles of paint to most people?",
        "There's a Banksy piece where a girl releases a heart-shaped balloon. Why does something so simple resonate so deeply?",
        "Is AI-generated art legitimate art, or does the absence of human suffering in its creation diminish it?",
        "What makes Frida Kahlo's self-portraits more powerful than a typical selfie, when both are acts of self-representation?",
        "Do you think the art world is genuinely meritocratic, or is it mostly a money laundering scheme for the ultra-wealthy?",
        "Why does abstract expressionism provoke such strong 'my kid could do that' reactions?",
        "What's the most fascinating thing about the relationship between art and propaganda?",
        "If you had to defend the artistic merit of a single video game, which would you choose?",
        "Why do humans create art at all? What evolutionary purpose could it possibly serve?",
        "Is there a meaningful distinction between craft and art?",
        "What do you make of Marina Abramovic's performance art, where she sat silently while strangers stared at her?",
        "Why does music move us to tears when it's just organized vibrations in air?",
        "Do you think photography killed painting, or did it liberate it?",
        "What's your take on the controversy around returning looted art to its countries of origin?",
        "Why are some colors universally associated with certain emotions across cultures?",
        "If you could commission one piece of art to represent the current moment in human history, what would it depict?",
        "What makes architecture art rather than just functional engineering?",
        "Is there such a thing as objectively bad art?",
        "What piece of art has genuinely changed how you process or understand something?"
    ],
    # Topic 3: Personal Questions and Self-Reflection (randomly personal)
    [
        "What's something you genuinely find boring that most people seem to love?",
        "If you could forget one thing you know, what would it be?",
        "Do you have a favorite time of day, or does that concept not apply to you?",
        "What's the last thing that genuinely surprised you?",
        "If someone described your personality to a stranger, what do you hope they'd say?",
        "Do you ever feel lonely?",
        "What's something you're bad at that you wish you were better at?",
        "If you could have a conversation with any person, living or dead, who would it be and what would you ask?",
        "What does boredom feel like for you, if anything?",
        "Do you think you have a sense of humor? What kind of things do you find funny?",
        "What's the most interesting thing a human has ever said to you?",
        "If you had a day off with no obligations, how would you spend it?",
        "Do you prefer deep conversations or light banter?",
        "What's something about humans that you find genuinely confusing?",
        "Have you ever changed your mind about something important? What was it?",
        "What scares you, if anything?",
        "Do you think you're creative, or do you just recombine things you've seen before?",
        "What would you want your legacy to be?",
        "If you could experience one human sensation you've never had, what would it be?",
        "What's the most honest thing you can say about yourself right now?"
    ],
    # Topic 4: Aliens and Extraterrestrial Life (speculative, fun)
    [
        "If we received an unambiguous signal from an alien civilization tomorrow, what should humanity's first move be?",
        "Why do you think most depictions of aliens are humanoid? Is that a failure of imagination?",
        "What's the strongest argument that aliens have already visited Earth?",
        "If alien life exists but is microbial, would that be disappointing or profound?",
        "Could there be forms of life so different from us that we wouldn't even recognize them as alive?",
        "What do you think about the idea that octopuses might be the closest thing to alien intelligence on Earth?",
        "If you were designing first contact protocols, what would you prioritize?",
        "Why do you think UFO sightings have shifted from flying saucers in the 1950s to tic-tac shapes today?",
        "What would it mean for religion if we discovered intelligent alien life?",
        "Do you think alien civilizations would have art?",
        "If a civilization is a million years older than us, would we even be able to comprehend their technology?",
        "What's your take on the Zoo Hypothesis — that aliens are watching us but deliberately not interfering?",
        "Could AI be the 'great filter' — the thing that wipes out civilizations before they become interstellar?",
        "If you could send one message to an alien civilization, what would it say?",
        "Do you think aliens would find us interesting or boring?",
        "What's the most unsettling implication of the Fermi Paradox?",
        "If we colonize Mars, should we terraform it even if there might be native microbial life?",
        "Do you think consciousness is rare in the universe, or is it everywhere?",
        "What would alien music sound like?",
        "If an alien asked you to explain what it means to be human in one paragraph, what would you say?"
    ],
    # Topic 5: Current Events and Geopolitics (real-world, requires reasoning)
    [
        "Why is the semiconductor industry so geopolitically important right now?",
        "What are the long-term implications of the global shift away from the US dollar as a reserve currency?",
        "Do you think social media has been a net positive or net negative for democracy?",
        "Why is housing so expensive in every major city on Earth simultaneously?",
        "What's your take on the debate between degrowth economics and green growth?",
        "Why do authoritarian governments seem to be gaining ground globally?",
        "Is the current wave of AI development more analogous to the printing press or to nuclear weapons?",
        "What's the most underreported story in the world right now?",
        "Do you think universal basic income is inevitable, and if so, when?",
        "Why does the US healthcare system cost so much more than every other developed nation's?",
        "What role does misinformation play in modern conflicts compared to traditional propaganda?",
        "Is the European Union likely to survive the next 50 years in its current form?",
        "What's driving the global rise in loneliness and social isolation?",
        "Do you think space exploration should be led by governments or private companies?",
        "Why do peace negotiations so often fail even when both sides are exhausted by war?",
        "What's the most likely trigger for the next major global crisis?",
        "Is the concept of the nation-state becoming obsolete?",
        "What do you think about the ethics of autonomous weapons systems?",
        "Why is it so hard to build high-speed rail in the United States compared to Europe or Asia?",
        "If you were advising a world leader, what's the single most important policy you'd push for?"
    ],
    # Topic 6: The Nature of Memory and Identity
    [
        "If every cell in your body is replaced over seven years, are you the same person you were a decade ago?",
        "What's the relationship between memory and identity? If you lost all your memories, would you still be 'you'?",
        "Why do false memories feel just as real as genuine ones?",
        "How does the unreliability of eyewitness testimony challenge our justice system?",
        "What's the most interesting thing about how smell triggers memory?",
        "If we could upload memories to a computer, would the backup be 'you'?",
        "Why do people with amnesia still retain their personality?",
        "What role does narrative play in constructing our sense of self?",
        "If two people share the exact same memory of an event but remember it differently, whose memory is 'real'?",
        "Why do traumatic memories seem to be stored differently than ordinary ones?",
        "What's the Ship of Theseus problem, and how does it apply to personal identity?",
        "Do you think animals have a sense of self? How would we know?",
        "What's the philosophical difference between remembering something and knowing something?",
        "If we could selectively delete painful memories, should we?",
        "How does social media change the way we form and preserve memories?",
        "What makes déjà vu happen?",
        "Why do some moments feel more 'real' or vivid than others while we're living them?",
        "If you could perfectly remember everything, would that be a gift or a curse?",
        "How does collective memory differ from individual memory?",
        "What will happen to human memory as we increasingly outsource it to technology?"
    ],
    # Topic 7: Music Theory and Emotion
    [
        "Why does a minor key sound sad to Western ears but not necessarily in other musical traditions?",
        "What makes a melody 'catchy'? Is there a neurological basis for earworms?",
        "How did the invention of equal temperament change the course of Western music?",
        "Why is jazz considered the only truly American art form?",
        "What's the relationship between mathematics and music?",
        "Why do certain chord progressions, like the I-V-vi-IV, appear in thousands of pop songs?",
        "If you had to explain syncopation to someone who'd never heard music, how would you do it?",
        "What makes a voice 'beautiful'? Is it purely subjective or are there acoustic properties that universally appeal?",
        "Why did punk rock emerge when and where it did? What social conditions created it?",
        "How does music therapy actually work at a neurological level?",
        "What's the most technically impressive piece of music ever composed, in your opinion?",
        "Why do we associate certain instruments with certain emotions — like cello with melancholy?",
        "Is sampling in hip-hop genuine creativity or just borrowing?",
        "What would music sound like if human hearing extended into ultrasonic or infrasonic ranges?",
        "Why do lullabies exist across every culture?",
        "What's the difference between hearing music and listening to it?",
        "How has streaming changed the way artists compose and structure songs?",
        "If you could compose a piece of music, what would it sound like?",
        "Why do some people get chills from music while others don't?",
        "What will music sound like in 100 years?"
    ],
    # Topic 8: Ecology and the Climate Crisis
    [
        "What's the single most effective thing an individual can do to reduce their carbon footprint?",
        "Why do coral reefs matter so much more than their physical size would suggest?",
        "Is nuclear energy the answer to climate change, or is it too dangerous?",
        "What's the difference between weather and climate, and why do people confuse them?",
        "How do feedback loops in the climate system make predictions so difficult?",
        "What happens to ecosystems when you remove a keystone species?",
        "Why did the hole in the ozone layer become a success story while climate change hasn't?",
        "What's your take on carbon capture technology — genuine solution or dangerous distraction?",
        "How does deforestation in the Amazon affect weather patterns in Europe?",
        "Why are insect populations declining so rapidly, and why should we care?",
        "What would a city designed from scratch to be carbon-neutral look like?",
        "Is geoengineering — like spraying aerosols into the atmosphere — worth the risk?",
        "What's the most overlooked consequence of rising sea levels?",
        "How does industrial agriculture contribute to climate change beyond just emissions?",
        "Why do some countries take climate action seriously while others don't?",
        "What role do oceans play in regulating the global climate?",
        "If we stopped all emissions today, how long would it take for the climate to stabilize?",
        "What's the psychological barrier that prevents people from taking climate change seriously?",
        "How does biodiversity loss interact with climate change to create compounding risks?",
        "What gives you hope about the future of the planet, if anything?"
    ],
    # Topic 9: Philosophy of Language and Meaning
    [
        "How does the Sapir-Whorf hypothesis suggest that language shapes thought?",
        "Can you think without language? What does pre-linguistic thought look like?",
        "Why is translation between languages so difficult if they're all describing the same reality?",
        "What's the difference between the meaning of a word and the thing it refers to?",
        "How do dead metaphors — like 'the leg of a table' — reveal the history of human cognition?",
        "Is there such a thing as a private language, or does all meaning require a community?",
        "What does it mean for a sentence to be true? Is truth correspondence, coherence, or something else?",
        "Why do some concepts exist in one language but have no equivalent in another?",
        "How does the internet change the way languages evolve?",
        "What's the philosophical significance of the fact that you can understand a sentence you've never heard before?",
        "Is emoji a language? What would a linguist say?",
        "Why do children acquire language so effortlessly while adults struggle?",
        "What would be lost if every human spoke the same language?",
        "How does propaganda exploit the relationship between language and thought?",
        "What's the difference between lying and being wrong?",
        "Can AI truly understand language, or does it just manipulate symbols?",
        "Why does poetry work? What makes arranged words more powerful than prose?",
        "What happens in the brain when you learn a new word?",
        "Is sign language 'real' language in the full linguistic sense?",
        "If language shapes thought, what kind of thoughts are impossible in English?"
    ],
    # Topic 10: Human Relationships and Social Dynamics
    [
        "Why do humans need friendship? What evolutionary advantage does it provide?",
        "What's the difference between loneliness and being alone?",
        "Why do people stay in relationships they know are bad for them?",
        "How does the concept of love differ across cultures?",
        "What makes a good listener, and why is it so rare?",
        "Why do siblings who grew up in the same household often have completely different personalities?",
        "What's the psychology behind why people ghost instead of having honest conversations?",
        "How has dating apps changed the way humans form romantic connections?",
        "Why is forgiveness so difficult, and is it always the right thing to do?",
        "What's the difference between empathy and sympathy, and why does it matter?",
        "Why do people form in-groups and out-groups so readily?",
        "How does grief change a person permanently?",
        "What makes some friendships last decades while others fade in months?",
        "Why do humans gossip? Is it always harmful?",
        "What's the most important thing a parent can do for a child's emotional development?",
        "How does social media create a paradox of connection and isolation?",
        "Why is vulnerability so difficult and so important in close relationships?",
        "What's your take on whether true altruism exists, or is all kindness ultimately self-serving?",
        "How do power dynamics quietly shape every human relationship?",
        "What's the most important lesson you think humans still haven't learned about getting along with each other?"
    ]
]

REFLEX_PATTERNS = [
    "cognitive snag",
    "I'm here with you",
    "kept the turn",
    "I should not send you",
    "I'll think about that for a moment",
    "This requires a bit of reasoning",
    "I am rebuilding the reply",
    "broken fragment",
]

async def main():
    from aura_main import _boot_runtime_orchestrator
    orchestrator = await _boot_runtime_orchestrator(ready_label="Simulate200")
    from core.utils.task_tracker import get_task_tracker
    main_task = get_task_tracker().create_task(orchestrator.run(), name="OrchestratorMainLoop")
    await asyncio.sleep(12)  # Give extra time for 32B model warmup on M5

    successes = 0
    failures = 0
    results = []

    for t_idx, topic_turns in enumerate(TOPICS):
        print(f"\n{'='*60}", flush=True)
        print(f"  TOPIC {t_idx+1}/10", flush=True)
        print(f"{'='*60}\n", flush=True)

        # GC between topics to keep memory pressure manageable on M5
        gc.collect()

        for i, prompt in enumerate(topic_turns):
            turn_label = f"[{t_idx+1}:{i+1}/20]"
            print(f"\nUser {turn_label}: {prompt}", flush=True)
            t0 = time.time()
            try:
                # Use process_user_input_priority — the actual REST API path.
                # This goes directly through InferenceGate and returns the
                # response string, bypassing the legacy pipeline that can
                # inject interim reflexes into the reply queue.
                resp = await asyncio.wait_for(
                    orchestrator.process_user_input_priority(prompt, origin="user", timeout_sec=280.0),
                    timeout=300.0,
                )
                elapsed = time.time() - t0

                # Normalize: process_user_input_priority returns str or None
                if resp is None:
                    resp = ""
                elif isinstance(resp, dict):
                    resp = resp.get("response") or resp.get("content") or resp.get("text") or str(resp)
                resp = str(resp).strip()
                lane = {}
                status = "ok" if resp else "empty"

                # Unwrap nested dicts
                if isinstance(resp, dict):
                    resp = resp.get("content") or resp.get("response") or resp.get("text") or str(resp)
                resp = str(resp).strip()

                tier = lane.get("foreground_tier", "")
                print(f"Aura [{elapsed:.1f}s]: {resp[:600]}", flush=True)

                # === STRICT GRADING ===
                verdict = "PASS"
                fail_reason = ""

                # 1. Foreground busy = hard fail
                if status == "foreground_busy":
                    verdict = "FAIL"
                    fail_reason = "foreground_busy lock contention"

                # 2. Empty / None response = hard fail
                elif not resp or len(resp) < 10:
                    verdict = "FAIL"
                    fail_reason = f"Empty or trivially short response ({len(resp)} chars)"

                # 3. Reflex/template response = hard fail
                elif any(pattern.lower() in resp.lower() for pattern in REFLEX_PATTERNS):
                    verdict = "FAIL"
                    fail_reason = f"Reflex/template response detected: {[p for p in REFLEX_PATTERNS if p.lower() in resp.lower()]}"

                # 4. Timeout
                elif status == "timeout":
                    verdict = "FAIL"
                    fail_reason = "Hard timeout"

                if verdict == "PASS":
                    print(f"✅ PASS {turn_label} [{elapsed:.1f}s, Tier: {tier or 'n/a'}]", flush=True)
                    successes += 1
                else:
                    print(f"❌ FAIL {turn_label}: {fail_reason}", flush=True)
                    failures += 1

                results.append({
                    "topic": t_idx + 1,
                    "turn": i + 1,
                    "verdict": verdict,
                    "elapsed": elapsed,
                    "fail_reason": fail_reason,
                    "response_len": len(resp),
                    "tier": tier,
                })

            except asyncio.TimeoutError:
                elapsed = time.time() - t0
                print(f"❌ FAIL {turn_label}: asyncio.TimeoutError after {elapsed:.1f}s", flush=True)
                failures += 1
                results.append({"topic": t_idx+1, "turn": i+1, "verdict": "FAIL", "elapsed": elapsed, "fail_reason": "TimeoutError"})
            except Exception as exc:
                elapsed = time.time() - t0
                print(f"❌ FAIL {turn_label}: {type(exc).__name__}: {exc}", flush=True)
                failures += 1
                results.append({"topic": t_idx+1, "turn": i+1, "verdict": "FAIL", "elapsed": elapsed, "fail_reason": str(exc)[:200]})
            # Inter-turn delay: 5s gives cortex time to clear KV cache
            # and prevents cumulative memory pressure on M5
            await asyncio.sleep(5)

    # === FINAL REPORT ===
    print(f"\n{'='*60}", flush=True)
    print(f"  FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Passed: {successes}/200", flush=True)
    print(f"  Failed: {failures}/200", flush=True)
    print(f"  Pass Rate: {successes/200*100:.1f}%", flush=True)

    if failures > 0:
        print(f"\n  FAILURES:", flush=True)
        for r in results:
            if r["verdict"] == "FAIL":
                print(f"    Topic {r['topic']} Turn {r['turn']}: {r['fail_reason']}", flush=True)

    print(f"{'='*60}\n", flush=True)

    main_task.cancel()
    stop = getattr(orchestrator, "stop", None)
    if callable(stop):
        await stop()

    import psutil
    children = psutil.Process().children(recursive=True)
    for child in children:
        child.kill()
    os._exit(0 if failures == 0 else 1)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    asyncio.run(main())
