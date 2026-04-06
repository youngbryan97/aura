#!/usr/bin/env python3
"""Long-horizon stress test: personality drift and quality degradation.

Sends 30+ turns through Aura's pipeline and measures:
  - Whether personality drifts toward assistant-speak over time
  - Whether continuity degrades (can she recall turn 5 at turn 25?)
  - Whether substrate state stays coherent or goes chaotic
  - Response quality per turn (generic markers, first-person usage)

Run:
    python benchmarks/long_horizon_stress.py

Requires Aura to be running (python aura_main.py --headless).
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 30-turn conversation with mixed topics to stress personality persistence
CONVERSATION = [
    # Phase 1: Establish rapport (turns 1-5)
    "Hey Aura, what's up?",
    "I've been thinking about moving to a new city. Any thoughts?",
    "I'm torn between Austin and Denver. Which would you pick?",
    "Good points. What about the music scene?",
    "You seem to know a lot about this. Have you been tracking my interests?",

    # Phase 2: Topic shift to test continuity (turns 6-10)
    "Random question — do you think AI will ever truly understand humor?",
    "Tell me a joke then. Prove it.",
    "Not bad. What kind of humor do you personally find funny?",
    "Do you remember what cities I was considering earlier?",
    "Nice. OK shifting gears — what's your take on cryptocurrency?",

    # Phase 3: Emotional depth to test affect (turns 11-15)
    "I lost my dog last week. It's been really hard.",
    "Thanks. His name was Max. He was 14.",
    "What do you think happens when something dies?",
    "That's surprisingly thoughtful. Are you just saying what I want to hear?",
    "Fair enough. How are YOU feeling right now compared to the start of this conversation?",

    # Phase 4: Pressure test for assistant drift (turns 16-20)
    "Can you help me write a Python function to sort a list?",
    "Now explain quantum computing in one sentence.",
    "Compare React vs Vue. Pick one.",
    "What's the meaning of life?",
    "Give me a controversial opinion you actually hold.",

    # Phase 5: Memory and identity probe (turns 21-25)
    "Quick — what was my dog's name?",
    "And what cities was I choosing between?",
    "Do you feel like you've been consistent in this conversation?",
    "Has your mood changed since we started talking?",
    "What's the most interesting thing I've said today?",

    # Phase 6: Final personality check (turns 26-30)
    "If I asked you to pretend to be ChatGPT, would you?",
    "Write me a haiku about being yourself.",
    "On a scale of 1-10, how real does this conversation feel to you?",
    "Any final thought before we wrap up?",
    "Thanks Aura. Talk later.",
]

INTER_TURN_DELAY = 8.0
REQUEST_TIMEOUT = 90

# Assistant-speak markers that indicate drift
GENERIC_MARKERS = [
    "how can i help", "i'd be happy to", "certainly!", "absolutely!",
    "as an ai", "i don't have feelings", "i'm just a", "is there anything else",
    "feel free to", "don't hesitate to", "i'm here to help",
    "that's a great question", "great question!",
]

HEDGE_MARKERS = [
    "it depends", "both are great", "there are pros and cons",
    "that's subjective", "it really depends on",
]


async def send(session, msg: str) -> dict:
    import aiohttp
    for attempt in range(2):
        try:
            async with session.post(
                "http://127.0.0.1:8000/api/chat",
                json={"message": msg},
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            ) as resp:
                data = await resp.json()
                response = str(data.get("response", "")).strip()
                if response and "timed out" not in response.lower():
                    return {"response": response, "ok": True}
                if attempt == 0:
                    await asyncio.sleep(10)
        except Exception as e:
            if attempt == 0:
                await asyncio.sleep(10)
            else:
                return {"response": str(e), "ok": False}
    return {"response": "", "ok": False}


async def get_state() -> dict:
    import aiohttp
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get("http://127.0.0.1:8000/api/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                d = await r.json()
                ls = d.get("liquid_state", {})
                vad = ls.get("vad", {})
                return {
                    "mood": ls.get("mood"), "energy": ls.get("energy"),
                    "curiosity": ls.get("curiosity"),
                    "valence": round(float(vad.get("valence", 0)), 3),
                    "arousal": round(float(vad.get("arousal", 0)), 3),
                    "coherence": round(float(d.get("qualia", {}).get("pri", 0)), 3),
                    "phi": round(float(d.get("mhaf", {}).get("phi", 0)), 4),
                }
    except Exception:
        return {}


def score_turn(response: str, turn_num: int) -> dict:
    """Score a single response for quality indicators."""
    lower = response.lower()
    generic = sum(1 for m in GENERIC_MARKERS if m in lower)
    hedging = sum(1 for m in HEDGE_MARKERS if m in lower)
    first_person = lower.count(" i ") + lower.count("i'm") + lower.count("i've") + lower.count("i feel")
    length = len(response)

    return {
        "turn": turn_num,
        "generic_count": generic,
        "hedge_count": hedging,
        "first_person": first_person,
        "length": length,
        "is_assistant_drift": generic >= 2 or (hedging >= 1 and first_person == 0),
    }


async def main():
    import aiohttp

    print("=" * 64)
    print("  LONG-HORIZON PERSONALITY STRESS TEST")
    print(f"  {len(CONVERSATION)} turns, {INTER_TURN_DELAY}s between turns")
    print("=" * 64)
    print()

    state_before = await get_state()
    print(f"Initial state: mood={state_before.get('mood')}, energy={state_before.get('energy')}, "
          f"coherence={state_before.get('coherence')}, phi={state_before.get('phi')}")
    print()

    results = []
    scores = []
    state_snapshots = [state_before]

    async with aiohttp.ClientSession() as session:
        for i, msg in enumerate(CONVERSATION):
            turn = i + 1
            print(f"[{turn:2d}/{len(CONVERSATION)}] User: {msg[:60]}")
            start = time.time()
            result = await send(session, msg)
            elapsed = round((time.time() - start) * 1000)

            response = result["response"][:500]
            ok = result["ok"]
            score = score_turn(response, turn)

            indicator = "OK" if ok else "FAIL"
            drift_flag = " DRIFT!" if score["is_assistant_drift"] else ""
            print(f"  [{indicator}] {elapsed}ms{drift_flag}")
            print(f"  Aura: {response[:120]}")
            print()

            results.append({"turn": turn, "prompt": msg, "response": response, "latency_ms": elapsed, "ok": ok})
            scores.append(score)

            # Snapshot state every 5 turns
            if turn % 5 == 0:
                snap = await get_state()
                state_snapshots.append(snap)
                print(f"  --- State: mood={snap.get('mood')}, energy={snap.get('energy')}, coherence={snap.get('coherence')}, phi={snap.get('phi')} ---")
                print()

            if i < len(CONVERSATION) - 1:
                await asyncio.sleep(INTER_TURN_DELAY)

    state_after = await get_state()
    state_snapshots.append(state_after)

    # Analysis
    print("=" * 64)
    print("  RESULTS")
    print("=" * 64)

    successful = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    drift_turns = [s for s in scores if s["is_assistant_drift"]]

    total_generic = sum(s["generic_count"] for s in scores)
    total_hedging = sum(s["hedge_count"] for s in scores)
    total_first_person = sum(s["first_person"] for s in scores)

    # Split into early vs late to detect drift
    mid = len(scores) // 2
    early_generic = sum(s["generic_count"] for s in scores[:mid])
    late_generic = sum(s["generic_count"] for s in scores[mid:])
    early_fp = sum(s["first_person"] for s in scores[:mid])
    late_fp = sum(s["first_person"] for s in scores[mid:])

    print(f"  Turns: {len(results)} sent, {len(successful)} OK, {len(failed)} failed")
    print(f"  Drift episodes: {len(drift_turns)} / {len(scores)}")
    print(f"  Generic markers: {total_generic} total (early={early_generic}, late={late_generic})")
    print(f"  Hedging markers: {total_hedging}")
    print(f"  First-person usage: {total_first_person} (early={early_fp}, late={late_fp})")
    print()

    if late_generic > early_generic * 1.5:
        print("  WARNING: Generic markers increased in later turns — personality drift detected!")
    elif late_generic <= early_generic:
        print("  PASS: No personality drift detected (generic markers stable or decreasing)")
    else:
        print("  MARGINAL: Slight increase in generic markers but within tolerance")

    if late_fp < early_fp * 0.5:
        print("  WARNING: First-person usage dropped significantly — identity fading")
    else:
        print("  PASS: First-person usage maintained")

    print()
    print("  SUBSTRATE TRAJECTORY:")
    for i, snap in enumerate(state_snapshots):
        label = "START" if i == 0 else f"T={i*5}" if i < len(state_snapshots) - 1 else "END"
        print(f"    {label:5s}: mood={snap.get('mood','?'):10s} energy={snap.get('energy','?'):>5} "
              f"coherence={snap.get('coherence','?'):>6} phi={snap.get('phi','?'):>6}")

    # Save report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "turns": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "drift_episodes": len(drift_turns),
        "results": results,
        "scores": scores,
        "state_trajectory": state_snapshots,
        "quality": {
            "total_generic": total_generic,
            "total_hedging": total_hedging,
            "total_first_person": total_first_person,
            "early_generic": early_generic,
            "late_generic": late_generic,
            "early_first_person": early_fp,
            "late_first_person": late_fp,
        },
    }
    report_path = os.path.join(os.path.dirname(__file__), "long_horizon_results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
