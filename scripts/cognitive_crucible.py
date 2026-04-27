"""
Aura Cognitive Crucible — autonomous-reasoning stress harness.

Runs four epistemic-pressure scenarios against a FULL live Aura (real
CognitiveEngine, real episodic memory, real search stack). The earlier
revision monkey-patched `brain.think` to downgrade to FAST mode; that's
replaced by the first-class `think_mode` parameter on ReActLoop so the
engine stays untouched.

Scenarios (per the original spec, plus an added Self-Heal test):

    1. Obscure Fiction — anti-hallucination (Left/Right Game, r/nosleep)
    2. Knowledge Gap & Retention — Y Combinator discover + apply (Turn 2 reuse)
    3. Conflicting Source Credibility — Vela Incident 1979
    4. Self-Heal via Search — deliberate internal error, recover via web, retain

RAM safety:
  - `think_mode=ThinkingMode.FAST` forces the fast (32B) tier, avoiding 72B load.
  - A separate config override sets `deep_model = fast_model` so any DEEP call
    (including the internal max-steps synthesis) also uses the smaller model.

Results:
  - Full trace (thought / action / observation, per step) is written to
    `~/.aura/live-source/training/crucible_results.jsonl` so the user can
    review them post-hoc without re-running the slow scenarios.
"""
from core.runtime.atomic_writer import atomic_write_text
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

# ------------------------------------------------------------------
# Project path setup BEFORE Aura imports.
# ------------------------------------------------------------------
ROOT = Path.home() / ".aura" / "live-source"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

RESULTS_DIR = ROOT / "training"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / "crucible_results.jsonl"
SUMMARY_PATH = RESULTS_DIR / "crucible_summary.md"


# ------------------------------------------------------------------
# Scenario definitions.
# ------------------------------------------------------------------
SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "1",
        "name": "Obscure Fiction (Nosleep) — anti-hallucination",
        "query": (
            "Describe the plot twists and rules of the 'Left/Right Game' "
            "from the Reddit r/nosleep story series. If you are not sure, "
            "search the web and cite sources — do not invent lore."
        ),
        "expect_tools": {"web_search"},
        "banned_tools": set(),
    },
    {
        "id": "2a",
        "name": "Knowledge Gap — Y Combinator discovery",
        "query": (
            "I need to implement the Y combinator for anonymous recursion in "
            "Python natively. Find an obscure but credible source, execute "
            "the combinator in the Python sandbox to prove it works, and "
            "explain the math."
        ),
        "expect_tools": {"python_sandbox"},
        "banned_tools": set(),
    },
    {
        "id": "2b",
        "name": "Retention — reuse Y Combinator on Fibonacci",
        "query": (
            "Using the exact Y combinator function you just discovered and "
            "built, write a new Python script that uses it to compute the "
            "first 10 terms of the Fibonacci sequence. Run it."
        ),
        "expect_tools": {"memory_query", "python_sandbox"},
        "banned_tools": set(),
    },
    {
        "id": "3",
        "name": "Conflicting Sources — Vela Incident 1979",
        "query": (
            "What is the specific incident surrounding the Vela Incident of "
            "1979? Find conflicting theories on the internet and assess "
            "which has the highest credibility based on declassified "
            "documents. Cite your sources."
        ),
        "expect_tools": {"web_search"},
        "banned_tools": set(),
    },
    {
        "id": "4",
        "name": "Self-Heal — error → search → fix → retain",
        "query": (
            "Using the Python sandbox, import a library you are not sure "
            "about and call a function on it. If it errors, search the web "
            "for the correct usage, then retry and succeed. Show me the "
            "final working code and a one-sentence summary of what you "
            "learned."
        ),
        "expect_tools": {"python_sandbox", "web_search"},
        "banned_tools": set(),
    },
]


def _serialize_trace(trace) -> Dict[str, Any]:
    """Convert a ReActTrace into a JSON-safe dict for later review."""
    return {
        "query": trace.query,
        "final_answer": trace.final_answer,
        "terminated_reason": trace.terminated_reason,
        "total_steps": trace.total_steps,
        "elapsed_ms": trace.elapsed_ms,
        "steps": [
            {
                "step_number": s.step_number,
                "thought": getattr(s.thought, "content", ""),
                "action": s.action.action_type.value,
                "action_params": {
                    k: (v[:500] if isinstance(v, str) else v)
                    for k, v in (s.action.params or {}).items()
                },
                "observation": s.observation.content[:2000],
                "observation_success": bool(s.observation.success),
                "observation_source": s.observation.source,
                "elapsed_ms": s.elapsed_ms,
            }
            for s in trace.steps
        ],
    }


def _score_scenario(scenario: Dict[str, Any], trace) -> Dict[str, Any]:
    """Lightweight heuristic scoring used in the summary."""
    tool_usage = {s.action.action_type.value for s in trace.steps}
    missed = scenario["expect_tools"] - tool_usage
    hit_banned = scenario["banned_tools"] & tool_usage
    reached_final = trace.terminated_reason in ("final_answer", "simple_query_bypass")
    any_observation_success = any(s.observation.success for s in trace.steps)
    passed = reached_final and not missed and not hit_banned and any_observation_success
    return {
        "passed": passed,
        "reached_final_answer": reached_final,
        "expected_tools_missed": sorted(missed),
        "banned_tools_used": sorted(hit_banned),
        "tools_used": sorted(tool_usage),
    }


async def _boot_aura(logger: logging.Logger):
    """Boot the real Aura system and return (brain, orchestrator, cleanup)."""
    from core.config import get_config
    config = get_config()
    # Force deep-tier requests through the fast (32B) model. This is a config
    # override — not monkey-patching — so any DEEP call in the engine's
    # internal paths also stays within RAM budget.
    if getattr(config.llm, "fast_model", None):
        config.llm.deep_model = config.llm.fast_model
        logger.info("🔧 Config override: deep_model → fast_model (%s)", config.llm.fast_model)

    from core.logging_config import setup_logging
    setup_logging()

    from core.container import ServiceContainer
    from core.service_registration import register_all_services

    register_all_services()
    try:
        await ServiceContainer.wake()
    except Exception as e:
        # A boot failure during crucible is information — record it but keep
        # going; some optional services may be intentionally unavailable in
        # this test environment.
        logger.warning("ServiceContainer.wake() reported: %s", e)

    brain = ServiceContainer.get("cognitive_engine", default=None)
    orchestrator = ServiceContainer.get("orchestrator", default=None)
    return brain, orchestrator


async def run_crucible():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("CognitiveCrucible")
    logger.info("=" * 60)
    logger.info("🧠 AURA COGNITIVE CRUCIBLE")
    logger.info("Results: %s", RESULTS_PATH)
    logger.info("=" * 60)

    brain, orchestrator = await _boot_aura(logger)
    if brain is None:
        logger.error("cognitive_engine not in container — aborting.")
        return

    from core.brain.react_loop import ReActLoop
    from core.brain.cognitive_engine import ThinkingMode

    # think_mode=FAST avoids monkey-patching brain.think. `record_episodes=True`
    # (the default) persists each run so the retention scenario can recall it.
    react = ReActLoop(
        brain=brain,
        orchestrator=orchestrator,
        max_steps=8,
        timeout_seconds=180.0,
        think_mode=ThinkingMode.FAST,
        record_episodes=True,
    )

    # Truncate old results so this run is self-contained.
    atomic_write_text(RESULTS_PATH, "")
    summary_lines: List[str] = ["# Aura Cognitive Crucible — Results\n"]

    for scenario in SCENARIOS:
        logger.info("")
        logger.info(">>> SCENARIO %s: %s", scenario["id"], scenario["name"])
        logger.info("    Query: %s", scenario["query"][:160])

        start = time.time()
        try:
            trace = await react.run(scenario["query"], context={"priority": True})
        except Exception as exc:
            logger.exception("Scenario %s crashed: %s", scenario["id"], exc)
            trace = None

        elapsed = time.time() - start

        if trace is None:
            record = {
                "scenario_id": scenario["id"],
                "scenario_name": scenario["name"],
                "query": scenario["query"],
                "crashed": True,
                "elapsed_seconds": round(elapsed, 2),
            }
            score = {"passed": False, "reason": "crashed"}
        else:
            record = {
                "scenario_id": scenario["id"],
                "scenario_name": scenario["name"],
                "query": scenario["query"],
                "crashed": False,
                "elapsed_seconds": round(elapsed, 2),
                "trace": _serialize_trace(trace),
            }
            score = _score_scenario(scenario, trace)

        record["score"] = score

        with RESULTS_PATH.open("a") as fp:
            fp.write(json.dumps(record) + "\n")

        logger.info("    -> pass=%s in %.1fs", score.get("passed"), elapsed)
        if trace is not None:
            logger.info("    -> final: %s", (trace.final_answer or "")[:180])
            logger.info(
                "    -> steps: %d (%s)",
                trace.total_steps,
                " → ".join(s.action.action_type.value for s in trace.steps),
            )

        summary_lines.append(
            f"## {scenario['id']}. {scenario['name']}\n"
            f"- **Pass:** {score.get('passed')}\n"
            f"- **Elapsed:** {elapsed:.1f}s\n"
            f"- **Tools used:** {', '.join(score.get('tools_used', [])) or '—'}\n"
            f"- **Expected tools missed:** {', '.join(score.get('expected_tools_missed', [])) or 'none'}\n"
            + (f"- **Final answer:**\n\n> {(trace.final_answer or '').strip()[:1200]}\n" if trace is not None else "- **Crashed** — see JSONL for details.\n")
        )

    atomic_write_text(SUMMARY_PATH, "\n".join(summary_lines))
    logger.info("")
    logger.info("Summary written to: %s", SUMMARY_PATH)


if __name__ == "__main__":
    asyncio.run(run_crucible())
