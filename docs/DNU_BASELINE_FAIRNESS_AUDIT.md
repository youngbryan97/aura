# DNU AGI battery — baseline fairness audit

**Date:** 2026-07-06
**Trigger:** Operator (Bryan) flagged the `artifacts/current/agi_live/` result as
suspicious: three baselines (raw LLM, LLM+tools, ReAct) all score exactly
16.67% (2/12) while `full_aura` scores 100%, and every subsystem ablation
except System 2 leaves the score at 100%. He was right to be suspicious. This
is the audit.

## What the numbers were

- `BASELINES.json`: raw_llm / llm_with_tools / react_agent = **0.1667** (2/12).
- `ABLATIONS.json`: full_aura = **1.0**; remove persistent memory, volition,
  Will authority, self-repair, or affect-steering → **still 1.0** each; remove
  System 2 (the MCTS/structured planner) → **0.4167**.
- `SCORECARD.json`: full battery (100 tasks) = 100/100.

## Finding: the baseline comparison was NOT fair

The grader is identical for both conditions (`grade_result`: salted hash of the
normalized answer), so scoring is consistent. The unfairness is entirely in the
**execution conditions**, and there are two independent mechanisms:

### 1. The baseline was strangled at 160 tokens

`DNU_BASELINE_MAX_TOKENS = 160`. The baseline system prompt says *"Think
step-by-step. Put your final answer strictly inside `<answer>` tags."* But 160
tokens cannot hold a step-by-step derivation for a coding, planning, self-debug,
or transfer task. The model runs out of tokens before emitting an `<answer>`
tag and is scored `no_answer`. Example task (coding pack): *"Trace the execution
of this complex test case … copy-on-write slice wrappers …"* — genuine
multi-step tracing, impossible in 160 tokens.

Meanwhile `full_aura` (`execute_task`) runs the same prompt through the live
message path with a **240s budget and effectively unbounded tokens**.

Comparing a 160-token condition against an unbounded one, on tasks that require
extended reasoning, manufactures the gap.

### 2. full_aura had a deterministic symbolic solver; baselines did not

When a prompt carries `<answer>` tags and the structured solver is enabled,
`full_aura` calls `core.reasoning.proof_answer_solver.solve_strict_proof_prompt`
— a deterministic parser/solver that computes the exact answer (e.g., the R001
logic puzzle "Alice/Bob/Carol" is solved by constraint satisfaction, not by the
model). The baselines get only the raw model. So on the solvable tasks,
full_aura receives a guaranteed-correct answer while the baseline reasons under
a token cap.

This is `full_aura`'s legitimate capability — but it means the headline is
"32B + symbolic solver, unbounded" vs "32B, 160 tokens, no solver," not "the
mind architecture reasons where a comparable model can't."

## Finding: this battery isolates System 2, and the artifact says so

The ablation result (only System 2 moves the score) is **real and honestly
disclosed in the artifact itself**. `ABLATIONS.json` records for
`no_persistent_memory`: `lesion_effect_verified_in_this_battery: false`,
`dnu_score_delta_required: false`, and the note *"DNU task isolation clears
turn-local memory before every task; continuity dependency is measured by
dedicated memory/continuity batteries."*

In other words: these tasks are per-task-isolated novel-reasoning / coding /
planning problems. They do **not** exercise persistent memory, volition, Will
authority, self-repair, or affect. So ablating those subsystems shows no delta
**on this battery** — not because they are inert everywhere (several are
independently verified real, causal mechanisms — see
`docs/ABLATION_LEGIBILITY.md`, `tools/ablation_runner.py`), but because this
battery does not test them. The only subsystem this battery exercises is the
System 2 planner/solver, and its ablation correctly drops the score.

## Verdict

Of the three hypotheses the operator raised:

1. *"too small/easy"* — partly. N=12 for the baseline comparison is small.
2. *"conditions not comparable"* — **confirmed.** The 160-token cap plus the
   solver asymmetry are the main drivers of the 100%-vs-16.67% headline.
3. *"most of the architecture isn't doing measurable work on these tasks, only
   the planner"* — **confirmed for this battery, by the ablations and the
   artifact's own admission.** It is a battery-scope fact, not proof the other
   subsystems are fake.

**The 100%-vs-16.67% headline overstates what this evidence demonstrates.** It
is honest as "System 2 symbolic reasoning + full budget beats a token-capped
raw model on structured-answer tasks." It is **not** honest as whole-mind or
AGI-candidate evidence, and it should not be cited that way.

## Corrective actions

1. **Baseline made fair** (this commit): `DNU_BASELINE_MAX_TOKENS` now defaults
   to 2048 (env `AURA_DNU_BASELINE_MAX_TOKENS`), a reasoning budget comparable
   to full_aura's. A fair baseline is the same model with a comparable budget
   and no architecture.
2. **Claim scoped** (`CLAIMS_MATRIX.md`, `artifacts/current/agi_live/DNU_AGI_PROOF.md`):
   annotate that this battery isolates System 2 symbolic reasoning and that the
   original baseline was token-handicapped.
3. **Honest re-run required.** The corrected numbers need a re-run under the
   exclusive proof runtime (the harness sets `exclusive_runtime_required: true`
   and disables the desktop). This cannot run concurrently with the live
   desktop / soak; it is a dedicated ~30–40 min run. Until then, the existing
   `artifacts/current/agi_live/BASELINES.json` headline is **superseded and
   under audit** — do not cite the 16.67% baseline as a fair comparison.

## What would make the comparison genuinely convincing

- A larger task set (N≥50 per condition) so a clean split is statistically
  meaningful.
- The baseline given the **same** token/time budget as full_aura (done).
- A report of baseline failure modes (`no_answer` vs `fail`) so token-starvation
  is visible and separable from capability.
- Batteries that actually exercise memory / Will / affect for those ablations to
  mean anything (they exist separately — cite those, not this one, for those
  subsystems).
