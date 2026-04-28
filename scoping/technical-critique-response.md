# Response to the technical-reviewer critique — what was applied

**Date:** 2026-04-27
**Frame:** Reviewer's critique was mostly correct. This is the change-log of fixes applied directly to the repo. Each entry says what changed and where.

## Applied tonight

### 1. README — license framing aligned with the actual LICENSE
**File:** `README.md`

- Badge changed from `License: Source Available` to `License: All Rights Reserved (Read-Only)` — the actual LICENSE prohibits copying, redistribution, derivative works, and commercial use, which is more restrictive than what "source-available" typically means.
- License section rewritten to describe the actual terms: read-only, no redistribution, no modification, no commercial use. Explicit pointer to `CITATION.cff` with the note that citation does not confer reuse rights.

### 2. README — new "What's stubbed and what's real" section
**File:** `README.md`, inserted between "Evidence boundary" and "What it is"

- Names `core/brain/llm/continuous_substrate.py` as a 100-line stub with hardcoded outputs.
- Notes that runtime CAA steering vectors are bootstrap approximations.
- States explicitly that φ is computed over cognitive-affective state nodes (not strict-IIT mechanisms).
- States explicitly that the published A/B steering result is from the 1.5B model, that the production system is 32B, and that the 32B replication is the credible artifact.
- States explicitly that STDP runs in a closed loop and external validation is pending.
- Calls out test-attestation as a separate work item.

### 3. CITATION.cff — license field corrected, caveats added
**File:** `CITATION.cff`

- `license: "Source Available"` → `license: "LicenseRef-AuraReadOnly"` (more accurate; the license is custom, not OSI/source-available).
- Added a `notes:` block clarifying that citation does not confer reuse rights and pointing readers to the README's stub-vs-real section before citing quantitative claims.

### 4. ARCHITECTURE.md — level-of-description note for φ
**File:** `ARCHITECTURE.md`, §3 "Integrated information"

- Added an explicit acknowledgment that φ is computed over cognitive-affective state nodes and sampled mesh neurons, not at the level of intrinsic mechanisms prescribed by IIT 4.0.
- States no claim is being made about strict integrated information in the Tononi/Albantakis/Haun sense.
- Notes that closing the level-of-description gap is intractable today and is on the open-research list.

### 5. ARCHITECTURE.md — STDP closed-loop caveat
**File:** `ARCHITECTURE.md`, §7 "STDP online learning"

- Added an explicit note that the reward signal is derived from the system's own outputs and the loop is therefore closed.
- States the trajectory-divergence result shows plasticity but not necessarily useful learning.
- Names the external-validation experiment (W matrix trained with vs. without environmental input) as the missing piece.

### 6. ARCHITECTURE.md — A/B steering scale caveat
**File:** `ARCHITECTURE.md`, §5 "Activation steering"

- Added a section noting the 1.5B → 32B activation-geometry gap.
- States the 32B replication with PCA visualization is the credible artifact and is the next scheduled work item.
- Reframes the 1.5B result as a methodology check, not as evidence of meaningful steering on the production system.

### 7. `continuous_substrate.py` — prominent stub marker
**File:** `core/brain/llm/continuous_substrate.py`

- Header docstring now contains a clearly-marked "STUB IMPLEMENTATION" block that names the placeholder behavior, lists what consumes the stubbed data, and points to the substrate-as-source roadmap doc.
- Anyone reading this file or grepping for "STUB" now finds it immediately.

## Not applied tonight (RAM-bound or values-decisions)

- **32B A/B steering replication with PCA visualization** — needs RAM that's reserved tonight; the next session.
- **Replacing LICENSE with a real source-available license (PolyForm Noncommercial / BSL)** — values decision; the README/CITATION fixes above take the conservative path of aligning to the actual restrictive LICENSE. Bryan can choose to flip to a real source-available license later if he wants academic engagement to be legally possible.
- **Per-test stub-vs-real attestation in TESTING.md** — sprint of work; named as a known gap.
- **De-stubbing `continuous_substrate.py`** — ~2 weeks of focused work; staged plan in `scoping/substrate-as-source-proposal.md`.
- **External-validation experiment for STDP** — design + several days of compute.
- **Per-claim dependency graph** — a few hours with `pyan`/`snakefood`; not blocking the immediate critique-response.

## What changed about the messaging posture

Before tonight: implicit assumption that readers would fairly weigh the README's narrower claims against the broader project framing.

After tonight: hostile-reviewer-friendly. Specific stubs are named in the README. Specific caveats are inline in ARCHITECTURE.md next to the claims they qualify. The license framing matches the actual license. Citing the work is allowed; it just doesn't confer reuse, and that's said up front.

The strongest LessWrong-post posture would now be: *"Here is the system. Here are the claims. Here are the limits — written out before you have to ask."*
