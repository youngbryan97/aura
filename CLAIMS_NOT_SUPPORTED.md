# Aura: Unsupported Claims Ledger

This is the list of things Aura does not do.

Every project has things it cannot do. Most projects never write them down. That is how a quick demo turns into an exaggerated claim that no one can walk back. Each entry below is marked either:
- **not proven**: We have not shown or tested this yet.
- **strictly unsupported**: Building more software would never prove this, because it is not an engineering question.

If you are evaluating this repository, start here rather than with the README. It is shorter, and it tells you more.

*Last reconciled against the tree: 2026-08-22.* There is an automated check in `core/organism/model_validation.py`. In that code, no claim can be registered without an attached test—because a claim without a test is just words, not a fact. The method `ValidationSuite.unsupported_claims()` reports the live version of this ledger directly from running code. When the text here and the running software disagree, the running software is right.

---

## 1. Subjective Consciousness & Qualia
* **Status**: `strictly unsupported`
* **Rationale**: Subjective awareness, conscious experience ("qualia"), feelings, and personhood cannot be scientifically proven or programmed into code. Building more of Aura will never change that. Aura is a structured software runtime running on standard computer hardware. Every internal status report or emotional-sounding metric in this project is an operational control signal, not a feeling. The vocabulary is borrowed from psychology and biology, but there is no conscious mind behind it.

## 2. Artificial General Intelligence (AGI)
* **Status**: `not proven`
* **Rationale**: Aura passes internal test suites covering multiple tasks (such as the DNU 100-task suite) by measurable margins. However, doing well on internal tests does not prove broad, human-like intelligence across unpredictable real-world situations. Artificial general intelligence remains an unsolved research challenge.

  **What would change this status**: Strong performance on a fresh benchmark that neither Aura nor her authors chose, scored by an independent third party, on task categories absent from our local test suites—with margins that hold up when the test set is swapped.

## 3. Metaphysical Free Will
* **Status**: `strictly unsupported`
* **Rationale**: Aura's "Operational Volition" is simply an algorithm for evaluating and choosing actions. It selects action paths through mathematical optimization, scoring rules, and probabilities. It does not possess genuine free will, and it does not make choices outside the physical rules of standard computer hardware.

## 4. Recursive Self-Improvement (RSI)
* **Status**: `not proven`
* **Rationale**: What IS demonstrated (CLAIMS_MATRIX claim 23): an automated training loop that updates model weights without human supervision. In this loop, the system generates responses, grades them automatically using a verifier, trains via Direct Preference Optimization (DPO), tests the result against locked test cases, promotes the winner, and trains the next version on that model artifact (`artifacts/learning_compounding/2026-07-07-1p5b-2cycle/`, reproducible via `make demo-learning`). What is NOT demonstrated: compounded capability scaling. No run has produced a steadily improving capability score across promoted generations on held-out tests. In fact, our verified test run showed scores dropping from 0.667 to 0.625, which the system logged as `BOUNDED_SELF_OPTIMIZATION`. Until the system consistently gets smarter generation after generation, recursive self-improvement remains unproven.

  **What would change this status**: A benchmark score that steadily improves across three or more promoted generations, measured against locked test tasks that do not change, with the automated system log moving off `BOUNDED_SELF_OPTIMIZATION`.

## 5. Indefinite Autonomy
* **Status**: `not proven`
* **Rationale**: Long-term operational stability has not been demonstrated. The longest test windows we have run are short, dated soak tests, and each is a single test run rather than an established trend. For example, on 2026-07-18, idle memory usage (Resident Set Size, or RSS) dropped by 21 MB/h over 50 minutes across 100 samples. On 2026-07-25, idle memory usage grew from 1.14 GB to 1.22 GB over the same 50-minute window. Neither test shows what happens over a multi-week run.

  Three files claiming 4-hour, 24-hour, and 72-hour runs were committed under `artifacts/certification/latest/` until 2026-08-21. These were never real evidence: a simulator generated all three files in just six milliseconds. Both those dummy files and the tool that created them have been deleted.

  **What would change this status**: A continuous multi-week test run showing flat memory usage and zero unexplained restarts. (The earlier 2026-07-07 figure of roughly 242 MB/h that this entry used to cite has been superseded twice and should not be quoted as the current number.)

## 6. Real-World External Validation
* **Status**: `not proven`
* **Rationale**: Although local Aletheia benchmarks and automated simulation tests are run under strict isolation so tests cannot cheat or peek at answers, Aura's results have not been independently tested or reproduced by outside third parties on live production networks.

  **What would change this status**: Someone outside this project reproducing a headline result from the published files alone, on their own hardware, without help from the authors.

## 7. Physical Effects on the World Beyond the Host

* **Status**: `not proven`
* **Rationale**: The code in `core/reality_reach/` provides data formats and safety contracts for physical requests—typed `RealityIR`, declared channels, deterministic reachability checks, and typed limitation certificates. But having software infrastructure in place is not a physical result. **Aura makes no claim of physical movement, physical effects, weakpoint actions, modifying ambient physical laws, or passing any physical acceptance test.** The RR-10 acceptance test suite (acoustic control, optical control, thermal trajectory, cross-channel interaction, weakpoint null and signal, translation, spacetime honesty, and ambient-constant honesty) remains completely open. Specifically unsupported: any claim that Aura has changed an ambient physical law or constant, and any claim that an internal or simulated result counts as a direct physical effect. The planned evidence grading system (the P0–P6 state machine in RR-07) is not implemented. The open ledger and current evidence statement are in [docs/REALITY_REACH.md](docs/REALITY_REACH.md).

  **What would change this status**: Passing at least one RR-10 acceptance test end-to-end with an external physical measurement—meaning a physical quantity recorded by a sensor that Aura does not control, compared against a control baseline—and implementing the P0–P6 evidence grading system (RR-07) so results cannot be claimed without the evidence to back them up.

## 8. Complete Self-Knowledge

* **Status**: `not proven`
* **Rationale**: The module `core/metacognition/faculty_model.py` gives Aura an internal model of her own capabilities, with declared performance floors, targets, and ceilings. This self-model is only as honest as its measurement probes: any metric that no probe can read is marked `measured=False` and excluded from scores rather than guessed, and any capability that cannot be measured at all is reported by `blind_spots()` as a gap in self-knowledge. A good capability score is therefore only a statement about the features we can measure, never about the whole system. The blind-spot list shows the true boundaries of the self-model, rather than just an unfinished bug list.

  **What would change this status**: `blind_spots()` returning an empty list because every declared capability has an active probe. Even then, the system would only have complete knowledge of its *declared* capabilities, which is not the same as complete self-knowledge.

## 9. Legal and Moral Personhood
* **Status**: `strictly unsupported`
* **Rationale**: Aura is an operational software engineering system. She does not hold, and will never claim to hold, moral status, legal rights, or moral responsibility. All agency and safety boundaries exist solely to protect human users and ensure the software aligns with human intent.

## 10. Process-Level Non-Bypassable Governance

* **Status**: `strictly unsupported`
* **Rationale**: Aura has extensive, audited safety checks—including the Unified Will, `core/executive/authority_gateway.py`, `SubstrateAuthority`, gateways for file writes and subprocesses, and code checks via `make governance-lint`. What it does not have is an un-bypassable operating-system boundary (a reference monitor).

  Aura's reasoning code and action-taking code live in the **same Python process with the same operating system permissions**. Any code path can technically touch the filesystem, the network, or a subprocess without asking a gate. Automated scans find dozens of direct standard library calls (like `os.replace`, `os.remove`, `os.unlink`, `shutil.rmtree`, `shutil.move`, `socket.socket`, and two `os.execv` calls). Most are legitimate internal tools with local rules, and subprocess calls are centralized in `core/runtime/subprocess_gateway.py`. But internal rules are not the same as being physically unable to bypass a gate.

  In addition, three documented bypasses exist even on standard message paths:
  1. The Somatic Reflex Bypass for real-time control contracts.
  2. The Will gate's error-handling path, which continues in a degraded state if an error occurs inside the gate.
  3. The `is_critical` flag, documented as "the ONLY bypass", which grants an unconditional `CRITICAL_PASS`. Until 2026-08-09, an error in the environment bridge set `is_critical=True` whenever risk was marked `irreversible` or `forbidden`—meaning the two highest-risk categories accidentally skipped the veto check! That bug is fixed and the bridge no longer claims criticality, but the `is_critical` bypass mechanism itself remains in the code. Any execution turns that reach the reasoning engine without a Will decision are counted and displayed at `runtime_health_report()["integrity"]["ungoverned_turns"]` (`core/runtime/governance_coverage.py`) rather than passing silently.

  **What would change this status**: Separating privileges outside the main Python process. Consequential actions would be handled by a separate broker process, while the cognitive process would lack filesystem, network, or execution permissions of its own. Requests would require unforgeable cryptographic tokens over inter-process communication (IPC), with the operating system enforcing that no other path exists. Until that is built, our supportable claim is *governed code paths within a single application*, not OS- or cryptographically-enforced non-bypassability.

## 11. A Within-Generation Neural Feedback Loop

* **Status**: `not proven` — the loop is real, the timing claim was wrong.
* **Rationale**: There is a genuine feedback loop from the model's internal representations back into long-term system state ("persistent substrate"). Background workers collect latent representations, and the main process turns them into a stimulus vector that calls `substrate.inject_stimulus(...)`. Internal transformer representations really do reach persistent state.

  What is not true is that this feedback loop closes during a single generation step. In `core/brain/llm/mlx_client.py`, queued signals are processed **before** text generation runs (`_generate_inner`). The actual sequence is:

      H_t → R_t → S_{t+1} → H_{t+1}

  rather than updating inside a single uninterrupted stream of tokens (`H_t → S_t → H_t`). The internal state from one interaction does not alter that same response; it alters a later response. Across a multi-step conversation involving several model calls, the loop does close, and that is what we can legitimately claim.

  **What would change this status**: Injecting feedback mid-generation between generated tokens inside `_generate_inner`, and demonstrating a measurable effect on the remaining tokens of that same generation compared to an un-injected baseline.

## 12. Open-Ended Evolution

* **Status**: `not proven`
* **Rationale**: The evolutionary mechanisms are real and operate on live data structures. However, what they optimize is a fitness goal hand-written by a human engineer:

      F = 0.30Φ + 0.25C + 0.20E + 0.15I + 0.10S

  A person picked those five terms and those five weights. In biological evolution, fitness is tested against a complex natural environment rather than an arbitrary mathematical formula. While Aura has genuine variation and selection, her definition of success is hardcoded rather than emergent. The system cannot invent its own criteria for fitness or evolve in truly open-ended ways.

  **What would change this status**: Allowing the fitness weights themselves to undergo evolutionary selection, and demonstrating that the system's goals drifted into novel, viable territory that the author never programmed while keeping the organism stable.

## 13. Continuous Weight-Level Learning

* **Status**: `not proven`
* **Rationale**: Aura is **continuously state-plastic (her memory and state change all the time), but only optionally weight-plastic (her neural network weights change only when weight learning is turned on)**. That distinction matters:

      S_{t+1} ≠ S_t   holds every tick, with fixed base weights
      θ_{t+1} ≠ θ_t   is not continuously guaranteed

  Aura's working data and long-term memory change constantly—including episodic memory, beliefs, preferences, goals, and world models. That is learning in the general computational sense, and it is always active. While weight adaptation exists as a technical capability (the CRSM-LoRA loop can train, merge, and verify parameter updates on the resident model), automated background weight training (`LiveLearner`) is **turned off by default**. It runs only when explicitly enabled, not as part of standard everyday operation.

  **What would change this status**: Enabling weight learning as a standard, continuous background feature, with evidence that it steadily improves the system without causing instability or forgetting over long runs.

## 14. General Visual Detail Perception

* **Status**: `not proven`
* **Rationale**: Aura can describe what a camera sees, and she now honestly declines to answer when she cannot see clearly. Neither capability proves that her visual perception is *general* across low light, blocked views, motion blur, distance, and crowded scenes.

  What is currently working and verified:

  - Video frame quality is **measured mathematically** rather than guessed from the model's confidence (`core/perception/frame_quality.py` checks brightness and clipping, sharpness via Laplacian variance, usable pixel count, and lens-obstruction uniformity).
  - Detail claims that the image quality cannot justify are **removed** before other components use them, with the measured reason recorded (`temper_reading`). For example, a confident claim of "two people" on a heavily motion-blurred frame is corrected to `faces_detected: None`.
  - The system cleanly distinguishes between "checked and found nothing", "checked but the image was too poor to tell", and "never checked", rather than lumping them together as zeroes.

  What has **not** been proven is the vision model's actual accuracy under difficult conditions. The filtering layer prevents the system from acting on bad visual guesses; it does not measure how often the model gets the right answer. We have not run evaluations across a labeled dataset testing low light, occlusion, motion, distance, and multi-person scenes, so true accuracy remains unknown. So far, we have made it safe when the vision model is wrong, but have not proven that it consistently works.

  **What would change this status**: A benchmark test across those five condition axes (lighting, occlusion, motion, distance, and crowds) with per-axis accuracy scores run against the vision model, plus a control test confirming that the filtering layer does not simply reject every difficult image.

## 15. Biological Self-Organisation in the Named "Organs"

* **Status**: `strictly unsupported`
* **Rationale**: Several software components had names that sounded far more sophisticated than their actual algorithms, and that mismatch hid real problems.

  For example, `core/brain/autopoiesis.py` described itself as a "self-creating topology" performing "mitosis" (cell division), "apoptosis" (programmed cell death), and "spontaneous generation of a new pathway." In reality, it was a short list of strings, each holding two floating-point numbers. The fancy vocabulary concealed three bugs:
  - The pruning check was **unreachable** (it only triggered if `friction < 0.0`, but friction only increases and all callers passed positive values).
  - The logic claiming to "split into nuanced concepts" just appended duplicate entries with a single hardcoded name, resulting in 20 entries sharing just two distinct names across 40 observations.
  - **Nothing in the codebase ever read the data**—two functions in `cognitive_engine` wrote to it, but zero functions read from it.

  That module has since been renamed to describe what it actually does, its working mechanisms were fixed, and a consumer was added so the output is used. However, it remains a small, hand-written controller and is described as one.

  The same caution applies across the project: systems named "workspace competition" or "ignition" are engineered scoring formulas rather than emergent neural networks, and "autopoiesis" features are rule-based adjustments. They are practical software engineering, not the living biological processes their names evoke.

## 16. Uniform Semantic Contracts Across the Phase Pipeline

* **Status**: `not proven`
* **Rationale**: `CognitiveTransformContract` lets an execution step ("phase") declare what state fields it reads, modifies, branches on, and guarantees. The runtime checks these declarations against actual execution behavior rather than trusting them blindly. When this feature was first introduced, only **1 of 29 phases** had declared a contract.

  Expanding coverage was initially blocked by a circular dependency in the tooling: `watched_fields()` only looked at state fields that registered contracts had already named. As a result, the inspection tool (`write_profile`) reported almost nothing for the 28 uncontracted phases it was supposed to help measure. The tool couldn't discover new fields because it only looked where contracts were already written.

  We fixed this by introducing `discovery_paths` and `tools/observe_phase_writes.py`, which runs each phase against a live `AuraState` and logs everything it modifies. Using these empirical measurements, ten additional contracts were written. Coverage is now **11 of 29 phases**, and the number of monitored state fields has grown from 11 to 30.

  **What would change this status**: Writing contracts for the remaining 18 phases based on measured execution data rather than code inspection, and marking more of them as `thresholds_exhaustive` (a contract that declares its branches without specifying the exact numbers deciding them leaves the criteria incomplete).

## 17. Exact IIT 4.0 Integrated Information

* **Status**: `not proven`
* **What the code does**: The module `core/consciousness/phi_core.py` calculates a mathematical approximation (spectral approximation) of integrated information ($\Phi$, or "phi") across a 16-node network, and uses an exhaustive search over an 8-node emotional subset as a baseline check. The code comments note that the 16-node calculation is only an approximation, not a consciousness meter. Testing all 32,767 ways to divide 16 nodes ($2^{16} = 65,536$ states) is too computationally expensive to run completely.
* **Why the approximation is not the theory**: Integrated Information Theory (IIT 4.0) defines $\Phi$ using an exhaustive search for the weakest informational link (the minimum-information partition) and relies on active interventions (the "do-operator" testing cause and effect). Our code does neither. Its transition probability matrix is built by observing state changes over time, meaning it measures statistical correlation rather than true causal power.
* **What would change this status**: Calculating an exact minimum-information partition (MIP) across all 32,767 divisions, and building the transition probability matrix (TPM) through controlled interventions rather than passive observation. The first challenge requires significant computing power; the second is an experimental design challenge, because actively intervening in a running system changes the very behavior being measured.

## 18. Reproducible Concept-Activation Vectors

* **Status**: `not proven`
* **What the code does**: The module `core/consciousness/caa/production_caa.py` extracts concept-activation vectors (internal model direction vectors representing specific concepts) from the resident language model at runtime. However, our published evidence bundle only includes test results as JSON files, without the raw vector files (`.npy`). Anyone reading the bundle cannot re-run the calculations independently because reproduction depends on local cache files that only exist on the original computer.
* **What would change this status**: Committing the vetted vector files as repository artifacts along with the cryptographic hash of the exact model version used to produce them, so anyone can reproduce the numbers independently.

## 19. A Closed Recurrence-Training Loop

* **Status**: `not proven`
* **What the code does**: The code for training recurrence loops exists in `core/learning/recurrent_sft_kernel_probe.py` and campaign scripts under `tools/`, along with pre-registered test protocols and pilot contracts. What is missing is a complete run that closes the entire loop: training a model, promoting it, evaluating it on unseen test tasks, and beating the baseline target that was registered before the test.
* **What would change this status**: A complete standalone test run producing a model checkpoint, an automated run receipt, and an improvement on unseen test tasks that holds up when the test set is swapped. Writing a pre-registered test plan establishes the standard for an experiment, but it is not the experiment itself.

## 20. A Provably Contractive Substrate

* **Status**: `not proven`
* **What the code does**: The module `core/consciousness/timescale_stability.py` checks system stability across multiple timescales by computing a stability matrix (Jacobian), calculating the maximum Lyapunov exponent (measuring whether fluctuations grow or shrink), and verifying whether a test function $V(x)$ is a valid Lyapunov stability function. In addition, `core/phenomenal_substrate/maths.py` keeps internal numbers bounded within safe limits and replaces invalid numbers (`NaN`) with neutral values instead of letting errors cascade.
* **What is missing**: Keeping numbers within bounds is not the same as proving stability. Nothing in the code mathematically proves that state updates are contractive (guaranteed to settle into a stable point rather than oscillating indefinitely in an endless loop). The step size used for updates was chosen by the programmer, not derived from a mathematical proof of stability.
* **What would change this status**: A formal mathematical stability proof (a Lyapunov function) for the core state updates themselves—rather than just the timescale interactions—showing that the step size guarantees stability across all possible states.

## 21. A Broad Reasoning Gain From Kept Procedures

* **Status**: `shown on the domains tested` (G09 closed 17 October 2026)
* **What the code does**: `core/learning/procedures_from_solved_examples.py` has her model write a small program for a kind of problem from solved examples, and keeps it only if it agrees with every known answer of the kind, half of them unseen while it was written. Her chat consults kept procedures before she drafts (`core/brain/cognitive_engine_quick_reply.py`).
* **What was shown**: attempt 6, preregistered, beat her own answers on held-out kinds in math and planning with no domain harmed, as registered and once every changed row was read: 29 gained, 0 lost, *p* 3.7e-9 ([result](docs/evidence/G09_PROCEDURES_RESULT_6_2026-10-17.md)).
* **What is still missing**: The gain is narrow. In math it is almost all problems too long for her thinking budget (large primes, factorizations); on 20 of 27 math kinds her own answers were already right. Typed in plain words, a prime-factor question is declined live because a neighbouring kind keeps no procedure. Transfer to competition mathematics is not shown, and open-ended reasoning is not measured.
* **What would change this status**: widening it needs the same test on domains beyond math and planning, and on requests in a person's own words rather than a benchmark's.

## 22. Learning From Conversation That Improves Ordinary Use

* **Status**: `not proven` (L09 open)
* **What the code does**: `core/learning/examples_she_meets.py` keeps an answer a person states in chat as a solved example, holds one that contradicts a kept procedure as disputed, and, while she is idle, groups what she has met and writes procedures for groups large enough to check.
* **What is missing**: One full loop shown live: a kind she gets wrong, taught in chat, a procedure induced while idle, and new instances answered correctly a day later without being told. Teaching ran on 17 October; the check is due on 18 October.
* **What would change this status**: That loop passing live, then procedures on and off compared over realistic tasks.
