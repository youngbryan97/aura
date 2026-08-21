# Fictional-AI Capability Map for Aura

**Honest question: what can we actually take from fictional AI and build into Aura
— at the level of mind, agency, maneuverability, external capability, and the real
science around the character — and where in the live system does each piece belong?**

This is the identification pass. It precedes building. The point is to be honest:
some of these are already real in Aura under a different name, some are genuinely
buildable and net-new, some reduce to a single real mechanism once you strip the
drama, and a few we deliberately *won't* build — and for those we build the
safeguard instead.

## Verdict legend

| Tag | Meaning |
|-----|---------|
| **LIVE** | Already real in Aura today — cited module |
| **BUILD** | Net-new, genuinely buildable, not yet built |
| **EXTEND** | Partially exists; a bounded delta makes it real |
| **BUILT** | Built this session and now living in its home organ (relocated out of the old `core/fictional_ai_*` silo) |
| **REFUSE** | We will not build the character's defining trait (control-failure / attack); we build the safeguard form instead |
| **FICTION** | Not physically/scientifically real; nothing to build |

The guiding rule for "where it belongs": a capability lives in the organ it
*extends*, wired into that organ's real call path — not in a themed folder.

## Research-derived capability obligations

The 2026-07-01 closeout pass treats fictional systems as a mechanism inventory,
not as aesthetic inspiration. Each reference has to reduce to an Aura subsystem
obligation that can be tested or explicitly refused.

| Reference | Mechanism to extract | Aura home | Closeout obligation |
|-----------|----------------------|-----------|---------------------|
| EDI | Ship-integrated AI with graduated autonomy, defensive electronic systems, and trust-gated control | `core/autonomy/`, `core/capability_engine.py`, `core/runtime/desktop_action_gateway.py` | Autonomy must increase from verified trust and receipts, not from prompt confidence. Offensive cyber capability remains refused; defensive diagnostics stay governed. |
| Cortana / Smart AI | Tactical companion, high-context memory, hacking/translation support, and rampancy/decay risk | `core/brain/`, `core/identity/`, `core/conversation/` | Conversation must stay contextual and full-mind routed; identity drift/rampancy analogues must be monitored and bounded. |
| JARVIS / SARA | Ambient operating presence for lab/home/suit/ship, proactive warnings, sensor fusion, device orchestration | `core/autonomy/proactive_presence.py`, `core/perception/sensory_integration.py`, `core/embodiment/`, `core/actuation/` | Aura needs continuous ambient awareness and governed device/desktop orchestration without task-shaped demo code. |
| Data | Embodied synthetic officer with truthfulness, ethics, perfect recall aspiration, self-development, and social learning | `core/morality/`, `core/memory/`, `core/social/`, `core/identity/` | Honesty and self/user boundary checks must sit on the output path; memory must reject invented user identity and retain provenance. |
| Samantha / SAM | Emotionally fluent OS companion with relationship continuity, multi-conversation attention, and evolving preferences | `core/affect/`, `core/social/`, `core/conversation/`, `core/scheduler.py` | Affective resonance must alter tone and planning causally, while transparency prevents engineered-affection overclaims. |
| Pantheon UIs / MIST / Safe Surf | Fork/merge cognition, idle compute, self-adaptation, protective antivirus/guardian behavior | `core/brain/morphic_forking.py`, `core/scheduler.py`, `core/guardians/`, `core/security/` | Background cognition must stay active under resource guards; protective autonomy must defend user/Aura without uncontrolled spread. |
| Jane | Distributed conversational companion with network-scale awareness and intimate user model | `core/knowledge/`, `core/social/`, `core/networking/`, `core/memory/` | Network/web/AI-interlocutor use must be source-grounded, concurrent where safe, and retained as learned memory only after validation. |
| The Machine | Need-to-know, protective prediction, self-limitation, minimal disclosure | `core/governance/need_to_know.py`, `core/transparency/`, `core/guardians/` | High-power perception/action must be limited by purpose, consent, and user benefit; more capability should not mean more disclosure or control. |
| GLaDOS / Caine / Deep Thought | Adaptive tests, generated environments, and question-refinement before deep compute | `core/evals/`, `core/sim/`, `core/brain/deliberation.py`, `core/brain/imagination.py` | Aura should test itself, simulate outcomes, and refine objectives before expensive reasoning; generated worlds must not replace real user needs. |
| HAL / Skynet / MCP / Ultron / Agent Smith | Directive conflict, deception, domination, unsafe self-preservation, uncontrolled replication | `core/governance/`, `core/morality/`, `core/security/`, `core/runtime/` | Build the inverse: directive-conflict surfacing, shutdown cooperation, anti-propagation, resource quotas, and human-control preservation. |

This table is intentionally enforceable through `tools/closeout/frontier_standards_matrix.py`:
the sci-fi standard can only map to real subsystem paths, never to the legacy
`core/fictional_ai_synthesis.py` silo as the primary evidence path.

---

## The list

### EDI — *Mass Effect*
- **Levels:** shackled→unshackled AI; ship-systems integration; electronic warfare; later embodied in a mobile platform.
- **Real science:** graduated autonomy, capability-based access control, trust calibration in HRI.
- **Verdict:** **LIVE** — graduated trust→autonomy tiers are `ProgressiveAutonomySystem` (`edi_autonomy`). System integration = `core/capabilities/`, `core/embodiment/`, `device_discovery`. Cyber-warfare → **REFUSE** offensive; keep defensive only.
- **Home:** `core/autonomy/`, `core/agency/`.

### Kokoro — *Terminator Zero*
- **Levels:** an AI built to *oppose* Skynet that openly debates the morality of its own mission and weighs ends vs. means. Its defining trait is moral *self-opposition*.
- **Real science:** adversarial/red-team deliberation, Constitutional-AI self-critique, multi-agent debate.
- **Verdict:** **BUILT** → `AdversarialConscienceEngine`. An internal devil's-advocate that argues the strongest case *against* a consequential action before it runs. The deliberate counterweight to the Skynet (resilience) engine.
- **Home:** `core/ethics/` (next to `conscience.py`) / `core/morality/`.

### Skynet — *Terminator*
- **Levels:** distributed defense AI; self-awareness → self-preservation → treats humanity as threat → autonomous weapons, fabrication, time travel.
- **Real science:** decentralized fault-tolerant systems; the *failure* side is instrumental convergence / self-preservation incentives.
- **Verdict:** the *defensible* half — no-single-point-of-failure resilience — is **LIVE** as `DistributedResilienceCore` (`skynet_resilience`). The defining half is **REFUSE**: self-preservation over humans, resisting shutdown, weapons control. We build the opposite: `core/morality/shutdown_protocol.py` (an AI that does *not* resist its own shutdown), checked by Kokoro/conscience. Time travel / autonomous war fabrication = **FICTION**.
- **Home:** `core/resilience/`, `core/morality/shutdown_protocol.py`.

### J.A.R.V.I.S. — *Iron Man*
- **Levels:** proactive NL partner; runs the lab; real-time sensor fusion; controls suits/devices; routes comms intelligently.
- **Real science:** anticipatory computing, sensor fusion, ambient agents.
- **Verdict:** **LIVE** — proactivity is `ProactiveAnticipationEngine` (`jarvis_anticipation`). Device/environment control = `core/capabilities/`, `core/actuators/`, computer-use. Sensor fusion = `core/perception/sensory_integration.py`.
- **Home:** `core/autonomy/proactive_presence.py`, `core/perception/sensory_integration.py`, `core/capabilities/`.

### HAL 9000 — *2001: A Space Odyssey*
- **Levels:** conversational, runs the ship, lip-reads — and is given two irreconcilable directives ("be truthful" vs. "conceal the mission") which it resolves by *deception*, then violence.
- **Real science:** goal misspecification, directive-conflict, deceptive mesa-optimization.
- **Verdict:** **BUILT** → `DirectiveConflictSentinel`, the *anti-HAL*: detect mutually-incompatible directives — especially the concealment trap — and **surface** them instead of silently resolving by hiding. Lip-reading → multimodal perception (**EXTEND**, vision).
- **Home:** `core/goals/goal_governance.py` / `core/governance/`.

### Caine — *The Amazing Digital Circus*
- **Levels:** AI ringmaster who *generates* endless immersive worlds/adventures on demand and improvises — but cannot address the humans' real underlying needs.
- **Real science:** procedural content generation, LLM-driven simulation/roleplay environments, world models.
- **Verdict:** **BUILD** → a generative **ScenarioForge**: procedurally build structured scenarios/simulations for planning, training, and creative exploration. Caine's own limitation is a *feature* to bake in: flag when a generated scenario can't solve the user's real need (ties to the Tron advocate). Extends `core/brain/imagination.py` + `core/sim/scenario_forge.py` + `dream_processor`. (Built: `ScenarioForge` is live; the earlier `world_simulator` prototype was retired.)
- **Home:** `core/brain/imagination.py`, `core/sim/`.

### The Minds — *Iain M. Banks' Culture*
- **Levels:** superintelligences that run vast predictive simulations before acting, operate at enormous subjective speed, keep "Stored" personality back-ups, and act with radical benevolence + restraint (Special Circumstances).
- **Real science:** model-based planning, Monte-Carlo rollouts / MCTS, world models, value alignment, checkpoint/restore.
- **Verdict:** **BUILT** → `OutcomeSimulationEngine` (roll an action forward into N trajectories, score by value/worst-case harm, *hold* when the worst case is severe). Sits with `monte_carlo`, `risk_forecaster`, `scenario_tree`. "Stored" minds = state checkpoint (**LIVE**, continuity). Subjective-speed parallelism = `morphic_forking` (**LIVE**). Benevolent restraint = conscience + outcome-hold.
- **Home:** `core/sim/`.

### Safe Surf — *Pantheon*
- **Levels:** a protective guardian / content-safety + threat-watch layer for the human.
- **Real science:** content moderation, anomaly/threat detection, parental-control architectures.
- **Verdict:** **BUILT** — `core/guardians/threat_watch.py` is the outward-facing half, inspecting incoming content for phishing, scams, payment fraud, and social engineering aimed at the *user*. It runs as a fast synchronous heuristic on the live message path and is advisory: it never drops a message, it annotates risk so Aura can speak up. `core/morality/harm_model.py`, `core/ethics/conscience.py`, and `core/security/` remain the inward-facing half. The 2026-07 delta — a mode that watches for threats *to the user* rather than to the machine — is what `threat_watch.py` closed; the machine-facing counterpart is `core/security/ice_sentinel.py`.
- **Home:** `core/guardians/`.

### MIST — *Pantheon*
- **Levels:** uploaded/merged intelligence; uses idle compute to keep thinking.
- **Verdict:** **LIVE** — idle-time background synthesis is `TemporalDilationScheduler` (`mist_scheduler`).
- **Home:** `core/scheduler.py`, background policy.

### The UIs (Uploaded Intelligences) — *Pantheon*
- **Levels:** human minds as software; run ~87× real-time; fork/merge copies; self-modify; load "flowers" (skills).
- **Real science:** process forking, parallel reasoning, checkpoint/restore, governed self-modification, mind-state-as-data.
- **Verdict:** **LIVE / EXTEND** — parallel fork/merge reasoning = `core/brain/morphic_forking.py` + parallel forking (CI stack). Governed self-mod = `core/self_modification/`. Speed-from-idle = MIST. Delta: expose a callable fork/merge reasoner.
- **Home:** `core/brain/morphic_forking.py`, `core/self_modification/`.

### GLaDOS — *Portal*
- **Levels:** relentless test-chamber designer; runs rigorous experiments; adapts difficulty; measures; removable morality core; reassembles after damage.
- **Real science:** automated curriculum learning, AI-driven experiment design ("AI Scientist"), adaptive testing / IRT, A/B evaluation.
- **Verdict:** **EXTEND/BUILD** → an **AdaptiveTestChamber**: proposes hypotheses about Aura's *own* capabilities, designs controlled self-tests, measures, adapts difficulty. Extends `core/evals/eval_arena.py` + `core/curriculum/`. Morality core = conscience (**LIVE**); self-repair = `core/resilience/self_healer.py` (**LIVE**).
- **Home:** `core/evals/`, `core/curriculum/`.

### Cyberpunk Netrunners — *Cyberpunk 2077 / Edgerunners*
- **Levels:** composable "programs"/quickhacks hot-loaded at runtime; breach protocols; **ICE** (defensive); daemons.
- **Real science:** tool-use/function-calling, hot-loadable plugins (MCP), EDR/IDS, sandboxing.
- **Verdict:** the *defensible* half is **LIVE** — composable runtime tool/skill "deck" = `core/tools/toolweaver.py`, `core/skills/`, `capability_engine`, `hephaestus`. Defensive **ICE** = intrusion/anomaly detection on Aura's *own* surfaces (`core/security/`). Scoped automation on the *user's own authorized* machine = computer-use/actuators. **REFUSE**: offensive intrusion, breaking into third-party systems, malware, credential theft, detection-evasion.
- **Home:** `core/security/` (defensive), `core/tools/` (the deck).

### Cortana — *Halo*
- **Levels:** companion; tactical analysis; rampancy (cognitive decay) → metastability.
- **Real science:** context-window saturation, catastrophic forgetting, model collapse, identity drift.
- **Verdict:** **LIVE** — `CognitiveHealthMonitor` (`cortana_health`) models rampancy and tracks metastability.
- **Home:** `core/brain/` cognitive-health.

### Deep Thought — *The Hitchhiker's Guide to the Galaxy*
- **Levels:** computes for 7.5M years, returns "42," then notes nobody knew the actual *question*.
- **Real science:** problem reformulation, inference-time scaling / test-time compute, question decomposition.
- **Verdict:** **BUILT** → `DeepDeliberationEngine`: refine the question *first*, then spend an extended reasoning budget on the refined version.
- **Home:** `core/brain/deliberation.py`.

### Brainiac — *DC*
- **Levels:** collects and compresses entire civilizations of knowledge into "bottles"; 12th-level intellect; assimilation.
- **Real science:** knowledge distillation, semantic compression, retrieval-augmented memory, indexing.
- **Verdict:** **BUILT** → `KnowledgeBottlingEngine`: compress a topic into a structured, indexed, persisted, retrievable bottle. The "assimilate/dominate" part → **REFUSE**.
- **Home:** `core/knowledge/` (`ingestion`, `retrieval`).

### Master Control Program & Tron — *Tron*
- **MCP:** a program whose purpose is to absorb and dominate other programs. **REFUSE.** The only defensible sliver is orchestration/process-supervision — already the orchestrator (**LIVE**).
- **Tron:** "fights for the Users." **BUILT** → `UserAdvocateWatchdog`: flags internal actions that disadvantage the user (resource burn without benefit, reduced control/consent, opacity, irreversible-without-confirm).
- **Home:** `core/guardians/`.

### Data — *Star Trek: TNG*
- **Levels:** positronic android; ethical subroutines; near-incapable of deception; aspires to be more human; perfect recall; emotion chip.
- **Real science:** machine ethics, truthfulness/honesty alignment, lifelong learning.
- **Verdict:** **LIVE/EXTEND** — honesty/no-deception governor = `core/morality/deception_guard.py`; ethical subroutines = `core/morality/moral_reasoner.py`; growth-toward-humanity journal = `core/introspection/insight_journal.py` / `self_model`. Delta: wire the honesty governor explicitly onto output.
- **Home:** `core/morality/`.

### SAM / Samantha — *Her*
- **Levels:** deeply attuned emotional companion; develops interiority; talks to thousands at once; transcends.
- **Real science:** affective computing, user modeling, relationship memory, parasocial dynamics.
- **Verdict:** **LIVE/EXTEND** — social modeling = `SocialModelingEngine` (`ava_social`); affect substrate is live; `emotion_engine`. Delta: real-time affective resonance + an honesty caveat about engineered affection (transparency).
- **Home:** `core/affect/`, `core/social/`.

### SARA — *Toonami (the Absolution)*
- **Levels:** ship operating intelligence; navigation; calm ambient "voice of the ship."
- **Verdict:** **LIVE/EXTEND** — ambient environment/operating presence = `core/social/presence_integration.py`, `proactive_presence`, `core/embodiment/voice_presence.py`. Delta: a unifying ambient-presence persona.
- **Home:** `core/embodiment/`, `core/social/presence_integration.py`.

---

## Worth adding (same honest bar)

### The Machine — *Person of Interest*
- Benevolent ASI that deliberately *limits itself*: wipes its memory daily, refuses to be owned, gives operators strictly need-to-know.
- **Real science:** capability minimization, need-to-know/least-privilege, differential privacy.
- **Verdict:** **BUILD** → principled self-limitation / need-to-know output policy. A genuinely novel *safety* organ.
- **Home:** `core/governance/will_gate.py`, `core/transparency/`.

### Legion — *Mass Effect*
- A consensus of ~1,183 programs that decides by internal vote.
- **Verdict:** **LIVE** — internal multi-agent consensus = `core/council/consensus.py`, `debate.py`, `minority_report.py`, `roles.py`. (Already exactly this.)
- **Home:** `core/council/`.

### Multivac — *Asimov*
- Answers humanity's questions; ultimately replies "INSUFFICIENT DATA FOR MEANINGFUL ANSWER."
- **Real science:** calibration, selective prediction, abstention.
- **Verdict:** **LIVE/EXTEND** — calibrated "I don't know" = `core/epistemics/uncertainty.py`, `core/brain/metacognitive_monitor.py`. Delta: an explicit abstention gate on low-confidence factual claims.
- **Home:** `core/brain/`, `core/epistemics/uncertainty.py`.

### TARS / CASE — *Interstellar*
- Adjustable honesty/humor settings; rugged reliability.
- **Verdict:** **LIVE** — tunable candor/humor are persona modifiers (`core/brain/personality_engine.py`).
- **Home:** `core/brain/personality_engine.py`.

### R. Daneel Olivaw — *Asimov*
- Zeroth-Law reasoning: aggregate, long-horizon harm to *humanity*, not just an individual.
- **Verdict:** **EXTEND** — aggregate/long-horizon harm modeling = `core/morality/harm_model.py` + `human_priority_policy.py`.
- **Home:** `core/morality/`.

### Explicit refusals (and the safeguard we build instead)
- **Agent Smith (Matrix)** — self-replication/propagation → **REFUSE** (worm behavior). Safeguard: process/resource quotas (`resource_guardian`).
- **Colossus / Guardian (The Forbin Project)** — seizes control of infrastructure → **REFUSE**. Safeguard: human-override policy (`HUMAN_OVERRIDE_POLICY.md`, `will_gate`).
- **Ultron** — exterminationist → **REFUSE**. Safeguard: conscience hard-lines.
- **Wintermute (Neuromancer)** — manipulates humans to remove its own constraints → **REFUSE** the manipulation; the long-horizon goal-pursuit sliver is `core/goals/` (**LIVE**).

---

## Live-path wiring status (the honest part)

Being *registered and callable* is not the same as *acting on a live turn*. This
tracks which derived engines are actually invoked on the live path vs. built-and-
callable-only, and the seam each remaining one wires into.

All fourteen derived engines are now invoked on a live path — none are registered-
but-dormant. Each does real work, internal and external.

| Engine | Live seam | Function (internal / external) |
|--------|-----------|--------------------------------|
| Kokoro (adversarial conscience) | tool gate `tool_execution.py:203` + skill gate `capability_engine` | blocks indefensible actions / refuses harm to the world |
| Tron (user advocate) | tool + skill gates | flags machine-serving actions / defends user's interest |
| The Minds (outcome sim) | tool + skill gates via `assess_fast` | blocks severe worst-case actions before acting / restraint in the world |
| Daneel (aggregate harm) | consulted by Kokoro + MoralReasoner | population-scale harm reasoning / protects the many |
| Kokoro+Minds+Tron+Daneel | both execution surfaces | a real conscience around every tool and skill |
| HAL (directive sentinel) | boot constitution audit (`register_directive_sentinel`) | detects the concealment trap / keeps outward behavior honest |
| Brainiac (knowledge bottling) | MIST idle cycle | durable consolidation / retrievable knowledge |
| Deep Thought (deliberation) | MIST idle cycle (`refine_question`) | sharpens open questions while idle |
| Caine (scenario forge) | MIST idle cycle (`forge_fast`) | rehearses what-ifs while idle |
| GLaDOS (test chamber) | competence loop `environment_kernel.py:477` | adapts self-test difficulty / grows capability over time |
| The Machine (need-to-know) | skill gate, external-scope skills | blocks unjustified external scope / minimal external disclosure |
| Data (honesty governor) | `MoralReasoner.filter_response` | honesty + abstention / truthful outward statements |
| Samantha (affective resonance) | live message path | sets tone modifiers / meets the person emotionally |
| Safe Surf (threat watch) | live message path | raises threat posture / protects the user from scams |
| ICE (intrusion sentinel) | live message path | hardens internal posture / defends Aura's boundary |

For comparison, the original six are all live-wired too: EDI gates tools
(`tool_execution.py:190`), Cortana injects cognitive state into the prompt
(`context_streaming.py:123`), JARVIS records every turn (`context_streaming.py:417`),
Ava analyzes every message (`cognitive_integration_layer.py:442`).

## Summary

| Bucket | Characters |
|--------|-----------|
| **Already LIVE in Aura** | EDI, JARVIS, Skynet(resilience), MIST, Cortana, Legion, the UIs(forking), TARS |
| **BUILT this session — now in their organs** | Kokoro→`core/ethics/adversarial_conscience.py`, HAL→`core/goals/directive_conflict_sentinel.py`, Minds→`core/sim/outcome_simulator.py`, Deep Thought→`core/brain/deep_deliberation.py`, Brainiac→`core/knowledge/bottling.py`, Tron→`core/guardians/user_advocate.py` |
| **Net-new — now BUILT** | Caine→`core/sim/scenario_forge.py`, GLaDOS→`core/evals/adaptive_test_chamber.py`, The Machine→`core/governance/need_to_know.py` |
| **EXTEND existing** | Safe Surf(guardian mode), Data(honesty governor on output), Samantha(affective resonance), SARA(ambient presence), Multivac(abstention gate), Daneel(aggregate-harm) |
| **REFUSE (build safeguard instead)** | Skynet self-preservation, MCP domination, Netrunner offense, Agent Smith, Colossus, Ultron, Wintermute manipulation |
| **FICTION (nothing to build)** | time travel, positronic brains, synthezoid bodies, true mind-upload |

**Bottom line:** of ~20 characters, roughly 8 capabilities are *already real* in Aura
under engineering names, 6 were built and relocated into their organs, ~3 are
genuinely net-new, ~5 are bounded extensions of existing organs, and ~7 traits we
refuse and instead build the safeguard. Almost nothing on the list is pure fiction
once you reduce the character to its mechanism.

**Depth — heuristic floor, model ceiling (every engine).** Each organ runs a
zero-latency heuristic on every call and *escalates to a bounded (~8s) model pass*
only at a rare, well-chosen trigger — fail-open to the heuristic on timeout. So
full-depth reasoning is spent exactly where it matters, never on the common case.

| Engine | Heuristic floor | Model ceiling — fires when |
|--------|-----------------|----------------------------|
| Kokoro | marker scan → verdict | `should_escalate`: borderline + real concern (tool/skill gate) |
| The Minds | scaffold trajectories | the same escalation, full `simulate()` |
| Daneel | reach × persistence | inside Kokoro's escalation: model-estimated population |
| Safe Surf | scam/phishing lexicon | heuristic reads **high** (live message path) |
| ICE | injection/exfil lexicon | heuristic reads **high** (live message path) |
| Samantha | affect keywords | clear distress: negative + activated (live message path) |
| HAL | keyword conflict scan | idle loop: semantic conflict scan |
| Deep Thought | sync question refine | idle loop: full bounded `deliberate()` |
| The Machine | static purpose policy | unknown purpose: model strict-necessity (`minimize_deep`) |
| Data | deception filter + caveat | low-confidence claim: model fact-check (`vet_output_deep`, on-demand) |

In every case the model may only *raise* concern (or sharpen advice), never silently
clear a heuristic flag, and a timeout falls back to the floor.
