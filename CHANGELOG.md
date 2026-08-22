# Changelog

Aura is calendar-versioned. The authoritative version string is `version` in
`pyproject.toml`.

This changelog starts at 2026-08-01 and is written forward from there.
Everything before it is summarised below from the commit history rather than
reconstructed release by release — 4,080 commits in five months, most of them
landing without a release boundary. The git log is the real record; this is
the shape of it.

## Format

Entries group by month, newest first. Each names what changed and why it
mattered. A change with no user-visible or operator-visible consequence
belongs in the commit log, not here.

Statuses used: **Added**, **Changed**, **Fixed**, **Removed**, and
**Not claimed** — the last for capability that shipped as infrastructure
without evidence to back a claim yet.

---

## 2026-08

### Added
- **Ownership, review and a branch policy** (`.github/CODEOWNERS`,
  `config/branch_protection_policy.json`, `tools/check_review_policy.py`,
  `tools/check_branch_protection.py`) — `main` reported `protected: false`
  with no required status checks, so sixteen CI jobs were advisory and two of
  them were red on `main` while configured to run. The policy is declared and
  applied with one command; the offline gate holds the required list against
  the workflows so a new job cannot be green and required by nothing.
- **A threat model with the attacks in it** (`docs/THREAT_MODEL.md`,
  `SECURITY.md`, `tests/security/`, `tools/check_threat_model.py`) — fourteen
  classes of attack, each with the control that answers it and a test that
  runs it from the attacker's side. The document states, and the gate keeps it
  stating, that no independent security engineer has attacked this system.
- **A dependency contract** (`config/dependency_contract.json`,
  `tools/check_dependency_contract.py`, `tools/derive_lockfile.py`) — one
  declaration of what every build installs, and the gate that caught all five
  contradictions between the declaration and the builds.
- **A release lifecycle** (`docs/RELEASE_LIFECYCLE.md`,
  `tools/check_release_ready.py`) — channels, supported platforms, upgrade
  compatibility and what a tag must produce. No release has been published;
  the document says so.
- **Canonical product facts** (`config/product_facts.json`,
  `tools/check_product_facts.py`) — the Python version, the license, the port
  and the package version have one owner each, and every other file that
  states them is checked against it.
- **A seam extractor that proves its own move** (`tools/extract_seam.py`) —
  refuses a cut that is not behaviour-preserving, and checks the helper body
  against the original token for token before writing.
- **A bounded resident-32B reasoning gain, replicated and lesion-dependent**
  (`core/brain/llm/semantic_neural_serving.py`) — on a frozen four-domain
  cohort of 60 typed tasks, the trained recurrent controller answered 60/60
  exactly against 16/60 for ordinary decode, with a matched wire base at 7 and
  a coefficient lesion at 5. No family regressed. One gained nothing because
  ordinary decode was already at ceiling on it (15/15) and the controller
  preserved all fifteen; the other three supplied the 44 conversions. Paired
  one-sided exact *p* = 5.7 × 10⁻¹⁴. Adjudicated `BOUNDED_WOW_SIGNAL`. It runs
  in the live path — 120/120 exact and 120/120 lesion-disrupted at a median
  5.325 ms — and CP824 removed the `desktop_required` coupling that had kept an
  active certified package unreachable from ordinary chat turns. It still
  cannot answer ordinary chat: `ordinary_chat_authorized` stays pinned
  `False`, and admission runs an answer-blind parser over the task grammar.
  Two entries left the programme's "not established" list as a result, and
  [docs/INTRINSIC_RECURRENCE.md](docs/INTRINSIC_RECURRENCE.md) names which.
- **Meaning decided from examples rather than word lists**
  (`core/language/learned_matcher.py`) — every matcher in the runtime was a
  regex with a list of words in it, and every one had been wrong the same way:
  a phrasing nobody thought of. "I saved it as sitting_timer.html" missed an
  action-claim rule by four characters. The labels already existed and nothing
  read them — each Observable declares examples and counter-examples, and the
  registry test fails a matcher that gets its own examples wrong. The boundary
  is measured by leave-one-out rather than chosen, the surface abstains
  between the worst positive and the best negative, and it needs an embedding
  rather than a generation, so there is nothing to steer. The desktop-routing
  decision measures AUROC 0.979 on held-out paraphrases.
- **A closed observe-decide-act loop for the browser**
  (`core/skills/sovereign_browser.py`, `pursue` mode) — a scripted action list
  presumes every selector is known before the first click, which fails for any
  flow whose next screen depends on the last answer. The loop carries a
  standing understanding across rounds rather than re-deciding from nothing,
  batches genuinely independent decisions, and bounds itself on progress
  rather than a clock — a working pursuit had been cancelled at 181 seconds
  mid-form and the person told the page had not responded. It takes no
  authority of its own; execution goes through the same lease, receipt, and
  effect verification a scripted interaction uses. New page:
  [docs/BROWSER_PURSUIT.md](docs/BROWSER_PURSUIT.md).
- **One registry for readings** (`core/brain/observable_registry.py`) — 26
  observables, each declaring how it is recognised, how it is read, and the
  examples that hold the recogniser honest. It was never about files: the
  clipboard, the screen, the clock, her own queued and completed work, her
  transcript, her lifetime, and what she has actually validated are all the
  same shape of question, and each was previously answered from weights or not
  at all.
- **She speaks while she thinks** (`core/conversation/reply_stream.py`) — the
  governed pipeline already produced its reply incrementally; nothing could
  read *its own turn's* chunks, because the telemetry topic they ride on is
  global. A channel bound to a turn's async context fixes that, and the voice
  lane now releases clauses as they are produced, each governed before it is
  synthesised. Time-to-first-audio no longer scales with total reply length,
  which is what the 45-word spoken cap existed to hide — and why spoken
  answers were shallower than the same question typed.
- **Ambient listening** (`core/voice/duplex/addressivity.py`) — you can talk
  without pressing anything. The wake word is demoted from the only gate to
  the strongest of several signals: name, whether a conversation was already
  open, phrasing, loudness against this speaker's own baseline, competing
  voices. A ladder rather than a score, because weights nobody measured are
  opinions with decimal points and a score cannot be argued with after a
  mistake. Every verdict carries its reasons; it fails closed.
- **Acoustic end-of-turn** (`core/voice/duplex/acoustic_endpoint.py`) — the
  pitch contour over the final voiced stretch, fitted with the F0 estimator
  already running for paralinguistics. It can only ever *extend* the wait, so
  a wrong reading costs a beat and can never cut somebody off. Addresses the
  single most common complaint about shipped voice assistants: being
  interrupted when you pause to think.
- **Media in the chat** (`core/media/`, `interface/routes/media.py`) — "play
  X" resolves against what is on this machine and plays in the conversation
  rather than handing off to another app. Range requests are honoured, so
  seeking works and a large file does not buffer entirely before it starts.
  The index is the allowlist: playback resolves an opaque id through it, so
  there is no path in the URL to sanitise.
- **Sight** (`core/senses/sight.py`, `core/senses/sight_intent.py`) — "how
  many fingers am I holding up" captures a frame *now*, at a resolution a
  model can read, and answers from it. Distinct from the presence lane, whose
  320×240 thumbnail is right for knowing somebody is there and useless for
  counting anything. "Turn on the camera" writes the same setting the UI's
  own switch writes, so the control, the privacy record and the device move
  together. Measured: worker up in 5.0 s, ~0.7 s per look, 4/4 on stylised
  hands — see the limits recorded in **Not claimed**.
- **Failures she can explain** (`core/conversation/failure_context.py`) — a
  failed capability records what it tried, what stopped it, how it knows, and
  what is still possible. Her turn reads those and words it herself. The
  runtime supplies facts; she supplies the sentence.
- **Reality Reach** (`core/reality_reach/`) — a physical request compiles to a
  typed contract with declared channels, and reachability is proven before
  anything executes. Unmeetable requests return a typed limitation
  certificate rather than an optimistic simulation. Dispatch, execution and
  `EFFECT_VERIFIED` are separate states, so transport success can never be
  recorded as a verified effect. Registered hardware routes through
  `HardwareManager` and `BaseHardwareDevice.safe_execute`.
- **Kernel-boundary sandboxing for model-written Python** (`core/sandbox/`) —
  `sandbox-exec` on macOS, `bwrap` on Linux, network denied. When no boundary
  is available it **refuses to run the code** rather than running it
  unconfined and reporting a normal result.
- One shared bounded numeric guard for values accepted from outside the
  process, and one structural redaction primitive.

### Changed
- **Shell execution is isolated by default** (`skills/shell.py`,
  `security/sandbox.py`) — `sandbox` defaulted to `False`, so the ordinary
  path ran the caller's program on the host with the whole filesystem readable
  and the network open. `SecurityLevel.CONFINED` keeps the seatbelt profile,
  the rlimits, the stripped environment and explicit read and write scope
  without a binary allowlist overruling the Will's decision. Host execution is
  a separately authorized escape hatch.
- **The layering rule covers the tree** (155 `DEPS` files, up from seven;
  `tools/generate_deps.py`) — every core package and every tree that imports
  core now has include rules, generated from the import graph, so a new
  cross-package dependency is an edit to a DEPS file.
- **One lint standard** (`config/ruff_strict_files.txt`) — the configured ruff
  rule set applied to seventeen hand-picked files while 4,072 of the
  repository's files passed it untouched. Black, isort and flake8 are gone
  from pre-commit; ruff and mypy remain.
- **Typing has a floor and a direction** (`tools/check_typed_surface.py`,
  `tools/typecheck_changed.py`) — 1,703 of 2,799 production modules annotate
  every parameter and return, and that number may not fall. A file a branch
  touches must pass strict mypy and is adopted into the allowlist when it does.

### Fixed
- **Path containment in the shell skill** — `rm` compared paths with
  `startswith`, which admits `/allowed/project-evil` for a root of
  `/allowed/project`. The workspace jail had the same defect in its
  denied-path list.
- **Three ungoverned execution surfaces** (`skills/shell.py`,
  `core/cybernetics/omni_tool.py`, `core/body/terminal_motor.py`) — each ran a
  caller-supplied command without asking the Will. The guard written to catch
  exactly this scanned only `core/` and only one spawn function.
- **The migration ledger checked nothing** (`core/db/migrations.py`) — the
  checksum was written on every apply and read by nothing, so a migration
  edited after it ran was skipped because the version number matched. An
  interrupted migration now leaves `started` in the ledger instead of nothing.
- **The release lane could not install its dependencies and continued anyway**
  — `pip install -r requirements/runtime.txt || pip install -r
  requirements.txt || true`, where the first file has never existed, before
  signing and notarizing the result.
- **The image claimed a licence the repository does not grant** — `MIT` in the
  Docker label against a LICENSE reserving all rights.
- **The method-size ratchet could launder its own debt** — `--write-baseline`
  recorded whatever it measured, including growth.
- **`make typecheck` was red** — three `no-any-return` errors from Any leaking
  through `--follow-imports=skip` out of callees that do declare their types.
- **Every vision call in the repository was failing, and the loudest way it
  failed was silently.** The worker's message carried no image part and the
  chat template was never told there was an image, so a call that succeeded
  produced a prompt with no image token — the model answered from the
  question alone, fluently and with complete confidence, and nothing in the
  output distinguished that from working sight. Two fatal defects sat
  underneath: the base64 payload was passed where a path was expected, and
  the resulting exception was outside the handler, so one bad call killed the
  worker rather than the request.
- `MLXVisionClient.stop()` called `join()` on a process that had never
  started, which asserts rather than returning — so any failure during spawn
  turned every later stop into "can only join a started process" and buried
  the reason the worker never came up.
- Every vision call site constructed its own client, and each construction
  spawns a subprocess holding 1.2 GB of weights. `get_vision_client()` is now
  the shared accessor.
- `torchvision` was missing and undeclared. `transformers` 5.x builds its
  image processors on it, so torch alone loads text models fine and cannot
  construct a vision processor at all.
- The clause-streaming carve-out had a complete test suite and **zero
  production callers** — the voice lane still blocked on the finished string
  and then chunked it, so the latency it was written to remove was entirely
  intact while the module reported itself present.
- The addressivity gate called a property as a method, so from the fourth
  utterance of every session onward it raised, failed open, and answered
  without checking — working perfectly for three turns and then silently off.
- Spoken and typed turns were different conversations: the voice socket
  passed its per-connection uuid downstream as the conversation identity, so
  switching from talking to typing lost the thread, and a socket that merely
  reconnected started a third one.
- A streamed reply was reconciled against the text *released to the
  synthesiser* rather than the text actually delivered, so a correction could
  claim the user heard words that never reached the speakers.
- Untrusted code inherited the parent's entire environment; the sandbox
  boundary binary is now resolved absolutely.
- Two benchmark harnesses executed Aura-written modules inside the privileged
  runner process.
- Cloud deployment accepted any host key at the target address.
- Importing the cloud launcher provisioned twenty regions as a side effect.
- A bench gate a model could pass while wrong on every case.

### Not claimed
No Aura physical actuation, physical effect, weakpoint, or ambient-law result
is claimed. The RR-10 acceptance battery is open and the P0–P6 evidence
promotion state machine is not implemented. See
[docs/REALITY_REACH.md](docs/REALITY_REACH.md).

**No voice latency number is claimed.** Clause streaming removes a structural
dependency — time-to-first-audio no longer scales with total reply length —
and that dependency's removal is what the tests establish. The end-to-end
figure on the live 32B under load has not been measured, and the previous
figures in `core/voice/duplex/config.py` describe the components (Kokoro's
synthesis rate, Whisper's decode) rather than the whole path.

**No addressivity accuracy is claimed.** The gate's rungs are rules with
stated rationales, tested against transcripts chosen to be plausible rather
than sampled from real use. False-accept and false-reject rates in a real
room, with a television on and a phone call happening, are unmeasured. The
thresholds in `acoustic_endpoint.py` are likewise literature-shaped priors,
not readings taken on this host — what *is* established is the asymmetry that
makes them safe, that a wrong reading can only add patience.

### Documentation
- **A gate for references** (`tools/lint_doc_drift.py`, `make doc-drift`) —
  `make writing` read the prose and `make claim-constants` read the numbers a
  claim cites; nothing read the paths, links, `make` targets, environment
  names, or counts, which is what a reader follows first. Baseline zero, wired
  into `make quality` and the ratchets workflow, with its own 31-case suite
  because a gate that cannot match reports green forever. What it found on the
  first run is recorded in [docs/DOC_STATUS.md](docs/DOC_STATUS.md) under
  2026-08-21 — including a threat-model control for a capability that does not
  exist, nine operator kill switches nothing reads, and a test suite eight
  documents sized identically and wrongly.
Full reconciliation of all 179 living docs against the tree. Corrected a
documented test count that was 3× low, an architecture map ~400 files stale,
four documented environment variables and files that did not exist, and a
supply-chain instruction that pointed at the wrong requirements file. Added
[docs/DOC_STATUS.md](docs/DOC_STATUS.md),
[docs/README.md](docs/README.md), [AGENTS.md](AGENTS.md), and runbooks for
all 19 known failure modes.

**Second reconciliation, 2026-08-13** (1,660 commits later). Resolved 2,317
code-path references across 272 tracked docs against the tree; nine were dead
and are fixed, and the ones deliberately naming absent files are recorded as
such rather than repaired. Corrected the test count (24,931 → **34,382**), the
architecture map (154/2,597 → **153/2,741**), the Brainstem lane
(Qwen2.5-7B → **Qwen3.5-9B**), the ASR engine (Whisper → **Parakeet TDT**), and
the embedding backend (MiniLM → **Qwen3-Embedding-0.6B**).

The Recursive Latent Cortex now has a landing page linked from the top of the
README — [docs/RECURSIVE_LATENT_CORTEX.md](docs/RECURSIVE_LATENT_CORTEX.md),
restructured so the status and claims ladder sit above the spec. Added
[docs/INTRINSIC_RECURRENCE.md](docs/INTRINSIC_RECURRENCE.md) (the training
front, which had existed only as ledger entries),
[docs/COGNITIVE_ARCHITECTURE_ADOPTION.md](docs/COGNITIVE_ARCHITECTURE_ADOPTION.md)
(Soar and ACT-R), [docs/MODEL_ROSTER.md](docs/MODEL_ROSTER.md) (every lane and
the measurement behind it), and `ARCHITECTURE.md` §19.

---

## 2026-07 — 2,041 commits (346 features, 681 fixes)

The heaviest month, and the one where fixes outnumbered features two to one.

- **The endurance ceiling turned out not to be cognition.** The "15-turn
  ceiling" was a prompt cache that was never constructed and then cleared
  every turn, so each turn re-prefilled the whole conversation from token 0.
  Root cause in `artifacts/closeout/endurance_ceiling/ROOT_CAUSE.md`.
- **A standing self-model** — `core/metacognition/faculty_model.py`. Faculties
  declare metrics with units, floors, targets and ceilings; unmeasured reads
  as a blind spot rather than as healthy; priority is headroom weighted by
  how much of the stack a faculty gates.
- **Associative entity memory** — one place where a person, place, thing,
  organization or concept accumulates traits, facts, events and relations,
  plus what it has come to mean to her.
- **Structural screen perception and native OS control** — window ownership,
  geometry and z-order instead of aiming actions at OCR'd pixels.
- **The engineering spine** — taint register, lockdep, PSI, OOM shed ladder,
  telemetry dictionary, invariants in `core/verify/`, the `make layering`
  gate. Seven clean-room adoption waves.
- **Recursive latent cortex and SPARK** — resident 32B recurrent SFT and GRPO
  campaigns, preregistered canaries, holdout discipline.
- Voice, UI legibility, and conversational-organ coverage work.

## 2026-06 — 1,016 commits (106 features, 85 fixes)

Reasoning and evidence discipline. Verifier-gated reasoning with a measured
verifier foundry, the frontier discovery engine's
PROVEN/SUPPORTED/CONJECTURE/REFUTED taxonomy, program-DNA reconstruction,
whole-system φ, the flight recorder, source-body proprioception, and the
Ulysses Covenant.

## 2026-05 — 598 commits

Evidence standards and security posture. The twelve `*_STANDARD.md` bars,
compliance mappings (OWASP, NIST SSDF, MITRE ATLAS), the threat model, the
permission matrix, and the incident runbooks.

## 2026-04 — 327 commits

Production hardening. The governance fence and `make governance-lint`,
capability-token lifecycle, stem-cell reversion, the SLO contract, the
platform posture decisions, and the first runbook set.

## 2026-02 / 2026-03 — 3 commits

Repository initialised 2026-02-23.

---

## Known version drift

`pyproject.toml` reads `2026.4.20` on a calendar-versioned repo that is now
well past April. The version string has not tracked the work. Flagged here
rather than silently bumped, because choosing the next version is a release
decision, not a documentation one.
