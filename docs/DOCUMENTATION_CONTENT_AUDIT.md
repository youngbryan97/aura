# Documentation content audit

Status: Guide · Reconciliation dated 25 October 2026 · Source baseline through 517c4807c

The earlier documentation pass checked targets and added reading explanations.
It did not establish that every statement in every linked page was current.
This pass found substantive contradictions and corrected the affected current
guides. This page records what was checked so that link coverage cannot be
mistaken for factual certification again.

## Whole-document reconciliation in progress

Bryan’s G09 counterexample exposed a real gap in the earlier pass: positive
G09 summaries coexisted with a blanket denial in the README and RLC table.
The current pass checks the content of guides and status pages against their
source owners and dated results. The previous validation record below does
not certify this expanded pass as complete.

Corrections made in this pass include G09’s passed criterion and separate
open claims, the L09 results through 20 October, the implemented soft-cancel
path, bounded IIT-inspired estimators, native versus operational embedding
limits, and installation/certification commands. Dated results remain intact.
Named repeated claims now have negative checks too: a correct summary does
not hide the same page’s conflicting blanket denial.

## Contradictions and stale readings corrected

| Earlier wording | Current explanation | Source or evidence route |
|---|---|---|
| The Cortex cannot process images | Vision uses the configured Cortex role; the artifact and image path still need to support the request | [LLMConfig](../core/config.py), [model card](../MODEL_CARD.md) |
| Every Cortex build is four-bit with the same context window | Quantization and native context belong to the measured artifact; safe serving limits are separate | [artifact profiler](../core/brain/llm/model_artifact_profile.py) |
| Three model roles mean three models running together | Admission and reservations constrain concurrent work; no second full Cortex is authorised | [lane controller](../core/runtime/model_lane_control.py), [hardware guide](../HARDWARE_PROFILES.md) |
| A 32 GB setup supplies full capability | A smaller configuration needs its own model profile and evidence | [model card](../MODEL_CARD.md), [system card](../AI_SYSTEM_CARD.md) |
| The Reflex is necessarily a CPU emergency/draft model compatible with the present Cortex | It is a configured auxiliary role; execution device, draft compatibility and admission need their actual contracts | [roster](MODEL_ROSTER.md) |
| Model training contained no personal data | This card cannot certify all foundation/persona data without their provenance | [model card](../MODEL_CARD.md) |
| Parakeet v3 is English-only or a short timing sample establishes general accuracy | NVIDIA describes multilingual support; the local timings retain their short warm-sample scope | [model card](../MODEL_CARD.md), [NVIDIA card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| The old encoder read 256 words | Its input limit used word-piece tokens; the 77% truncation figure belongs to the measured example | [model card](../MODEL_CARD.md), [original encoder card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| A 32K native embedding window makes a whole-page GPU encode safe | The source declares smaller operational views and a padded-work budget, with coverage of long text | [embedding source](../core/memory/embedding_model.py), [roster](MODEL_ROSTER.md) |
| Changing another shell's environment changes an already-running Aura process | These readers read the Aura process environment; external shell changes are not an in-process settings update | [context assembler](../core/brain/llm/context_assembler.py), [continuity guide](COHERENCE_LEVERS.md) |
| Cloud-model fallback can be opted into | Current model routing is local; remote-only model requests are refused. Authorised browser/tool traffic can still leave the host | [request contract](../core/brain/request_contract.py), [data card](../DATA_CARD.md) |
| Six test chunks are a safe general recommendation, or smoke always takes under ten seconds | Chunk count is a memory budget; duration is measured per host/revision | [TESTING](../TESTING.md), [reviewer packet](REVIEWER_PACKET.md) |
| A named consciousness/personhood unit suite means the current ISC battery passed | Unit contracts, historical catalogue results and the current ISC experiment have different scopes | [TESTING](../TESTING.md), [ISC status](ISC_STATUS.md) |
| Every natural-language belief becomes formal logic and a checked path is error-free | The checker supports a bounded representation and can report no error found within that scope | [memory card](../MEMORY_CARD.md), [reasoning guide](REASONING_ENGINES.md) |
| Gaussian whole-system phi has a generally exact Queyranne minimum | Exactness is certified at the declared exhaustive-search limit; the general theorem requires submodularity | [estimator](../core/consciousness/integrated_information.py), [terminology](TERMINOLOGY.md) |
| Temperature above two is mathematically uniform | Two is the interface ceiling; the probability distribution still depends on logits | [configuration](../core/config.py), [worked formula](FORMULAS.md#sampling-temperature) |
| Embedded and standalone proof kernels use interchangeable package commands | The embedded package is aura-operational-cognition; the older standalone package keeps its own names | [embedded demo](../proof_kernel/README.md), [standalone repo](https://github.com/youngbryan97/aura-consciousness-proof-kernel) |
| Old “live,” “not started” and “established” labels describe today's whole project | The language, capability and phenomenology pages now identify the particular earlier programme/run being discussed | [language programme](LANGUAGE_SUBSTRATE_AND_GENERALITY.md), [phenomenology](phenomenology/README.md), [capability matrix](OMEGA_CAPABILITY_MATRIX.md) |
| The consequence-bus repair alone accounts for all later isolation work | The degradation tracker also needed restoration; the reported pre-flight is distinct from a replacement campaign | [current ISC status](ISC_STATUS.md), [reproduction](ISC_REPRODUCTION.md) |
| Naming the derived closure margin solved the estimator question | The least-domain rule was withdrawn; the estimator changed and no margin is currently in force | [ISC status](ISC_STATUS.md) |

The former whole-system-phi claim also survived in a function docstring after
the module-level correction. That docstring and the temperature-ceiling
comment were corrected without changing their algorithms.

## Checks that now protect repeated facts

The existing [product-facts gate](../tools/check_product_facts.py) now compares
ten additional model/embedding facts with their source owners: Cortex, Deep,
Vision, Brainstem and Reflex defaults, embedding model and dimensions, native window,
operational view and padded-work budget. It checks the named copies in the
model card and installation guide and reports a mismatch or removed statement.
It also checks the two current closure-margin settings against the status
page. The gate is now part of `make doc-drift`.

[The configuration](../config/product_facts.json) lists exactly which
statements are checked. It does not parse every sentence or certify every
capability. Regression tests check that a real budget change is caught,
formatting such as 1_024 versus 1,024 is accepted, removed statements are
reported, and unsupported normalization cannot silently pass.

The [reading-note renderer](../tools/render_reader_navigation.py) keeps the
glossary and shared definitions consistent. The
[link audit](../tools/audit_documentation_links.py) checks files/headings and,
when requested, bounded public HTTP reachability. These are different checks. Frozen v6 preregistration references now receive
the same historical protection as v5: the name of a retired test remains part
of its original protocol. The dated settings audit is also indexed as a
historical audit; its older cloud-setting row does not describe current
model-routing permission.

## Generated references

The FMEA, runtime contract and architecture map were regenerated through their
source tools. The progress page was also regenerated from the current
registry and evidence ledger. Its certification figure is evidence credit
under that ledger, not a measurement of intelligence or the percentage of
Aura's functions that work. Missing or stale evidence cannot be promoted into
credit by changing the prose. The regeneration produced 0.00% certified credit,
rather than the earlier 0.33%; inspected rejection examples had stale source
manifests. This is a source-bound certification/backfill result, not a
measurement that Aura has zero capability.

## Definitions and arithmetic

The [glossary](GLOSSARY.md) provides contextual definitions. The
[formula guide](FORMULAS.md) defines symbols, units or scales, examples,
practical interpretation and limits. Its examples are illustrative numbers,
not additional experiment results. Shared words such as phi, free energy,
alpha and subject must retain the definition used by the relevant page.

## Validation record

The [dated validation record](evidence/DOCUMENTATION_CONTENT_AUDIT_2026-10-25.json)
records the 81 focused checks, 164 passing smoke tests with one skip, zero
current document-drift findings, regenerated references and the public URL
results. It also records resource limits and what was not measured. The final
link pass covered 982 Markdown pages: no ordinary relative file/heading target
was missing; 314 public URLs answered and ten remained unverified.

## What remains conditional

Historical records and frozen protocols intentionally preserve earlier states
and methods. A later method is linked as a correction or successor; old
outputs are not silently rescored. Some current guides include earlier
catalogues or measurements, which must retain their explicit run/date scope.

Source inspection establishes a configured route or formula, not that the
current installed process used it successfully. Current activation, live
latency, broad capability, long-run stability and independent replication
require their own receipts or experiments. This documentation pass does not
launch them on the shared host.

Reachable external pages can still disagree, change or require access. A
citation supports the passage and scope actually checked; it does not certify
Aura's adaptation. A universal “no stale or contradictory information anywhere”
claim is not established by these checks. New changes require the same source,
evidence and cross-document review.
