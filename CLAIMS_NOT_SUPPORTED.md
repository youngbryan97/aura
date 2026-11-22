# Open Aura claims

These are questions the current experiments leave open. Each section points
to the evidence the project has and the measurement still needed.

## 1. Subjective consciousness and qualia

Seed 101 passed the ISC-v7 operational checks, and the bridge returned
`AT_PARITY` under the study's assumptions. The interpretation depends on
those assumptions, the measured substrate, and the controls that remain
open. Whether the system has subjective experience remains unresolved.
[ISC results](docs/ISC_STATUS.md)

## 2. Artificial general intelligence

The DNU comparison used a smaller output budget and different tools for the
baselines. Its retraction records which conclusions were withdrawn. A broad
capability claim needs fresh tasks and matched comparisons evaluated by
someone outside the project.
[Retraction](artifacts/current/agi_live/RETRACTION.json)

## 3. Metaphysical free will

Aura implements selection and authorization of proposed actions. Its test
results concern those algorithms. [Decision code](core/governance/will.py)

## 4. Recursive self-improvement

The July 1.5B demonstration completed two promotion cycles. The sealed
scores were 0.667 and 0.625. A sustained-gain claim needs a series of
improvements against matched held-out tasks.
[Training run](artifacts/learning_compounding/2026-07-07-1p5b-2cycle/)

## 5. Indefinite autonomy

The July soak verdicts record specific workloads and time windows. Longer
operation needs measured resource use, task completion, and recovery.
An earlier set of certification reports came from a simulator and was
withdrawn from the evidence for longevity.
[Soak standard](docs/LONGEVITY_SOAK_STANDARD.md) ·
[Withdrawal record](docs/DOC_STATUS.md#what-was-corrected-on-2026-08-21)

## 6. Independent validation

The second-Mac substrate repeats were run by the author. One changed its
synergy decision. Independent replication requires another experimenter
using the recorded inputs and protocol.
[Reproduction guide](docs/ISC_REPRODUCTION.md)

## 7. Physical effects outside the host

The code can describe a physical request and check whether a declared
channel can supply the required evidence. A physical result needs an
independent measurement against a baseline.
[Physical evidence and open work](docs/REALITY_REACH.md)

## 8. Complete self-knowledge

The faculty model lists the capabilities covered by its probes and exposes
missing measurements. A broader claim needs coverage beyond the declared
probe set. [Faculty model](core/metacognition/faculty_model.py)

## 9. Legal and moral personhood

Legal status and moral standing require their own grounds. The application
operates under the person's authorization and permissions.
[Human oversight policy](HUMAN_OVERRIDE_POLICY.md) · [ISC results](docs/ISC_STATUS.md)

## 10. Process-level governance

The current safeguards cover action routes that reach the application's
gateways. A process-level guarantee would require an enforced separation
of privileges between the application and the process that performs
consequential operations.
[Tool use policy](TOOL_USE_POLICY.md) · [Threat model](docs/THREAT_MODEL.md)

## 11. Within-generation neural feedback

The model can leave state that influences a later request. A claim about
feedback inside one uninterrupted generation needs a test that changes
the state mid-generation and measures its effect on the remaining tokens.
[Model client](core/brain/llm/mlx_client.py)

## 12. Open-ended evolution

The project has experimented with changing and selecting parts of the
architecture under defined scoring rules. A broader claim needs a
prospective test of sustained adaptation across new conditions.
[Architecture governor](docs/AUTONOMOUS_ARCHITECTURE_GOVERNOR.md)

## 13. Continuous model weight learning

Memory updates and model training use separate paths. Background LoRA
training has resource and admission checks. Continuous beneficial
weight adaptation needs evidence from promoted checkpoints over time.
[Training governor](core/adaptation/online_lora_governor.py)

## 14. General visual detail perception

The frame-quality checks can flag image conditions that make a detail
unreliable. Accuracy across low light, occlusion, motion, distance, and
crowds requires labeled examples and both coverage and error rates.
[Frame-quality checks](core/perception/frame_quality.py)

## 15. Biological self-organization

The project uses biological terms as names for software mechanisms.
Read the implemented update rules and their evidence for each one.
[Terminology](docs/TERMINOLOGY.md) · [Architecture](ARCHITECTURE.md)

## 16. Uniform phase contracts

Phase contracts declare reads, writes, and branch conditions where that
information has been checked. Coverage across the whole pipeline needs
an inventory of declarations and observed effects.
[Phase blueprint](core/runtime/pipeline_blueprint.py)

## 17. Exact IIT 4.0 integrated information

Aura's integration estimates use selected variables and declared
approximation or search rules. Exact IIT 4.0 would require the causal
state model and partitions that theory specifies.
[Estimator](core/consciousness/phi_core.py) ·
[Whole-system estimates](docs/WHOLE_SYSTEM_PHI.md)

## 18. Reproduction of steering vectors

The reported steering effects are bound to the model, basis, and test.
Reproduction requires the matching inputs or a documented procedure for
regenerating them.
[Model card](MODEL_CARD.md) · [ISC reproduction](docs/ISC_REPRODUCTION.md)

## 19. Capability gains from recurrence training

The project tested frozen loops and separate learned readers. A gain from
training the model to recur needs a promoted checkpoint and a held-out
comparison that isolates recurrence at matched cost.
[Recurrence experiments](docs/RECURSIVE_LATENT_CORTEX.md)

## 20. A provably contractive substrate

The stability tools check selected dynamics and can test a candidate
Lyapunov function. A global claim needs a proof for the actual update
map over its declared state domain and step size.
[Stability checks](core/consciousness/timescale_stability.py)

## 21. Broad reasoning gain from kept procedures

* **Status**: `CLOSED` under G09’s registered math and planning criterion.

Attempt 6 met its preregistered criterion. The row review found 29 gains
and zero losses (p = 3.7e-9). The test
covered 444 math and planning requests. No competition MATH question
was admitted by the routing rule. Transfer to other tasks remains open.
[Attempt 6 results](docs/evidence/G09_PROCEDURES_RESULT_6_2026-10-17.md)

## 22. Learning From Conversation That Improves Ordinary Use

In the live date trial, Aura used a learned procedure to answer three
new questions 25.6 hours after the user supplied examples. A broader
comparison on October 18 was underpowered. The second attempt gained
three and lost four. The third stopped at its registered baseline check
after all 80 sampled answers were graded correct. The L09 claim about
ordinary use remains open.
[Date trial](docs/evidence/L09_LEARNING_IN_USE_LOOP_2026-10-18.md) ·
[October 18](docs/evidence/L09_ORDINARY_USE_RESULT_2026-10-18.md) ·
[October 19](docs/evidence/L09_ORDINARY_USE_RESULT_2_2026-10-19.md) ·
[October 20](docs/evidence/L09_WHERE_SHE_IS_WRONG_PREREGISTRATION_3_2026-10-20.md)
