# Swarm shards off: no script sent to her model, nothing kept for her, 19 October 2026

## What a shard did

`core/agency/agency_shard_work.py::_shard_wrapper` sent her model a script through
`StructuredLLM(ShardResponse)`: "[SOVEREIGN SWARM SHARD] ... You are an autonomous
cognitive shard ... DO NOT GUESS ... CRITICAL: You MUST respond with a valid JSON
object", with `python_sandbox` and `web_search` described in prose and a JSON protocol
for calling them. When the JSON did not parse, the shard wrote her answer itself:
"I am experiencing cognitive formatting degradation. Taking a defensive posture."

`~/.aura/logs/desktop-launch.log` holds 13 shard monologues, and 5 of them are that
fallback. The others read "As the Opponent, I must..." and "The goal is to argue for
the validity and necessity of 'internal knowledge graph continuity'".

## Who spawned shards, and what read their replies

The spawners read only whether a shard was admitted:

| Spawner | Goal it sent |
|---|---|
| `SovereignSwarm.run_deliberation` (autonomy conductor, boot debate) | "Debate Perspective - Proponent: Argue for the validity and necessity of the concept." and two other roles |
| goal genesis (`agency_core`) | "Initial research and mapping for: {topic}" |
| self-architect refinement | "Analyze the structural integrity and identify technical debt..." |
| self-architect evolution (`SkillEvolution.spawn_evolution_shard`) | "Research and propose an optimization for the '{skill}' skill" |
| social reflection | "Reflect on my evolving bond with {name}" |
| autonomous research | "Analyze the structural integrity and design patterns of {file}" |
| `HealingSwarmService.attempt_repair` | "Identify root cause for {subsystem} failure and suggest/trigger fix." |
| `CriticEngine.spawn_critical_shard` (no caller) | "Critically audit this research insight: ..." |

Every goal is an order. Sending it as evidence would still put an instruction in
front of her model.

A shard's reply went to four places:

- **The dialectical crucible** (`core/adaptation/dialectics.py`). The shard was its
  only caller. By default it wraps the text as "Provisional belief: {reply} This
  remains revisable until future evidence confirms it." and commits that to her belief
  engine. With `AURA_CRUCIBLE_BACKGROUND_LLM` on, it runs three scripted roles
  (ANTAGONIST, DEFENDER, ARBITER) instead.
- **The abstraction engine**, when the reply ran past 80 words or asked for a tool. It
  sends the reply to her model inside a worded request ("The one general rule that made
  this resolution work...") and keeps the answer in `data/first_principles.json`.
- **Her inner insights**, only when the crucible could not be imported.
- **The mycelial pulse**, which carries no content.

None of these needs the reply to be hers. The two that keep it reword it first.

## What her stores held on 19 October

- `~/.aura/data/beliefs/belief_system.json`: 266 of her 288 beliefs have source
  `dialectical_crucible`, created between 26 April and 18 October. 251 of them start
  "Provisional belief:". One is the fabricated line, with evidence count 38.5. Others
  read "The opponent's approach seems aimed at ensuring the reliability of their
  position." and "The `agency_core.py` file appears to be a modular design".
- `~/.aura/data/first_principles.json` (76 entries after the 18 October quarantine)
  still holds shard-shaped principles, such as "When a critical system enters a stale
  state, thorough investigation and adaptive solutions are essential to restore
  functionality." and "Your critique is valid, and I see the flaw in my initial
  perspective."

The 18 October fix stopped the fallback from reaching the crucible and the abstraction
engine, but the fabricated belief was already kept.

## What changed

- `AURA_SWARM_SHARDS` is declared in `core/runtime/flags.py`'s registry, owned by
  `core.agency.agency_shard_work`, and defaults to off. The reason is the module's
  docstring and the comment on the flag.
- Off: `spawn_shard` admits nothing. `run_deliberation` records status `off` and
  spawns nothing, and the conductor's deliberation job returns `off` before it picks a
  topic. Goal genesis, self-architect refinement and evolution, social reflection and
  autonomous research send no shard and claim none. A repair is neither announced nor
  logged as a capacity failure.
- On: a shard gives her model one message in the runtime-evidence role, carrying
  `{"goal": ..., "context": ...}` as JSON. There is no system text, no schema, no
  output shape and no tools. Her reply is logged as it came, and a pulse follows a
  reply. Nothing is handed to the crucible, the abstraction engine or her insights.
  No reply means nothing happens.
- Deleted: the script, `StructuredLLM(ShardResponse)`, the fabricated fallback, the
  formatting-repair receipt, and the described tool protocol. Also deleted: the tool
  allowlists and the value-graph and causal-model restraints on shard tools, since a
  shard can no longer ask for a tool. They can be read at
  `c4052f62d:core/agency/agency_shard_work.py`.
- `config/prompting_baseline.json` loses the shard's 782 characters.
  `core/agency/DEPS` loses `core.schemas`.

## Tests

`tests/test_what_a_shard_sends_her_model.py` (9 tests) checks:

- the flag is declared off;
- while off, no shard is scheduled, a deliberation spawns nothing, a reflection sends
  no shard, a repair is neither sent nor announced, and the conductor holds no
  deliberation;
- on, her model is given exactly `shard_messages(goal, context)`. With the goal and
  context taken out, what remains is the two banner lines;
- on, an empty reply writes to no store, and a long reply that asks for a tool in the
  old protocol's words reaches no store and no tool.

`test_proposing_is_not_acting.py::test_a_shard_keeps_nothing_and_writes_nothing_for_her`
replaces the old pin and reads the module's code with its docstrings removed. Tests
that exercised the deleted tool dispatch and restraints were removed with it, from
`test_agency_core_governance.py`, `test_agency_core_runtime_contract.py`,
`test_constitutional_spine_fails_closed.py`, `test_governance_under_stress.py` and
`test_cognitive_pipeline_2026.py`.

## Gates

Passing: ruff (`make lint`), `tools/lint_ai_writing.py --code` (at baseline 58),
`make method-size-changed`, `make layering`, `make deps-check`, `make compile`,
`make smoke` (164 passed), and the 11 affected test files (139 passed).

Red on `origin/main` before this change, and unchanged by it:

- `make prompting`: `core/brain/executable_reasoning.py` (1958 > 1785) and
  `core/reasoning/written_as_constraints.py` (590 > 0).
- `tests/test_flag_ratchet.py::test_raw_env_reads_do_not_grow`: 577 against a budget
  of 567.
- `tools/lint_module_size.py` exits 1 on two stale entries
  (`core/brain/cognitive_engine.py`, `interface/routes/chat.py`).

## What is left

1. The 266 crucible beliefs are still in her belief store. The belief engine holds
   them in memory and rewrites the file, so moving them out has to happen while she is
   stopped. `tools/quarantine_what_she_did_not_conclude.py` is the place to add it.
2. The crucible has no caller now. Its LLM path is three scripted roles, and its
   default path puts a template around what it is given. It should be removed or
   rebuilt before anything calls it again.
3. The abstraction engine still words its request, and `self_play` still feeds it.
4. If shards are ever to do something, each spawner has to hand over data instead of an
   order. `SkillEvolution.spawn_evolution_shard` calls `spawn_shard(objective=...)`
   with no `goal`, which raises `TypeError`.
5. Tools offered natively would need a path with no prose fallback. The router's
   `think_and_act` falls back to a JSON contract written in prose. The restraint gate
   would also have to come back.
6. The self-architect's stability alert says "I'm initiating a critical shard to
   stabilize" and spawns nothing.
7. Not run live. Her app was not restarted, because the L01 attempt 3 chain owns it.
   The router's layers below a background call carrying only runtime evidence were not
   traced end to end. While the flag is off, that path does not run.
