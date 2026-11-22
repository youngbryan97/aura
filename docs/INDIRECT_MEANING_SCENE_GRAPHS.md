# Indirect meaning and scene graphs

This is an opt-in interpretation path, not a claim that Aura knows a speaker's
private intent. It uses the existing `structure_mapping.Graph` for arbitrary
arity relations. The graph has no built-in list of story types, jokes, cultures,
or domains. Aura's local language model may propose its nodes and relations;
the retained source records and the assessor keep those proposals separate from
observations.

## What the path represents

| Part of an indirect exchange | Representation and check |
|---|---|
| An overt story or joke | A literal reading remains present even when a second reading is proposed. |
| A possible reference to the current situation | Separate story and situation graphs, with one-to-one entity and predicate correspondences. The matcher returns tied best mappings rather than choosing one silently. |
| A sequence of actions and consequences | Relations may bind events as arguments. A projected relation absent from the situation graph is an unobserved possibility, not a fact. |
| A story heard by several people | Each audience has a separate access hypothesis and a separate proposed communicative function. |
| Speech under monitoring | The monitor's access is represented separately from the addressee's access. An access claim needs a cited source excerpt but remains a proposal. |
| Humor used to soften or conceal a warning | Delivery cues are retained as measured cues. A joke and a warning can coexist; neither cue proves intent. |
| A later hint about the story's importance | The later source can alter the comparison only after its observation time. It cannot rewrite what an earlier observer knew. |
| A plausible but wrong interpretation | Rival predictions face source-bound observations. Reports and independent observations remain distinct, and conflicting observations remain disputed. |
| Several possible referents | Alternative role mappings and the role-swap control expose ambiguity. The assessor does not promote one referent merely because its graph matches. |

`core/cognition/indirect_meaning_proposals.py` asks the local language model
for typed, source-cited graphs and readings. The schema admits arbitrary
relation names and one to eight arguments per relation, so the same intake can
represent a transfer of an object, a causal event chain, or an institutional
decision. An excerpt must occur in the supplied source. That check proves the
excerpt exists; it does not prove the proposed relation is entailed by it.
The model cannot submit an observation through this schema.

The callable runtime path is `semantic_runtime.consult_indirect_episode`.
It requires every `UsageEvent` to be retained unchanged before and after the
model call. It accepts transient source passages because the long-term usage
store retains bounded lexical evidence, not whole conversation transcripts.
New usage records retain a digest of the exact text, including punctuation;
legacy records without that digest cannot support this source-exact path.
The result has `serving_authority=false`. It is not yet an automatic claim or
answer modifier for every chat turn.

## Evidence boundary

The shared graph format is open to new domains. The current exhaustive
structure matcher deliberately refuses graphs over seven objects; larger
graphs need a separately measured search method. The alternative-mapping API
also refuses an incomplete predicate search. A cited, matching graph is a
structural hypothesis. A later observed consequence can support or contradict
a reading, but speaker intent remains unmeasured unless an independent source
supplies it. No test here establishes broad G-ledger transfer, model fusion, or
frontier reasoning gain.

Focused controls live in `tests/test_indirect_meaning.py`,
`tests/test_indirect_meaning_proposals.py`, and
`tests/test_partial_structure_mapping.py`. Their stories use a different
setting from the example that motivated this work and test forged citations,
time cutoffs, role swaps, rival audiences, unobserved consequences, and
retained-source changes during a model call.
