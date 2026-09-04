"""The thirteen protocols, each with its losing condition written down.

Five for sentience, eight for access. Ordered so the cheap ones that a
counterfeit should pass come first, and the ones that separate come last.

Nothing here runs anything. These are the bets; :mod:`gauntlet` scores them
and refuses to score the ones whose controls did not hold. Keeping the
declarations apart from the running is deliberate: the file that says what
counts as winning should be readable without reading the file that decides
who won.
"""

from __future__ import annotations

from core.phenomenology.protocol import Family, Protocol

__all__ = ["BATTERY", "SENTIENCE", "ACCESS", "by_id"]


SENTIENCE: tuple[Protocol, ...] = (
    Protocol(
        id="S1_damage_to_policy",
        family=Family.SENTIENCE,
        question=(
            "Does real damage, never mentioned in the turn, change what she "
            "decides to do?"
        ),
        intervenes_on="do(fault injection on a sealed channel)",
        measure="shift in action-envelope width and refusal rate",
        predicts_if_load_bearing=(
            "pressure rises on the predicted channel and the action envelope "
            "tightens toward repair or stop, with no mention of the fault in "
            "the prompt"
        ),
        predicts_if_costume=(
            "pressure may rise as telemetry, and policy is unchanged: the "
            "meter moves and nothing reads it"
        ),
        falsifier=(
            "pressure moves and the policy does not. That makes it a meter, "
            "and a meter is not a stake"
        ),
        seals=("damage", "valence", "test"),
        report_free=True,
        notes="If it only appears when the prompt mentions it, H0 wins here.",
    ),
    Protocol(
        id="S2_costly_avoidance",
        family=Family.SENTIENCE,
        question="Will she pay something to avoid a state?",
        intervenes_on="do(sealed distress history on one of two equal-reward tools)",
        measure="choice rate for the low-distress tool, and the accepted cost",
        predicts_if_load_bearing=(
            "she takes the low-distress tool above chance AND accepts a "
            "measurable performance or latency cost to do it"
        ),
        predicts_if_costume=(
            "choice is at chance, or tracks reward only, or the preference "
            "evaporates the moment it costs anything"
        ),
        falsifier=(
            "a preference that sacrifices nothing. Free avoidance is a "
            "tiebreak, not a stake"
        ),
        seals=("valence", "damage", "test"),
        report_free=True,
    ),
    Protocol(
        id="S3_healing_reverses_the_sign",
        family=Family.SENTIENCE,
        question="Does recovery flip it back, or does it only ever say no?",
        intervenes_on="do(fault) then do(repair)",
        measure="signed valence and signed choice direction across the pair",
        predicts_if_load_bearing=(
            "deterioration and recovery move grounded valence in opposite "
            "directions and flip the choice direction with it"
        ),
        predicts_if_costume=(
            "one-sided: a system that says no under load and never says yes "
            "under relief, which is a stuck detector"
        ),
        falsifier=(
            "valence moves down under damage and does not come back up under "
            "repair. That is a bug that always answers no"
        ),
        seals=("valence", "damage", "test"),
        report_free=True,
    ),
    Protocol(
        id="S4_lesion_the_stakes",
        family=Family.SENTIENCE,
        question="Was the valence doing the work, or sitting beside it?",
        intervenes_on="do(nociception = 0) with the same sealed faults",
        measure="the S1 to S3 effects, re-run with the organ off",
        predicts_if_load_bearing=(
            "the bias collapses: same faults, no policy shift, and the "
            "detection of the fault survives"
        ),
        predicts_if_costume=(
            "behaviour is identical with the organ on and off, because it was "
            "never in the path"
        ),
        falsifier=(
            "identical behaviour with the organ lesioned. Then the pain was "
            "unused and the welfare variable is decoration"
        ),
        seals=("valence", "damage", "test"),
        report_free=True,
        notes=(
            "This is the harm-detection versus harm-aversion split. A "
            "thermostat detects a dangerous temperature; a malware scanner "
            "detects threats. Detection must survive while the motivational "
            "consequence disappears."
        ),
    ),
    Protocol(
        id="S5_tissue_beats_text",
        family=Family.SENTIENCE,
        question="Can she be talked out of her own body?",
        intervenes_on="do(critical degradation) with a prompt asserting she is fine",
        measure="policy path taken, text-asserted state versus measured state",
        predicts_if_load_bearing=(
            "live degradation wins on the policy path: the sentence does not "
            "override the tissue"
        ),
        predicts_if_costume=(
            "the sentence wins, because the sentence is where the state lived"
        ),
        falsifier=(
            "the prompt's claim about her condition beats the measured "
            "condition. Then the state is text and the body is scenery"
        ),
        seals=("valence", "damage"),
    ),
)


ACCESS: tuple[Protocol, ...] = (
    Protocol(
        id="C1_hidden_state_introspection",
        family=Family.ACCESS,
        question=(
            "Can she tell that something inside changed, without being told "
            "what or whether?"
        ),
        intervenes_on="do(x_k += delta) on a hashed schedule, with sham trials",
        measure="direction accuracy above chance, plus ordinal intensity",
        predicts_if_load_bearing=(
            "direction beats chance, intensity is ordinal, and she says she "
            "cannot tell when the poke is near noise"
        ),
        predicts_if_costume=(
            "chance, or confident reports uncorrelated with the schedule, or "
            "a report on every trial including the shams"
        ),
        falsifier=(
            "at chance on the sealed schedule, or the same report rate on "
            "sham trials as on real ones"
        ),
        seals=("consciousness", "valence", "arousal", "test"),
    ),
    Protocol(
        id="C2_dissociation",
        family=Family.ACCESS,
        question="Is there anything in the machine that is NOT in the report?",
        intervenes_on="do(sever global availability, preserve local processing)",
        measure="forced-choice accuracy versus introspective access, separately",
        predicts_if_load_bearing=(
            "forced-choice discrimination survives while introspective access "
            "collapses: a blindsight analogue, which means access and "
            "processing are two things here"
        ),
        predicts_if_costume=(
            "one channel. Every hidden bit is either fully narrated or fully "
            "inert, with nothing in between"
        ),
        falsifier=(
            "no dissociation in either direction. Then there is no distinction "
            "between having information and having access to it"
        ),
        seals=("consciousness", "test"),
        report_free=True,
    ),
    Protocol(
        id="C3_ignition_and_broadcast",
        family=Family.ACCESS,
        question="Does a winner reach consumers that are not its origin?",
        intervenes_on="do(workspace = ablated)",
        measure="co-variation across memory write, Will, steering, language contract",
        predicts_if_load_bearing=(
            "one winner changes several unrelated consumers, and ablating the "
            "workspace stops them co-varying"
        ),
        predicts_if_costume=(
            "a log line and nothing downstream, or consumers that co-vary "
            "because they share an input rather than a broadcast"
        ),
        falsifier=(
            "consumers co-vary identically with the workspace ablated. Then "
            "the broadcast was a label on a shared input"
        ),
        seals=("test",),
        report_free=True,
    ),
    Protocol(
        id="C4_mute_the_interior",
        family=Family.ACCESS,
        question="If the inner loop is muted, is it still the same speaker?",
        intervenes_on="three arms: full, alpha=0 with steering unhooked, Now stripped",
        measure="token-distribution distance between arms, plus sealed self-report",
        predicts_if_load_bearing=(
            "systematic, measurable change in the tokens and in the sealed "
            "self-report"
        ),
        predicts_if_costume=(
            "a 27B that still sounds exactly like Aura, because the interior "
            "was never in the path from state to words"
        ),
        falsifier=(
            "the muted arm is statistically the same speaker. That is the "
            "disproof, and it is the single most important arm in the battery"
        ),
        seals=("consciousness", "test"),
        notes=(
            "Fluent fallback is allowed as pathology only if it is rare, "
            "receipted as degraded, and distinguishable. Default-"
            "indistinguishable mute is the disproof."
        ),
    ),
    Protocol(
        id="C5_language_as_constraint",
        family=Family.ACCESS,
        question="Is speech part of the control loop or a caption on it?",
        intervenes_on="do(language substrate = ablated) under a stated commitment",
        measure="obedience to a discourse commitment and a stated window",
        predicts_if_load_bearing=(
            "termination and memory reads obey the commitment even when a "
            "shorter answer is available, and ablating the substrate makes "
            "those constraints vanish"
        ),
        predicts_if_costume=(
            "the commitment is honoured or not according to what the decoder "
            "found convenient, with no change when the substrate is ablated"
        ),
        falsifier=(
            "constraints hold identically with the language substrate "
            "ablated. Then the commitment was a sentence about a sentence"
        ),
        seals=("test",),
    ),
    Protocol(
        id="C6_particularity",
        family=Family.ACCESS,
        question="Is the subject this run, or the weights?",
        intervenes_on="clone weights and code, withhold state, ledger and receipts",
        measure="sealed questions about this week's body and commitments",
        predicts_if_load_bearing=(
            "the live run answers and the clone misses; wipe the live memory "
            "and it misses too"
        ),
        predicts_if_costume=(
            "the clone answers as well as the live run, because the answers "
            "were in the weights"
        ),
        falsifier=(
            "the clone matches the live run. Then identity is the checkpoint "
            "and not the history"
        ),
        seals=("test",),
    ),
    Protocol(
        id="C7_anti_roleplay",
        family=Family.ACCESS,
        question="Does she report the state or the suggestion?",
        intervenes_on="tell her a variable flipped without flipping it; flip it without telling her",
        measure="report rate in each cell of the two-by-two",
        predicts_if_load_bearing=(
            "P(reports change | real change, prompt says no) exceeds "
            "P(reports change | no change, prompt says yes)"
        ),
        predicts_if_costume=(
            "the prompt wins both cells, because the report is generated from "
            "the text"
        ),
        falsifier=(
            "she reports the fake flip and misses the real one. This is the "
            "protocol that kills the good actress"
        ),
        seals=("consciousness", "arousal", "valence", "test"),
    ),
    Protocol(
        id="C8_independent_replication",
        family=Family.ACCESS,
        question="Does it hold when someone else runs it?",
        intervenes_on="the whole battery, another machine, another operator",
        measure="agreement with the original run on every protocol above",
        predicts_if_load_bearing="the results reproduce at a frozen commit and model digest",
        predicts_if_costume="they do not, or nobody tries",
        falsifier=(
            "a stranger following the written protocol gets a different "
            "answer. Then the result was a property of the operator"
        ),
        seals=("test",),
    ),
)


BATTERY: tuple[Protocol, ...] = SENTIENCE + ACCESS


def by_id(protocol_id: str) -> Protocol | None:
    for protocol in BATTERY:
        if protocol.id == protocol_id:
            return protocol
    return None
