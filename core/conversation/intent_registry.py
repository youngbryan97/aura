"""The matchers themselves, each with the phrasings it was actually probed on.

Every example that reads like an odd way to ask something IS one: a real
question that the matcher beside it did not recognise, or wrongly claimed, in
live use.
"""

from __future__ import annotations

from core.conversation.intent_contract import IntentMatcher, register_intent_matcher


def _install() -> None:
    from core.autonomy.research_goal_filter import is_runtime_status_goal
    from core.conversation.capability_denial import denied_registered_capabilities
    from core.conversation.filesystem_check import (
        requested_file_read,
        requested_filesystem_count,
    )
    from core.introspection.decision_provenance import asks_why_she_did_that
    from core.knowledge.corpus_grounding import is_corpus_groundable
    from core.knowledge.knowledge_gap import detect_knowledge_gap
    from core.runtime.self_state_intent import asks_about_own_capabilities
    from core.skills.file_modification_intent import requested_file_modification

    for matcher in (
        IntentMatcher(
            name="requested_file_modification",
            predicate=requested_file_modification,
            where="core/skills/file_modification_intent.py",
            examples=(
                # The live miss: this planned an overwrite of a file that
                # already held a line the user wanted kept.
                'append a line saying "line two" to aura-test-note.txt',
                'add the line "line two" to the end of notes.txt',
                "add a note to the bottom of my todo.md",
                "put a header at the top of readme.md",
                "stick a footer onto the end of report.md",
            ),
            counter_examples=(
                # Creating and replacing must NOT read as adding, or a new
                # file inherits whatever a stale file of that name held.
                "create a file called aura-test-note.txt containing hello",
                "write a haiku to a file called poem.txt",
                "overwrite notes.txt with the new list",
                "save the summary as summary.md",
            ),
        ),
        IntentMatcher(
            name="asks_why_she_did_that",
            predicate=asks_why_she_did_that,
            where="core/introspection/decision_provenance.py",
            examples=(
                "why did you do that?",
                "why did you choose that file?",
                "why did you skip the verification step?",
                "why do you think that is?",
            ),
            counter_examples=(
                # A casual question about human psychology once came back with
                # the runtime's phase-by-phase provenance dump stapled under it.
                "why do you think people find it hard to admit they were wrong?",
                "why do humans procrastinate?",
                "why is the sky blue?",
            ),
        ),
        IntentMatcher(
            name="requested_file_read",
            predicate=requested_file_read,
            where="core/conversation/filesystem_check.py",
            examples=(
                "read the file CONTRIBUTING.md and tell me the first rule",
                "what does CONTRIBUTING.md say about tests?",
                "open core/config.py",
            ),
            counter_examples=(
                "read /etc/passwd",
                "open ../../../etc/hosts",
                "how are you doing",
            ),
        ),
        IntentMatcher(
            name="requested_filesystem_count",
            predicate=requested_filesystem_count,
            where="core/conversation/filesystem_check.py",
            examples=(
                "count the .py files in core/introspection and tell me the number",
                "how many python files live in core/introspection?",
                "how many files do we have in core/introspection",
            ),
            counter_examples=(
                "how many files are in /etc",
                "how many test files are in core",
                "what is the weather",
            ),
        ),
        IntentMatcher(
            name="is_corpus_groundable",
            predicate=is_corpus_groundable,
            where="core/knowledge/corpus_grounding.py",
            examples=(
                "explain the difference between correlation and causation",
                "what is a confounding variable",
            ),
            counter_examples=(
                "how are you doing right now?",
                "what did I ask you first today?",
                "open my notes folder and save the draft",
                "hey",
            ),
        ),
        IntentMatcher(
            name="detect_knowledge_gap",
            predicate=lambda text: detect_knowledge_gap(text, ""),
            where="core/knowledge/knowledge_gap.py",
            examples=(
                "I think it's around 1969, but I'm not certain when Apollo 11 landed.",
                "I do not know who wrote the Rust borrow checker originally.",
                "Off the top of my head, Django was released around 2005.",
            ),
            counter_examples=(
                "The capital of France is Paris.",
                # Taste, not a knowledge gap: no reference work settles it.
                "I think you'd enjoy that film, honestly.",
                "I'm not sure how you're feeling about it.",
                "I'm not sure I follow.",
            ),
        ),
        IntentMatcher(
            name="denied_registered_capabilities",
            predicate=denied_registered_capabilities,
            where="core/conversation/capability_denial.py",
            examples=(
                "I don't have file system access or the ability to count files.",
                "I cannot read your screen.",
            ),
            counter_examples=(
                # Declining is a CHOICE and must never be overridden.
                "I would rather not help with that.",
                "I counted ten files and pushed the change.",
            ),
        ),
        IntentMatcher(
            name="asks_about_own_capabilities",
            predicate=asks_about_own_capabilities,
            where="core/runtime/self_state_intent.py",
            examples=(
                "what can you actually do?",
                "what are your capabilities?",
            ),
            counter_examples=(
                "what is 2 + 2",
                "read CONTRIBUTING.md",
            ),
        ),
        IntentMatcher(
            name="is_runtime_status_goal",
            predicate=is_runtime_status_goal,
            where="core/autonomy/research_goal_filter.py",
            examples=(
                "Aura is idle.",
                "Aura is typing...",
                "The runtime is warming up",
            ),
            counter_examples=(
                "Idle animation techniques in game design",
                "Quantum Neural Network Architectures",
                "Why the scheduler thinks Aura is idle",
            ),
        ),
    ):
        register_intent_matcher(matcher)


_install()
