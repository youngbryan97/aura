#!/bin/bash
OUTPUT="aura_architecture.txt"
echo "# Aura Core Architecture: The Integrated Mind (Zenith Phase)" > $OUTPUT
echo "# Compiled: $(date)" >> $OUTPUT
echo "# This file contains the primary source code for Aura's architecture." >> $OUTPUT

FILES=(
    "core/cognitive_kernel.py"
    "core/inner_monologue.py"
    "core/language_center.py"
    "core/epistemic_tracker.py"
    "core/inquiry_engine.py"
    "core/concept_linker.py"
    "core/insight_journal.py"
    "core/belief_revision.py"
    "core/agency_core.py"
    "core/volition.py"
    "core/narrative_thread.py"
    "core/agency_bus.py"
    "core/consciousness/qualia_synthesizer.py"
    "core/consciousness/self_prediction.py"
    "core/consciousness/global_workspace.py"
    "core/consciousness/attention_schema.py"
    "core/consciousness/conscious_core.py"
    "core/affect/damasio_v2.py"
    "core/emotional_coloring.py"
    "core/brain/personality_engine.py"
    "core/event_bus.py"
    "core/container.py"
    "core/main.py"
    "core/config.py"
    "core/orchestrator/main.py"
)

for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        echo -e "\n\n================================================================================" >> $OUTPUT
        echo "FILE: $f" >> $OUTPUT
        echo -e "================================================================================\n" >> $OUTPUT
        cat "$f" >> $OUTPUT
    fi
done
