# Aura memory card

Aura stores conversation history and semantic records on disk. In chat it
can retrieve relevant records and add them to the model's input. A question
about the first or last message can instead look up that turn by position.
The generated reply can still misquote the record it retrieved.
[Retrieval](core/memory/rag.py) · [Positional recall](core/conversation/grounded_recall.py)

## How stored records are used

| Record or operation | What the software does |
|---|---|
| Working context | Holds information for the current request. |
| Episodic memory | Stores events with context for later retrieval. |
| Semantic memory | Supplies stored facts and associated source information. |
| Procedures | Supplies a checked program to run when a request matches its problem type. |
| Recall and consolidation | Updates stored metadata, such as strength or related cues, as records are used or consolidated. |

[Memory architecture](docs/MEMORY_ARCHITECTURE.md) · [Episode storage](core/memory/episodic_memory.py) · [Semantic memory](core/brain/semantic_memory.py) ·
[Procedure selection](core/phases/kept_procedures_in_a_turn.py)

In a 40-task recall test on August 6, the 32B model answered 40 with relevant
history retrieved and none when given the most recent twelve turns. It
answered 39 when it could see the full transcript. That comparison measured
earlier facts outside the fixed recency window. [Run record](artifacts/ablation/capability_scorecard_32b_longhistory.json)

## Review saved memories

Use the memory panel to view records and inspect their provenance. Its
controls can edit, freeze, contest, mark false, or delete supported items.
The mutation handler runs the selected operation after an authorization
decision. The pane's download exports semantic memory records.
[Memory API](interface/routes/memory.py) · [Data and backup scope](DATA_CARD.md)
