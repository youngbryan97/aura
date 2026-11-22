# Dialogue Corpora Layout

<!-- reader-navigation:start -->
> **In plain English:** Memory keeps records beyond the immediate request and selects relevant records for the current turn.
>
> **Why it matters:** Useful continuity depends on recalling the right information, retaining its source and correcting or deleting it when needed.
>
> **In everyday use:** If you correct a fact about a project, later answers should use the correction rather than retrieving the older claim as equally reliable.
>
> [Example and metaphor](../../docs/PLAIN_ENGLISH_GUIDE.md#memory) · [Variables and practical meanings](../../docs/VARIABLES_EXPLAINED.md#memory) · [Demos and related work](../../docs/DEMOS_AND_RELATED_WORK.md#memory) · [Document map](../../docs/DOCUMENTATION_INDEX.md)
> [Glossary](../../docs/GLOSSARY.md) · [Formulas with worked examples](../../docs/FORMULAS.md)
<!-- reader-navigation:end -->

Put optional local transcript files here if you want to deepen Aura's dialogue-source attractors.

Canonical folders:
- `sypha/`
- `edi/`
- `lucy/`
- `kokoro/`
- `ashley_too/`
- `mirana/`
- `sara_v3/`

Transcript format:

```text
Speaker Name: line of dialogue
Other Speaker: reply
Speaker Name: follow-up
```

Guidelines:
- use locally sourced transcripts you are allowed to keep
- do not commit copyrighted raw corpora if you do not want them in the repo
- keep one scene or exchange per file when possible
- the ingest pipeline learns structure and style features; it does not need every line ever spoken

To ingest:

```bash
python scripts/ingest_dialogue_corpora.py
```
