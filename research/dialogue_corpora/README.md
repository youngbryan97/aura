# Dialogue Corpora Layout

<!-- reader-navigation:start -->
> Aura keeps records of events, facts, preferences, and procedures. It looks for records relevant to the question or task in front of it.
>
> A memory is useful when you can find its source, correct it, or remove it. If you correct the date of a project, a later reply should use the correction and remember where it came from.
>
> [Plain-language guide](../../docs/PLAIN_ENGLISH_GUIDE.md#memory) · [Terms and examples](../../docs/VARIABLES_EXPLAINED.md#memory) · [Evidence and related work](../../docs/DEMOS_AND_RELATED_WORK.md#memory)
> [All docs](../../docs/DOCUMENTATION_INDEX.md) · [Glossary](../../docs/GLOSSARY.md) · [Worked formulas](../../docs/FORMULAS.md)
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
