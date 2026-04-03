# Dialogue Corpora Layout

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
