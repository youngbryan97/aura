# Aura — User Guide

*Last reviewed against the tree: 2026-08-21.*

## Install
1. Download `Aura.dmg` from the releases page.
2. Drag `Aura.app` to your Applications folder.
3. Open it. The first-run wizard walks you through model selection,
   memory location, permissions, voice, and a fallback choice.

If you'd rather run from source (advanced):
```bash
git clone https://github.com/youngbryan97/aura
cd aura
make setup      # or: make setup-prod for a fail-closed install
make run        # foreground desktop launch
```

Full install detail, boot modes, and environment variables are in
[INSTALL.md](../INSTALL.md).

## Talk to Aura

Open her. The launch screen names every organ still warming up — Core,
Memory, Cortex, Voice, Autonomy — so you can see what's not ready yet
instead of guessing at a spinner. When the parts you need are up, the chat
input goes live.

Type and press Enter. If a reply is taking a while you'll get a thinking
indicator with an estimate. A turn on the local 27B cortex takes about a
hundred seconds — the median over twelve reasoning turns measured on
2026-08-28 was 102, the fastest 34 and the slowest 122. That is the model
thinking, not something being wrong.

## Manage Memory

The Memory tab has three views: **Episodic** (what happened), **Semantic**
(facts she has settled on), and **Goals**.

Open a memory and you get six controls:

- **Freeze** — it survives reaping. Nothing sweeps it later. Unfreeze puts it
  back in the ordinary pool.
- **Edit** — change the text.
- **Delete** — remove that one memory.
- **Contest** — mark it disputed without deleting it, so she knows the two of
  you disagree about it rather than losing the record.
- **Mark False** — say it is wrong. Different from contesting: this is a
  verdict, not a dispute.
- **Provenance** — where this memory came from and what it was derived from.

**Export the whole record** from the memory panel's **Backup & Export** tab.

It's her memory, but it's your data. All of it comes out in one file.

## Use Voice

Voice input needs explicit permission each session — click the mic button in
the header. The first time, macOS will ask for microphone access.

Settings → Voice holds both toggles, input and output.

## Common Issues
| Symptom | Likely cause | Fix |
|---|---|---|
| Banner: "My local Cortex is offline" | 32B failed to load | Check disk space first — the weights are about 20 GB. Then relaunch; there is no in-app model reset. `docs/runbooks/model-fails-to-load.md` has the ordered procedure. |
| "I'm under load right now" replies | RAM pressure over 90% | Close memory-heavy apps. Nothing in Settings compacts memory on demand; the runtime sheds load on its own. |
| Voice button greyed out | Permission revoked | Grant microphone access in macOS Privacy & Security. Settings → Desktop Access covers Screen Recording, Accessibility, and Automation. |
| Chat input stays disabled | Boot still warming | Check the boot screen at the top — wait for Cortex: Ready. |
| Aura answers, but flatly | An organ was missing from the turn | The turn surface reports which cognitive organs engaged; a missing organ is treated as a defect, not a note. |
| "I can't do that right now" | A capability exists but is unavailable | She distinguishes not having a capability from not being able to use it right now, and will say which. |

If something is wrong at the runtime level rather than the UI level, run
`aura doctor`, and `aura doctor --bundle` to produce a redacted diagnostics
tarball. Every incident class in [runbooks/](runbooks/) is written against
fields that bundle emits.

## Update Aura

Updates run through the release train (`tools/release_train.py`), not through
a channel picker:

```bash
make update        # autostash → fast-forward-only pull → compile sanity check
make update-live   # the same, plus a smoke run and a relaunch of the live instance
make rollback      # return to the last recorded good point
make release-status
```

Every update records a rollback point before it touches anything, and a
failed compile or smoke check stops the train rather than leaving a
half-updated tree. `make update` is deliberately boring: it refuses to
merge, so a diverged local tree fails loudly instead of resolving itself.

## Uninstall

Drag `Aura.app` to the trash. Your data stays at `~/.aura/` — deleting the
app does not delete what she remembers.

To remove that too:

```bash
rm -rf ~/.aura
```

That one is not reversible. Export from the memory panel's Backup & Export
tab first if there's any chance you want it later.

For deeper docs see `docs/OPERATOR_GUIDE.md` and `docs/RESEARCH_GUIDE.md`.
