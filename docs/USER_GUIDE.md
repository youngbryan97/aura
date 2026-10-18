# Aura — User Guide

*Last reviewed against the tree: 2026-08-21.*

## Install
1. Download `Aura.dmg` from the releases page.
2. Drag `Aura.app` to your Applications folder.
3. Open it. The first-run setup wizard walks you through choosing an AI model, setting where memories are stored, granting required system permissions, setting up voice, and choosing a fallback model if needed.

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

Open Aura. The startup screen lists each system component as it loads — Core runtime, Memory, Cortex (the local language model), Voice, and Autonomy. This shows you exactly what is still starting up instead of showing a generic loading spinner. When the parts you need finish loading, the chat input box becomes active.

Type your message and press Enter. If a reply takes a while, you will see a thinking indicator with an estimated wait time. Running complex reasoning on the local 27-billion-parameter model (27B Cortex) takes around 100 seconds per response. In benchmark testing on 2026-08-28, the median response time across twelve reasoning tasks was 102 seconds (the fastest was 34 and the slowest was 122). When this happens, the model is simply working through the problem, not frozen.

Some answers are computed rather than written. For kinds of problem she keeps a procedure for (large primes, prime factors, date arithmetic once she has learned it), her chat runs the procedure before she replies. When your question is worded the way that kind's examples are and her draft does not state the computed answer, the computed answer is what you see; when it is worded otherwise, the computed answer is put in front of her as evidence and her reply stands. Factual questions are looked up in her offline Wikipedia copy and in pages her searches kept, and each source is labelled with where it came from and its date.

If she gets something wrong, say so with the answer: "No, it's 1049." She keeps that as a solved example, and while she is idle she can learn a procedure from several of them. An answer that contradicts a procedure she already trusts is kept as disputed rather than believed.

## Manage Memory

The Memory tab provides three views: **Episodic** (what happened in past conversations), **Semantic** (facts she has settled on), and **Goals** (active objectives).

Open a memory and you get six controls:

- **Freeze** — Protects the memory so automated cleanup never deletes it. Clicking Unfreeze returns it to standard memory management.
- **Edit** — Change the text of the memory directly.
- **Delete** — Remove that one memory permanently.
- **Contest** — Mark the memory as disputed without deleting it. This keeps the record while letting Aura know the two of you disagree about it.
- **Mark False** — Mark the memory as factually wrong. Unlike contesting, this records a final verdict that the statement is incorrect.
- **Provenance** — View where this memory came from and what observations or inputs created it.

**Export the whole record** from the memory panel's **Backup & Export** tab.

It's her memory, but it's your data. All of it comes out in one file.

## Use Voice

Voice input needs explicit permission each session — click the mic button in the header to enable it. The first time, macOS will ask for microphone access.

Settings → Voice holds both toggles, input (speech-to-text) and output (text-to-speech).

## Common Issues
| Symptom | Likely cause | Fix |
|---|---|---|
| Banner: "My local Cortex is offline" | The large language model (32B/27B) failed to load | Check available disk space first — the model weights need about 20 GB. Then relaunch the app, as there is no reset button inside the interface. Follow the step-by-step instructions in `docs/runbooks/model-fails-to-load.md`. |
| "I'm under load right now" replies | RAM pressure over 90% | Close memory-heavy apps. Aura frees up memory automatically when needed; there is no manual "clear memory" button in Settings. |
| Voice button greyed out | Permission revoked | Grant microphone access in macOS Privacy & Security. In Aura, Settings → Desktop Access covers Screen Recording, Accessibility, and Automation permissions. |
| Chat input stays disabled | Boot still warming | Check the startup indicator at the top of the window and wait until it displays "Cortex: Ready". |
| Aura answers, but flatly | An organ was missing from the turn | The turn surface reports which cognitive organs engaged; a missing organ is treated as a defect by the runtime, not expected behavior. |
| "I can't do that right now" | A capability exists but is unavailable | She distinguishes not having a capability from not being able to use it right now, and will say which is the case. |

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
