# Aura — User Guide

<!-- reader-navigation:start -->
> Aura runs on an Apple Silicon Mac. The desktop app opens a conversation, and local software handles speech, memory, tools, and background work.
>
> Start with the parts you need and check what the machine can run. The setup guides give the hardware requirements and link to measured capabilities. To work on a document, Aura needs a model that has finished loading, permission to read the file, and enough memory to handle the request.
>
> [Plain-language guide](PLAIN_ENGLISH_GUIDE.md#getting-started) · [Terms and examples](VARIABLES_EXPLAINED.md#getting-started) · [Evidence and related work](DEMOS_AND_RELATED_WORK.md#runtime)
> [All docs](DOCUMENTATION_INDEX.md) · [Glossary](GLOSSARY.md) · [Worked formulas](FORMULAS.md)
<!-- reader-navigation:end -->

Use this guide to start the desktop app, talk to Aura, and manage its
memory and voice settings.

## Contents

- [Install](#install) · [Talk to Aura](#talk-to-aura) · [Use voice](#use-voice)
- [Manage memory](#manage-memory) · [Troubleshoot](#common-issues) · [Update](#update-aura) · [Uninstall](#uninstall)

## Install

The setup procedure was reviewed on 25 October 2026. Install the app from source. The
[November GitHub release](https://github.com/youngbryan97/aura/releases/tag/isc-preprint-2026-11)
accompanies the ISC preprint and has no desktop installer attached. If you
have a separately built `Aura.app`, use the setup instructions and
provenance for that build.

```bash
git clone https://github.com/youngbryan97/aura
cd aura
make setup      # or: make setup-prod for a fail-closed install
make run        # foreground desktop launch
```

Full install detail, boot modes, and environment variables are in
[INSTALL.md](../INSTALL.md).

## Talk to Aura

### Start a conversation

Open Aura. The startup screen shows the status of Core runtime, Memory,
Cortex (the local language model), Voice, and Autonomy as they load. The
chat box activates when the required components are ready.

Type a message and press Enter. The thinking indicator shows an estimated
wait time. A twelve-task sample from 28 August recorded a median of
102 seconds. The fastest task took 34 seconds and the slowest took 122.
Ordinary chat and voice need their own measurements. Check progress
and logs if you need to tell ongoing inference from a stall.

### When Aura looks something up or runs a procedure

Factual questions can use the offline Wikipedia copy and pages saved by
earlier searches. Each passage carries its source and date.

Questions about large primes, prime factors, or a date calculation can run
a kept procedure before the draft. For a wording match, the procedure's
answer can replace a draft that omits it. For a meaning match, Aura sees
the computed answer as evidence and returns the model's reply.

### Correct an answer

Give the correction as part of your message, for example “No, it's 1049.”
Aura keeps it as a solved example. It can learn a procedure from several
examples during idle time. When the correction conflicts with an existing
kept procedure, Aura marks it disputed.

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

The export covers the stores implemented by that endpoint. Consult
[the data card](../DATA_CARD.md) for scope; it is not an export of every model
cache, experiment artifact or external-service record.

## Use Voice

Voice input needs explicit permission each session — click the mic button in the header to enable it. The first time, macOS will ask for microphone access.

Settings → Voice holds both toggles, input (speech-to-text) and output (text-to-speech).

## Common Issues
| Symptom | Likely cause | Fix |
|---|---|---|
| Banner: "My local Cortex is offline" | The large language model (32B/27B) failed to load | Check available disk space first — the model weights need about 20 GB. Identify the failed process and existing workload before an authorised graceful relaunch; do not start a second Cortex beside a live instance. Follow the step-by-step instructions in `docs/runbooks/model-fails-to-load.md`. |
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
make update-live   # includes a runtime relaunch; requires an authorised restart window
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
