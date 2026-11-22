# Use Aura

Aura's November GitHub release contains the research source. Set up the
app with Python 3.12 and a local model on Apple Silicon. The installation
guide explains the model files a source checkout needs.
[Release](https://github.com/youngbryan97/aura/releases/tag/isc-preprint-2026-11) · [Installation](../INSTALL.md)

## Chat

Open the desktop app and enter a message when the chat field is available.
For a factual question, ask Aura which source it used. It can search locally
indexed articles when the index is installed. For a task that uses a tool,
check the action record to see whether the tool ran and how it ended.
[How a turn works](../HOW_IT_WORKS.md)

To teach an answer, correct it in the same conversation. Aura can save a
usable correction as a worked example. Samples it stores are available
for later learning. [Live date-question trial](evidence/L09_LEARNING_IN_USE_LOOP_2026-10-18.md)

## Voice

Enable voice input or output in Settings. Use the microphone control to
start listening and grant microphone access if macOS asks. For how Aura
decides when to listen or start speaking, see the [voice guide](AMBIENT_VOICE_AND_MEDIA.md).

## Saved memories

Open the memory panel to review saved items. The controls let you edit or
delete an item, freeze it against automated cleanup, contest it, or mark
it false. Open **Provenance** to see the record's source when available.
Use **Backup & Export** to download semantic memory records.
[Memory card](../MEMORY_CARD.md) · [Data card](../DATA_CARD.md)

## When something stops working

| Symptom | First check |
|---|---|
| Chat stays disabled or the model reports that it is offline | Check startup progress and the model logs. [Model loading runbook](runbooks/model-fails-to-load.md) |
| The mic control is unavailable | Check the voice setting and the app's microphone permission in macOS Privacy & Security. |
| A request stalls | Check whether the model is still generating and whether the process is under memory pressure. [Resource exhaustion](runbooks/resource-exhaustion.md) |
| A file or screen is inaccessible | Check the request's permission and any related macOS permission before retrying. [Platform and permissions](PLATFORM_POSTURE.md) |

Run `aura doctor --bundle` from an installed environment to collect a
redacted diagnostic archive. [Operator guide](OPERATOR_GUIDE.md)

## Update or remove the app

For a source installation, run `make update` to fetch and fast-forward,
then run `make release-status` to check the recorded revision. `make update-live`
also relaunches the runtime. Schedule that command for a time when the app
can be interrupted. [Release lifecycle](RELEASE_LIFECYCLE.md)

Quit Aura before removing its app bundle or checkout. The default state
directory is `~/.aura/`. If you plan to erase that directory, first export
the records you want to keep. [Data locations and retention](../DATA_CARD.md)
