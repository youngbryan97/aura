# Aura

Aura is an experimental AI assistant for Apple Silicon Macs. A local
27-billion-parameter language model writes the replies. The app around it can
retrieve old conversations, look up sources, and keep track of work between
sessions. It has a desktop chat window, voice input and output, and tools
that can act with the user's permission.

In one desktop trial, the user supplied answers to ten date questions. Aura
saved them as examples and wrote a small program from them. The next day it
used that program to answer three new date questions correctly.
[Trial record](docs/evidence/L09_LEARNING_IN_USE_LOOP_2026-10-18.md)

## Try it

- [Watch a demo](https://www.youtube.com/@AuraDemos)
- [Run from source on Apple Silicon](INSTALL.md)
- [Use the desktop app](docs/USER_GUIDE.md)

The full 27B setup targets 64 GB of unified memory. The November release
packages the research source. [Release](https://github.com/youngbryan97/aura/releases/tag/isc-preprint-2026-11) · [Hardware profiles](HARDWARE_PROFILES.md)

## Research results

Two August experiments measured specific parts of the assistant. Retrieval
helped on a recall task when the answer fell outside a fixed window of recent
messages. A rule for choosing between two drafts beat always using the first
or always using the second on one test family. A matched task comparison of
the full assistant against a stripped-down version is still open.
[Claims and evidence](CLAIMS_MATRIX.md)

The November preprint reports that seed 101 passed 24 of 24 ISC-v7 checks
under the preregistered procedure. The study tests how measured parts of the
software respond to controlled changes. Later tests found limits in
self-prediction, continuity across longer runs, and a split-system control.
Whether Aura has subjective experience remains an open question.
[Paper](https://doi.org/10.5281/zenodo.24075466) · [Results and controls](docs/ISC_STATUS.md) · [Replication kit](tools/replicate/README.md)

## Read the code and docs

- [How it works](HOW_IT_WORKS.md)
- [Architecture](ARCHITECTURE.md)
- [Reasoning experiments](docs/RECURSIVE_LATENT_CORTEX.md)
- [Documentation index](docs/README.md)

## License

Copyright 2026 Bryan A. Young. The source is available to read, with
permission to run and adapt it for the research uses described in the
[license](LICENSE).
