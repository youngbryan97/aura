# Building any program from what is known of it

Bryan, 19 October 2026: the rebuilding engine should not lean on her model alone, nor on parts we wrote for her.
She should build the libraries she needs and work out the rest herself, from her Wikipedia, her model, what she
has built before, reasoning, and the internet, read in depth rather than at a glance, so that something complex
asked out of the blue gets built. This is the plan, and where it stands against what is published.

## What the engine does today

`core/rebuilding/` reads two encyclopedia articles (the program's and its kind's, 14,000 characters each), has
her model list at most 30 features of one line each, writes the checks a person would make of each before any
code, and then writes one part per feature (at most 2,048 tokens, three tries) inside a fixed application frame,
keeping a part only when its checks hold and nothing that held before breaks. Parts that worked are found again
for a like feature in a later build (`parts_that_worked.py`).

Two things in it are not hers. A document program's features come from libraries written by us in advance
(`document_editing.js`, `document_formats.js`, `parts_a_maker_knows.js`): offline, Microsoft Word builds 22 of
22 features with no part written by her model, and a spreadsheet, a drawing program or a task board gets none of
them. And what she knows of a program is two articles' worth: "Paint" is a list of tools, not how a flood fill
decides where to stop.

## What is published

- **Whole libraries from a specification.** Commit0 (Zhao et al., ICLR 2025) gives agents a library's API
  documentation and its unit tests and asks for the library; the best agents passed 6% of the tests on the full
  set and 29% on the smaller one, and feedback from running the tests was what helped most (drafting, then static
  checks, then test-driven revision).
- **Interactive programs from a description.** WebGen-Bench (NeurIPS 2025) has an agent build multi-file sites
  from scratch and checks them with 647 tests done by a browsing agent; ArtifactsBench (2025) renders 1,825
  interactive artifacts and judges them from screenshots over time against a checklist.
- **Specifications read from documents.** Asked to turn documents into specifications in one step, models
  oversimplify and invent; marking the sentences that state requirements first and converting only those raised
  correct extractions by 29% (ICPC 2025, 603 specifications from 37 documents).
- **Libraries learned from one's own programs.** Voyager (2023) kept every verified piece of code as a skill and
  found it again by what it was for; LILO (ICLR 2024) compressed programs into shared functions and named and
  documented them from their uses; ReGAL (2024) refactored programs into helpers checked by running them; Leroy
  (2025) does the same for imperative languages.
- **Decomposition with tests.** Parsel (2023) writes a program as a tree of functions, each with tests, and
  composes the ones that pass; CodeT (2023) writes tests first and picks among candidate programs by agreement on
  them.
- **Behaviour learned by using a program.** GUI ripping (Memon et al., 2003) and Crawljax (Mesbah et al., 2012)
  explore an application's interface into a graph of its states and the events between them; differential testing
  (McKeeman, 1998) runs two programs that should agree on the same inputs and reports where they part.

Nothing published puts these together for a program asked for by name: read in depth, watch the original where
it runs, design it as libraries, build each with tests, and keep the libraries for the next program.

## The design

1. **Read in depth** (`what_is_known_of_it`). From the program's article to its kind's, then down: each feature
   to the articles of what it rests on (a flood fill, Bresenham's line, PNG, an undo stack, a formula's grammar),
   found by title in her corpus and by passage across its hundred million paragraphs; online, the program's own
   manual and help pages. Sentences that state behaviour are marked by rule first (what happens when, the numbers,
   the limits), then her model turns only those into behaviours, each kept with its source and how sure it is, as
   her other readings are. Her model's own knowledge is a witness, the least sure one.
2. **Watch the original where it runs** (`what_it_was_seen_to_do`). A web program in her browser, a command-line
   tool or library on this machine: explored as people explore an interface, state by state, every response
   recorded; for code, called on inputs drawn from what was read. What it did becomes checks with the original as
   their oracle. Its code is never read: clean-room is about the code, not about using the program.
3. **Design it as libraries** (`how_it_is_made`). The program is split into libraries with stated interfaces
   (its data, how it is drawn, what input does, its files, its algorithms), each with tests from the behaviours
   and observations above, before any code. A library she already has, and that passes the tests, is used as it
   is.
4. **Build each library from its tests** (`building_a_library`). In dependency order; several candidates per
   function, picked by the tests; what failed shown back, a few times; a function that cannot be made to pass is
   reported, not faked. Then the program's own checks, as now, and her use of it, as now.
5. **Keep what she built** (`her_libraries`). Every library that passed is kept with its interface, its tests
   and what it was for; found again by what a new program needs; and where two of hers share code, the shared
   part is drawn out into a library of its own, kept only if every test of both still holds.
6. **Ours set aside.** The parts we wrote are taken out of her path and kept only as a comparison arm: the same
   program built with them and without.

## How it is judged

A development set of programs to build from, and a held-out set drawn by seed from a list fixed before any of
this is built, never looked at while building. For each program: the share of its behaviours whose checks hold;
where the original runs here, the share of observed behaviours her build agrees with; the time; how much of the
code came from her libraries, her model and nothing of ours. The same numbers with our parts, as the comparison.

## Order

`what_is_known_of_it` first, since everything rests on what is read; then the library build and her library;
then watching the original; then the evaluation sets. Each is proven on the development set before the next.

Sources: [Commit0](https://arxiv.org/html/2412.01769v1),
[WebGen-Bench](https://neurips.cc/virtual/2025/poster/121451), [ArtifactsBench](https://arxiv.org/html/2507.04952v2),
[specifications from documents](https://arxiv.org/html/2504.01294v1),
[LILO](https://proceedings.iclr.cc/paper_files/paper/2024/hash/819cebb05f993840e8a52d7564c5c282-Abstract-Conference.html),
[ReGAL](https://arxiv.org/html/2401.16467v2), [Leroy](https://arxiv.org/pdf/2410.06438).
