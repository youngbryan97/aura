# Writing rules

<!-- reader-navigation:start -->
> **In plain English:** The documentation contains current guides, dated experiment records, frozen rules, generated reports and open plans. They answer different questions.
>
> **Why it matters:** A reader should not treat an old proposed run as today’s result or a completed engineering task as a passed scientific check.
>
> **In everyday use:** An entry saying the next run is queued can remain correct history after that run finishes; the maintained status page gives the later outcome.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#documentation) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#documentation) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#measurement) · [Document map](DOCUMENTATION_INDEX.md)
> [Glossary](GLOSSARY.md) · [Formulas with worked examples](FORMULAS.md)
<!-- reader-navigation:end -->

Status: Guide · Standing conventions for prose in this repository

Everyone learned the em-dash trick, so the tells moved. These are the patterns
that now read as machine-generated, and the repository is checked against them:

```bash
make writing            # the front-facing docs, ratcheted
python tools/lint_ai_writing.py --all      # every guide
python tools/lint_ai_writing.py --code     # docstrings and comments
python tools/lint_ai_writing.py FILE       # one file
```

Docstrings and comments are checked from the same rulebook. They are read more
often than the guides are, and until August 2026 nothing checked them at all.

`tools/lint_ai_writing.py` is the executable half of this page. **Adding a
pattern here without adding it there leaves the rule unenforced**, which is how
the last set of writing conventions quietly stopped applying.

The ratchet lives in `config/ai_writing_baseline.json` and only goes down.

---

## The forbidden patterns

### 1. "That's not X. That's Y."

> That's not compliance. That's stalling.

Say the second half only: *They're stalling.* The negated first half exists to
make the second sound earned. It didn't earn it.

### 2. Two short bits stapled together

> Fast. Simple.
> No fluff. Just answers.

Pick one and write it as a sentence.

### 3. Two pictures and no advice

> Less a hammer, more a scalpel.

Say what to do instead.

### 4. Clapping for itself

> And that matters. That's the part everyone misses. Which is exactly the
> point. It's worth stating that…

Delete them all. Nothing is lost. If a sentence needs to be announced as
important, it isn't.

### 5. The analogy that assumes both referents

> It's the Excel of AI agents.

Only works if the reader knows both things. Usually they don't.

### 6. Warming up before talking

> Here's the thing. Let me be clear. The truth is.

Start one sentence later.

### 7. Always three things

> Faster, cheaper, smarter.

Real reasons come in twos and fives. A triad that keeps appearing is a rhythm
the writer fell into, not a count of anything.

### 8. Ranges standing in for measurements

> 5 to 10 minutes.

If you ran it, say what it took.

**Exception, and it matters here.** A range that *is* the measurement stays.
`10–50 ms per φ evaluation` and `30–60 seconds while Metal compiles shaders`
describe real distributions across inputs and machines. Collapsing them to one
invented number would be worse writing and factually wrong. The rule targets
estimates nobody made, not variance somebody measured. The linter flags both;
reviewed-and-kept ranges sit in the baseline.

### 9. The ending that repeats the whole thing

> In short… At the end of the day…

Stop typing.

### 10. "Not just X — it's Y"

> The "I" is not just a data structure, but a lived reality.

Rule 1 with a comma instead of a full stop, and the form every guide to
spotting machine prose names first. `not only X but also Y` and
`not X, but rather Y` are the same move. Say the second half.

### 11. The participle that restates the sentence

> The system retries twice, ensuring reliability.
> Latency fell by half, underscoring the win.

The clause after the comma adds nothing the clause before it did not already
say. End the sentence. If the consequence is worth stating, give it its own.

### 12. Hedging before the fact

> It is important to note that the gate is advisory.

Say *the gate is advisory*. A fact introduced as important arrives weaker
than a fact stated.

### 13. Who says so

> Studies show naive repetition is unstable.
> Many argue that recurrence damages reasoning.

Name the source or cut the sentence. This repo registers claims against the
tests that validate them, so an unsourced claim here is two defects: prose
nobody wrote and a claim nobody can check.

### 14. Asking yourself a question

> The result? Nothing rendered at all.
> Why does this matter?

Answer it without asking it first.

### 15. Nobody is here with you

> Let's dive into the scheduler.

Say what the scheduler does.

### 16. The stock opening

> In today's fast-paced world…
> In the ever-evolving landscape of…

Start at the first sentence that carries information.

### 17. The long word where the short one was exact

> We utilize the gateway in order to write state prior to the flush.

*We use the gateway to write state before the flush.* Latinate inflation is
the oldest tell there is, and it survives every round of models learning not
to use em dashes.

### 18. A chat reply pasted in unread

> As a large language model, I cannot verify that.
> Hope this helps! Let me know if you would like more.

Delete the line. Nothing else in this file matters if these ship.

---

## The four principles underneath

From William Zinsser, *On Writing Well*:

1. **Simplicity**
2. **Brevity**
3. **Clarity**
4. **Humanity**

ASD-STE100 — the controlled-English standard used for aircraft maintenance
manuals, where an ambiguous sentence can kill somebody — delivers the first
three and drops the fourth. Ask for it by name when writing procedures, gates,
runbooks, and API documentation. Do not ask for it when the writing has a
voice: it flattens anything with a heart into a parts catalogue.

## Give a newcomer a way in

Research and engineering pages need an explanation a person can use before
learning the project vocabulary. After the opening, a reader should be able to
say “so essentially…” and explain the mechanism and its purpose to someone
else. If that paraphrase still needs the unexplained technical terms, rewrite
the introduction. State what the mechanism does, why someone
would care, and what it changes in an ordinary interaction. Define a variable
by its meaning, unit or scale, consumer and practical consequence.

Use a worked example when arithmetic helps. Use a metaphor when it makes an
unfamiliar mechanism easier to picture, and state where the comparison stops.
An illustration is labeled as an illustration; it is not an extra experiment.

Keep the technical detail available after that introduction. A reader should
be able to explain the idea aloud without first translating a dense paragraph.
The shared [tour](PLAIN_ENGLISH_GUIDE.md) and
[variable definitions](VARIABLES_EXPLAINED.md) support the page-specific reading
notes; their source is `config/documentation_reading_topics.json`.

Connect a claim to a witnessed demo, controlled result or source check where
one exists. Label a planned demonstration as a plan. Cite primary research for
related ideas. An extension in implementation scope does not establish a
performance advantage without a comparable test. The
[demos and related-work guide](DEMOS_AND_RELATED_WORK.md) records those limits.

Cite the source beside the statement it supports, including scientific design
inspirations, empirical findings, borrowed formulas and real-world incidents.
Prefer the original study, authors' paper or official technical reference.
Name the population or system studied. A result about people does not supply
an Aura result. Identify a theoretical proposal as a proposal, an adaptation
as an adaptation, and an invented example as an illustration. If only a
bibliographic record or abstract was checked, say so before drawing a more
detailed conclusion. Do not add references merely to make a paragraph look
scientific.

For a quantitative Aura claim, retain the run, source revision, measurement,
controls and uncertainty beside the reference. Preserve failed results and
later method corrections. Use the [scientific source map](SCIENCE_AND_SOURCES.md)
and [reusable bibliography](../research/references.bib) when preparing a paper;
check the supporting passage, version and correction status again before
submission. This extends the source-attribution convention above; it adds no
automatic claim of scientific validity.

`python tools/render_reader_navigation.py --check` verifies the generated
reading notes and document map. It checks coverage and consistency; judging
whether the explanation is clear and the comparison is warranted remains
editorial work. Frozen protocols, historical evidence and compliance text
retain their original wording and are indexed separately.

## What this does not apply to

**Append-only records are never edited for style.** The execution tracker, the
RLC ledger, `docs/evidence/`, dated verdicts, and benchmark results say what
was true when they were written. Editing a July entry in August is falsifying
a record, not improving prose. The linter excludes them by path; see
[DOC_STATUS.md](DOC_STATUS.md) for which documents are which.

Quoted material keeps its original wording, including when the original
commits one of the eighteen. The linter blanks backticked spans and quoted
spans before the rules run, so a docstring that names the phrase it detects
is not read as having written it.

**The scope disclaimer stays.** Rule 1 targets a negation that exists to make
the second half sound earned. A negation that rules out a reading somebody
would otherwise make is doing work:

> This is not a configuration file. It is Aura's foundational identity.
> That is not a tuning delta. It is the same model, the same checkpoint.

Both name a category the reader was about to assign and take it away. Deleting
the first half loses the correction. Eleven of these sit in the code baseline;
they were read one at a time before they went there.

## Where the rules came from

The 2026 crop of "how to spot AI writing" guides agrees on more than it
disagrees on, and rules 10 through 18 are the overlap: negative parallelism,
the trailing participle, the disclaimer, the unsourced claim, the rhetorical
question, the false "let's", the stock opening, Latinate inflation.

One dissent shaped the design more than the agreement did. Writing in defense
of the patterns everyone bans, [Robots Ate My
Homework](https://robotsatemyhomework.substack.com/p/ai-writing-patterns)
argues the tricolon and the binary contrast are ordinary rhetoric, and that
"the pattern is fine; autopilot is the problem." That is why the triad rule
fires only on evaluative words. This codebase enumerates three real states
constantly — `warming, recovering, handshaking` — and a linter that called
those a tell would be ignored inside a week, correctly.

## Extending this

Add a pattern the moment you catch one in a draft. Three edits:

1. A section here, with the example that caught it.
2. A rule in `RULES` in `tools/lint_ai_writing.py`, with a regex and the
   one-line fix shown to whoever trips it.
3. A line in `POSITIVES` in `tests/test_ai_writing_rules.py`. That suite
   fails if any rule has no worked example, because a rule that cannot match
   reports green forever. Rule 7 shipped that way and nobody noticed until
   somebody counted the rules on each side.

Then run `python tools/lint_ai_writing.py --all --baseline` to record where the
tree sits. The file gets better every time somebody notices something.
