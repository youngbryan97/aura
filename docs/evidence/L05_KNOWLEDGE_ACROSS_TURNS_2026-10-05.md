# L05: one turn reaches her model, the offline copy, the web and what she kept

L05 asks for unified access to model knowledge, offline Wikipedia, the web and
retained memory, with provenance, freshness and uncertainty, across turns. Every
turn below was typed into her chat in the desktop app. The lines quoted from
the log are the `🌍 What the world says for this turn` record that each turn
writes (`~/.aura/logs/aura_json.log`, times in UTC).

## Each source, live

| When (UTC) | Asked | What the turn read | Seconds |
| --- | --- | --- | ---: |
| 3 Oct 19:23 | "Which was better, Bam's 83 point game or Kobe's 81 point game?" | offline Wikipedia: both articles (0.72, 0.73) | 0.447 |
| 3 Oct 19:26 | the Wikipedia link for Bam's game | the page sent | 0.353 |
| 4 Oct 00:53 | "Google it. Bam had an 83 point game" | web: 5 pages | 0.204 |
| 4 Oct 01:06 | "What did you learn?" | the 5 pages, "read for an earlier turn" | 0.001 |
| 4 Oct 06:47 | "What's the latest Miami Heat news this week?" | web: 5 pages | 0.219 |
| 4 Oct 06:50 | "Who designed the Kaseya Center, and what did it cost to build?" | offline Wikipedia, as of 2026-07-03 (0.689) | 0.241 |
| 4 Oct 06:59 | "How much did the Kaseya Center cost to build?" | web, looked up because the offline copy answered in part: 4 pages (0.70 to 0.78) | 3.828 |
| 4 Oct 07:11 | the same question, after a restart | kept from a web search, read 2026-10-03 23:59 (0.821), Wikipedia URL | 0.723 |
| 5 Oct 05:58 | "When did the Kaseya Center open?" | web: 5 pages, and two kept searches | 0.237 |
| 5 Oct 06:04 | "How sure are you about the FTX part, and where did that come from?" | web: 5 pages about the exchange's collapse | 0.627 |

After the fixes below, on `dea136b0f`, with the app restarted:

| When (UTC) | Asked | What the turn read | Seconds |
| --- | --- | --- | ---: |
| 5 Oct 06:48 | "Why do leaves change color in the fall?" | nothing: her own knowledge | 0.014 |
| 5 Oct 06:51 | "When did the Kaseya Center open?" | web: 5 pages; kept from a web search, read 2026-10-04 22:58; offline Wikipedia, as of 2026-07-03; this turn's own kept search not read back; "saved to memory" | 0.262 |
| 5 Oct 06:54 | "How sure are you about the FTX part, and where did that come from?" | the 7 sources the last answer was read from, "read for the answer being asked about"; a search for the claim itself, which read AP and theScore on the FTX naming | 5.263 |

The leaves answer (1,809 characters) came from her model with no source
attached. The Kaseya answer gave December 31, 1999, the Gloria Estefan concert
and the February 1998 groundbreaking, and it repeated a detail from her
earlier answer that the sources contradict: FTX bought the naming rights "in
2022" in a deal that "lasted barely two months". The page she had read says
"In March 2021, FTX acquired the naming rights". Asked how sure she was, the
cortex answered in about three minutes. She listed what the Wikipedia infobox
confirms (American Airlines Arena 1999–2021, FTX Arena 2021–2023, Miami-Dade
Arena 2023) and what she got wrong ("in 2022", "barely two months"), and she
said the $135 million had come from general knowledge she could not point to.

After the last fixes below, on `4f8a8e734`, with the app restarted:

| When (UTC) | Asked | What the turn read | Seconds |
| --- | --- | --- | ---: |
| 5 Oct 16:12 | "Which was better, Bam's 83 point game or Kobe's 81 point game?" | offline Wikipedia, as of 2026-07-03 (Kobe's game, 0.721); kept from a web search, read 2026-10-03 18:11 (Bam's game, 0.708); no web search | 2.613 |
| 5 Oct 16:16 | "Tell me about the FTX naming deal for the Heat's arena. When did it start and end?" | web: CBS, Wikipedia, ESPN and others, 6 sources in all | — |

The first answer is her view, with the facts as support (Kobe behind at the
half and fifty-five in the second; the Wizards fouling to keep the ball from
Bam in a decided game) and no "Source:" line. The second dates the deal
correctly: rights acquired in March 2021, renamed FTX Arena in June 2021,
FTX collapsed in November 2022, Miami-Dade Arena until Kaseya in 2023. It
names Wikipedia and CBS beside the facts they gave. Before it was served, the
source check logged "Read 5 dated claim(s) of her draft against 6 source(s);
none dated otherwise." The turn took 4 min 43 s.

## What was wrong, and what changed

Each was found in the turns above and fixed on main before the next round.

* **What a search kept began with the page's menus.** The research pipeline
  stored the whole converted page, so her kept note on the Kaseya Center began
  "Jump to content Main menu". Pages are now kept as their prose
  (`39b271f94`).
* **A kept search was labelled with the dump's date.** Found through the local
  corpus, it read "offline copy (web_retained), as of 2026-07-03". It now reads
  "kept from a web search, read <date and time>" and names its page
  (`117420000`).
* **Every chat turn spent 8 s on the corpus.** File paths were searched for as
  words and the any-term fallback ran to the offline deadline of 5 s. Paths are
  addresses, and the lookup runs at the conversation lane's 0.25 s; the Pong
  request went from 8.2 s to 0.375 s (`3d052627e`).
* **Kept notes crowded out the encyclopedia.** For "When did the Kaseya Center
  open?" two short kept notes took both corpus places and the article was never
  read. The two shelves are searched separately, a turn's own kept search is
  not read back as a second copy, and a search kept by the pipeline's rule
  says so (`7fbcb648a`).
* **A question about her grounds read the wrong things.** "How sure are you
  about the FTX part, and where did that come from?" searched the web for the
  question as written and read five pages about the exchange's collapse. The
  question is now read by the language substrate; the turn gets back what the
  answer was read from, and a double-check searches the claim it points at
  (`fb220a63c`).
* **The same question was taken for a complaint that she drifted.** "where did
  that come from" sat in a phrase list of drift challenges, so the cortex was
  handed a canned "I may have drifted from the thread" block, wrote a drift
  apology, the gate refused it, and the smaller model answered (`dea136b0f`).
* **A reply in other words was logged as abandoning the thread.** "$213
  million. That's the documented construction figure" shares no content word
  with "How much did the Kaseya Center cost to build?". The meaning judge is
  now asked before a reply is called abandoned (`ec28e3319`).

* **Her screen notes were read as part of the question.** The perception
  block attached to "When did the Kaseya Center open?" made the response
  contract call it a structured learning bundle (a 71,046-character prompt)
  and made the coverage check count a second question. Both now read the
  person's words (`dc2d540b9`, `ba51a8454`).
* **The first answer did not use what the sources said.** Bryan: those
  sources should be used the first time too. A dated claim in the draft is
  now read against the whole text of the turn's sources; when the source
  sentence most about the same thing dates it otherwise, that sentence joins
  the turn's sources and the draft is asked again (`da3a2d9f2`; the closest sentence must reach the judge's matched floor, `675ee2a65`; a revised reply is served as the repair retry's, `23c503aaa`).
  Against the live page and encoder, the 5 October answer gives one
  disagreement of four dated claims: the FTX clause, against "In March 2021,
  FTX acquired the naming rights" (0.795).

* **An opinion was treated as research.** "Which was better, Bam's 83 point
  game or Kobe's 81?" names two things, and with the offline articles below
  the judge's matched median it went to the web and came back with a
  "Source:" line under her view. Bryan: an opinion is a subjective view, not a
  research question, though evidence can support it. A question asking her
  view is now read by the language substrate and kept out of the research
  triggers; the offline copy is still offered (`72da323e5`).

## What this does not show

* That every claim is checked. The source check reads dated claims only; the
  "$135 million" and "barely two months" in the 4 October answer carry no
  year of their own. A figure or a duration her sources contradict goes out
  unchecked.
* That a wrong claim is always caught. On its first live run the check
  reported a true claim (a bankruptcy in November 2022 against a sentence
  about the 2021 naming deal, 0.516) and the turn ended in the canned
  apology. Both causes are fixed (`675ee2a65`, `23c503aaa`); the corrected
  check has caught the 4 October error offline against the live page and
  encoder, and has not yet met a live disagreement.
* Speed. The seconds above are the evidence step; whole turns took 88 s (the
  kept search) to 4 min 43 s.
* Model knowledge is shown on one question (the leaves, 1,809 characters, no
  source). How well she judges when her own knowledge is enough is not
  measured.
