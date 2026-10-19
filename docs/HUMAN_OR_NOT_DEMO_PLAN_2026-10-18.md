# Human or Not: a demo plan, 18 October 2026

Bryan's ask: Aura plays Human or Not, a public game in which strangers chat for two minutes and each guesses whether
the other is a person or an AI. Measure how often she is taken for an AI against how often bots, a top model,
people and chance are; aim at the rate people get. No fine-tuning to the game and nothing written for the site: she
uses the site, reads what the game is, and plays it, through abilities that work on any site.

What it is for, in his words: not her telling people "I'm a human", but that if people talking with her in real time
take her for one of their own, without tricks, prompts or a persona, that is evidence for a conversation about
what such a system is owed. So she plays as she is, and knows the game is a game: she takes a role in it the way
a person playing it does.

## The numbers to stand against

- People, in the original game (AI21, 2023, over 1.5 million players): a human partner taken for a human 73% of the
  time, a bot caught 60% of the time, 68% right overall. So a person is taken for an AI 27% of the time, a 2023 bot
  passes 40% of the time.
- A top model, three-party test (Jones and Bergen, 2025): GPT-4.5 with a written persona was picked as the human
  73% of the time; LLaMa-3.1-405B with it 56%; GPT-4.5 without it 36%; GPT-4o 21%; ELIZA 23%.
- Chance: 50%.

## The site

humanornot.ai now redirects to ai21.com; the game is played at humanornot.so (operator unnamed, terms under Israeli
law, licence for personal non-commercial use, nothing said of automated players). Two-minute chats; its bots run
GPT-5, Claude and Gemini Flash. From its own code: when the partner is a person, that person's verdict on you is
shown at the end (`CHAT_GAME_VOTE_BY_PARTNER`), so how often she is caught is measurable on those games; every
message costs a coin, with 50 free and coins earned by guessing right (no coins are bought without Bryan's word).

## What she needs that she does not have

1. **A conversation on a page.** Seeing that a page holds a live exchange with someone (two parties' messages, a
   box to write in, new messages arriving), keeping its turns, noticing a new message from the other side, and
   seeing it end. From the page's document where it has one, from the picture where it does not. Any chat site,
   support chat or game, not this one.
2. **A reply in the time a conversation keeps.** Her chat, as it is, answers in tens of seconds with her thinking
   open. A turn with a deadline gets a decode that fits it: the answer clock her inference gate already keeps
   (`[ANSWER CLOCK]`) sets the budget, her private channel closed when it would not finish, the reply her model's own.
   No delay is added to look human, and none is taken away to look fast: she answers at her pace.
3. **The game's point, read.** The page's own rules, read once by her model, set what she is doing: talking with
   someone for two minutes, then saying whether she thinks they are a person. That she is an AI asked to play is
   hers to hold; how she plays it is hers to decide.
4. **The end of each game, kept.** The partner's kind as revealed, her guess, and, where the partner was a person,
   their guess about her, written to a record per game.

## The run

At most about two hours, ideally under one. Each game takes about three minutes with matching, so 20 to 40 games;
only games with a person as partner give a verdict on her, perhaps half. That is enough to tell 40% (a 2023 bot)
from 73% (people) and not to tell 60% from 73%: the result is reported with its interval.

## Order

After the night's work on her model (L09, L01). First the conversation ability and the deadline-sized decode, each
proven on a local chat page Aura has never seen, then a single game on the site watched start to end, then the run.

## Built, 19 October, before any game

- **A conversation on a page** (`core/perception/a_conversation_on_a_page.py`): the stream is the part of the page that
  grows, her lines are what she sent, the place to write is not a search box or one field of a longer form, and the
  conversation is over when the place to write is gone.
- **Held to its end** (`core/skills/talking_on_a_page.py`): each new line from the other side answered, written and
  sent (Return, else the control beside the box); after six quiet seconds she may open it.
- **Her line, in the time a conversation keeps** (`core/skills/her_reply_in_a_conversation.py`,
  `core/brain/llm/an_answer_in_the_time_given.py`): her model, given the conversation as turns and what she was asked
  and what the page says as evidence, with `answer_within_s`; the gate's answer clock fits the decode to that time.
- **Chosen, not run** (`core/skills/sovereign_browser_conversation.py`): her browser pursuit offers the page's
  conversation as one element, as it offers what a page draws; she chooses it or not, and choosing it holds the
  conversation.
- **Kept** (`data/conversations_held_on_pages.jsonl`): each conversation, with the page's words at its end and on the
  next four pages on the same site, where who the other side was and what they made of her are said.

Proven on a local page Aura has never seen (`tests/fixtures/conversations/`), with a stand-in for her model. Not yet
run with her model, and not yet on the site.
