# G05 closed-channel pilot: where a settled answer reaches a person

Two pilots ([compositions](G05_PUBLIC_ANSWER_PILOT_2026-10-06.md),
[long lists](G05_LONG_LIST_PILOT_2026-10-06.md)) found her free answers exact
with or without the reader's evidence. Both decoded with her private channel
open. That is not the path the reader's evidence takes to a person.

Her runtime closes the private channel at the render stage when an upstream
phase has settled the answer (`chat_format.thinking_enabled_for_generation`):
the reader settles the program, and she renders it without private
reasoning. This pilot measured that path, on consumed requests only.

## Arms

`tools/run_g05_public_answers.py --thinking closed` (`e834f9115` onward),
greedy, through her own template: ordinary (the request alone), assisted (the
request, then the reader's program and values as runtime evidence), sham
(another request's reading in the same form). Reader: G03's v12.

## Result

| Requests | Arm | Exact | Median tokens | Median seconds |
| --- | --- | ---: | ---: | ---: |
| compositions, bundle v1 (48) | ordinary | 31 | 808 | 73.3 |
| | assisted | 48 | 7 | 1.8 |
| | sham (first 24) | 19 of 24 | 738 | 86.3 |
| compositions, bundle v2 (48) | ordinary | 30 | 759 | 45.1 |
| | assisted | 48 | 7 | 1.5 |
| lists of 40 to 64 entries (24) | ordinary | 21 | 434 | 29.2 |
| | assisted | 24 | 2 | 1.9 |
| | sham | 9 | 2 | 2.1 |

No decode stopped at the budget. The reader's answer was right on all 120.
On compositions, 35 requests were exact only with the reading and none only
without it. The ordinary misses are arithmetic done in the open: a digit
misread from the request, then carries lost in long multiplication
(2,104,802,751,446 for 2,104,802,751,450; 868,376,329,316 for
868,375,819,316).

With the channel closed she copies the evidence she is given. A wrong list
reading was answered in two seconds and wrong 15 times in 24; on compositions
the sham cost less (19 of 24 exact), because she worked those out instead.
With the channel open, seventeen shams across the earlier pilots never misled
her.

## What follows

* The correctness gap G05 needs exists on the path her runtime serves: a
  render with the private channel closed. The confirmatory plan
  (`tools/g05_public_answer_protocol.py`) makes its primary claim there, on
  fresh compositions, and plans its size from this pilot's conservative
  bounds (discordance 0.269, assisted share 0.918 over 96 pairs): 44
  requests.
* The lists stratum cannot carry a primary test (three disagreements in 24)
  and serves the sham comparison instead, where its pilot is decisive.
* The ordinary arm with the channel closed is the control for the
  configuration, not her live path for a request nothing upstream settled.
  That path, channel open, is the plan's fourth arm.

## What this does not show

* Anything confirmatory: every request here is consumed.
* That a wrong reading is harmless: on lists it is copied.
