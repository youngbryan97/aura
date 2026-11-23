# F16: what her own goals show about the change verifier (22 November 2026)

An outside review (ChatGPT, passed on by Bryan on 22 November) asked whether the stricter verifier lowers how often her
repairs land. Her goal records answer it better than a constructed set could: `~/.aura/data/her_goals/`, 66 goals.

| Goals | Count |
| --- | ---: |
| Working | 47 |
| Stalled | 12 |
| Merged | 6 |
| Offered | 1 |

Only four of her drafts ever reached the outside check: one was held and merged (`1d03d4c00`), three were refused. Five
of the twelve stalls came before any draft: the test she wrote to make the fault again did not raise what she had
raised while running. The verifier is not where most of her repairs stop. Reproducing the fault is.

Of the three refusals, one was the verifier's fault. Her change `429b8199c` to `core/runtime/collector_pauses.py` was
refused because `freeze_what_boot_made()` returned 16,172 where the code before returned 16,069. That function counts
every object her process holds (`gc.get_freeze_count()`), so any change to any file moves it, and the check then
handed her a test asserting 16,069 that no later draft could pass. The second run of the code before now holds 4,096
unrelated objects, and a value that moves with them measures the process and is not compared. `429b8199c` now lands on
probation, with the function named as untried. `tests/test_a_value_that_counts_the_process_is_not_compared.py` fails
on the verifier before this change. The other two refusals stand: one changed what three tests gave, and one ran past
its time on a test suite that has since been narrowed to the one the repository declares (`b069b74d9`).

## The rest of the review, checked against the code

- Her F16 repair search calls `mend(..., model=None)`. The review called that a limitation. It is Bryan's choice of 16
  November: the repair is engineered, and a model proposes nothing in it unless he asks. Unchanged.
- The outside check runs once a draft is chosen, not on every candidate. True. Its lost inputs come back as tests her
  next search must keep, on the next pass of her loop.
- `core/skills/coding_skill.py` gave back a Python draft that failed the examples it was given, with no second try.
  It now mends the draft against those examples with the same rounds an outside repair gets
  (`core/self_modification/repairing_python.py`, `repaired_source`), and gives back the mended code when it passes.
- A draft returned with no examples to check it still reports `ok`, with `verification` saying it was not measured.
  That is unchanged: `ok` says a draft came back.
