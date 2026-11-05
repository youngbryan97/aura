# Replicate the substrate battery on your Mac

This kit runs seed 101's substrate campaign on your machine and compares the result with two reference runs, line by
line. The campaign decides 22 of the 24 lines of the battery. Her language model is stubbed in it, so no model weights
are needed and 16 GB of memory is enough.

## What you need

- An Apple Silicon Mac (M1 or later) with macOS 14 or later.
- 16 GB of memory and 40 GB of free disk.
- Python 3.12. To install it: `brew install python@3.12`.
- The conversation tape: `conversation_tape.json`. See "The tape" below.
- About nine hours of machine time and fifteen minutes of yours. Keep the Mac plugged in, with the lid open.

## Procedure

1. Clone the repository: `git clone https://github.com/youngbryan97/aura.git`
2. Quit heavy applications.
3. Run: `zsh aura/tools/replicate/run_substrate_battery.sh /path/to/conversation_tape.json`
4. To see progress, in a second Terminal window: `zsh aura/tools/replicate/run_substrate_battery.sh status`
5. When it says "done", send `~/aura-replication-kit/runs/replication_report.md`, or open a replication issue on
   GitHub and paste it in.

The script checks your machine, checks out the reference commit (`aeefdca31`) in `~/aura-replication-kit/code`, makes
the Python environment the reference runs used, and confirms the tape is the reference tape. Then it runs a short
wiring check and the campaign, and writes the comparison. If memory stays under 5% free for five minutes, it stops the
run and says which files to send.

## What "replicated" means

The same verdict, pass or fail, on each of the 22 lines. The numbers are not expected to match exactly: her runtime
depends on timing, so the same seed gives a similar run on another machine, not an identical one. The report lists
each line's verdict here, on the first machine, and on a 16 GB MacBook.

Two lines need a note, written before any outside run:

- **synergy** is borderline. Its weakest triple (perception and memory onto the world model) passed its one-direction
  test at p = 0.0052 on the first machine, against a bar of 0.01, and read p = 0.60 on the MacBook. A flip on this
  line is the result already seen once. A flip on any other line is new.
- **causal_closure_of_the_core** fails in the campaign on both reference runs. It is the campaign's observational
  reading. The 24 of 24 decides closure by intervention, in the carrier sweep, which this kit does not run.

## What this kit does not test

The two lines the carrier sweep decides, the reports runs (they need her 27-billion-parameter model and 64 GB), the
day runs, and self-prediction. A replication here shows the 22 campaign lines do not depend on one machine. It is not
the whole result.

## The tape

The campaign replays a recorded conversation: the 1,025 things her owner said to her, in order. It holds his own
conversations, so it is not in this repository. Ask for it in a replication issue. The script refuses any tape whose
digest is not the reference digest (`1e2046ea87a5...`), so you will know you have the same one.

## The reference runs

`reference/seed101.json` holds both reference runs' verdicts, environments and the tape's digest. To compare a run of
your own by hand: `python3 tools/replicate/compare_with_reference.py <run directory>`
