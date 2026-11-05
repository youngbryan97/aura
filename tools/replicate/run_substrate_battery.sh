#!/bin/zsh
# Replicate seed 101's substrate campaign on an Apple Silicon Mac, and compare it with the reference runs.
#
#   zsh tools/replicate/run_substrate_battery.sh /path/to/conversation_tape.json
#   zsh tools/replicate/run_substrate_battery.sh status
#
# Read tools/replicate/README.md first. It runs about nine hours; leave the Mac plugged in with the lid open.
set -eu
KIT=${0:A:h}
REPO=${KIT:h:h}
COMMIT=aeefdca31
HOME_DIR=$HOME/aura-replication-kit
CODE=$HOME_DIR/code
RUNS=$HOME_DIR/runs
export PATH=/opt/homebrew/bin:$PATH
say() { print -r -- "[$(date '+%H:%M:%S')] $*"; }

if [[ "${1:-}" == "status" ]]; then
  for f in smoke campaign; do
    [[ -f $RUNS/$f.log ]] || continue
    say "$f.log, last stage lines:"; grep -E "^\[[0-9:]+\]" $RUNS/$f.log | tail -4
  done
  [[ -f $RUNS/peak_memory.txt ]] && say "peak memory so far: $(cat $RUNS/peak_memory.txt)"
  exit 0
fi

TAPE=${1:-}
[[ -n "$TAPE" && -s "$TAPE" ]] || { say "give the conversation tape's path as the first argument (see the README)"; exit 1; }
TAPE=${TAPE:A}
[[ "$(uname -m)" == "arm64" ]] || { say "refusing: this needs an Apple Silicon Mac"; exit 1; }
(( $(sw_vers -productVersion | cut -d. -f1) >= 14 )) || { say "refusing: this needs macOS 14 or later"; exit 1; }
(( $(sysctl -n hw.memsize) >= 15 * 1024 * 1024 * 1024 )) || { say "refusing: this needs 16 GB of memory"; exit 1; }
(( $(df -g "$HOME" | awk 'NR==2 {print $4}') >= 40 )) || { say "refusing: this needs 40 GB of free disk"; exit 1; }
PY=$(command -v python3.12 || true)
[[ -n "$PY" ]] || { say "refusing: python3.12 not found. Install it with:  brew install python@3.12"; exit 1; }

mkdir -p $HOME_DIR $RUNS/logs
if [[ ! -d $CODE ]]; then
  say "checking out the reference commit $COMMIT"
  git -C $REPO fetch --quiet origin
  git -C $REPO worktree add --quiet --detach $CODE $COMMIT
fi
[[ "$(git -C $CODE rev-parse --short=9 HEAD)" == "$COMMIT" ]] || { say "refusing: $CODE is not at $COMMIT"; exit 1; }
[[ -z "$(git -C $CODE status --porcelain --untracked-files=no)" ]] || { say "refusing: $CODE has local changes"; exit 1; }
cd $CODE

if [[ ! -x .venv/bin/python ]]; then
  say "making the Python environment (a while: the dependencies are pinned by hash)"
  $PY -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  # The environment the reference runs used. The lock's pins for diffusers conflict with its own huggingface-hub and
  # safetensors, so it is installed without resolving dependencies; PyTorch is not in the lock and is added; and the
  # first machine ran scikit-learn 1.8.0, where the lock says 1.9.0.
  .venv/bin/pip install --quiet --require-hashes --no-deps -r requirements_lock.txt
  .venv/bin/pip install --quiet torch==2.11.0
  .venv/bin/pip install --quiet --no-deps scikit-learn==1.8.0
fi
digest=$(.venv/bin/python -c "import sys; from pathlib import Path; from core.subject.conversation_tape import ConversationTape; print(ConversationTape.load(Path(sys.argv[1])).digest)" "$TAPE")
want=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['tape_digest'])" $KIT/reference/seed101.json)
[[ "$digest" == "$want" ]] || { say "refusing: this tape is not the reference tape (digest ${digest[1,12]}, want ${want[1,12]})"; exit 1; }
say "the tape is the reference tape (${digest[1,12]})"

export AURA_LOG_DIR=$RUNS/logs
export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
COMMON=(--turns 3 --seed 101 --null-workers 4 --calibrated-axes --one-clock --conversation-tape $TAPE)

# Run a step, recording its peak memory and stopping it if memory stays under 5% free for five minutes.
guarded() {
  local log=$1; shift
  caffeinate -dims "$@" > $log 2>&1 &
  local pid=$! low=0 peak=0 now=0 freepct=100
  while kill -0 $pid 2>/dev/null; do
    now=$(ps -A -o rss=,command= | awk 'index($0, "tools/run_subject_core.py") {s += $1} END {printf "%d", s / 1024}')
    (( now > peak )) && peak=$now && print -r -- "${peak} MB" > $RUNS/peak_memory.txt
    freepct=$(memory_pressure -Q 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%", "", $2); print int($2)}')
    if [[ -n "$freepct" ]] && (( freepct < 5 )); then low=$((low + 1)); else low=0; fi
    if (( low >= 10 )); then
      say "stopping: under 5% of memory free for five minutes (peak ${peak} MB). Send $log and $RUNS/peak_memory.txt"
      pkill -TERM -P $pid 2>/dev/null; kill -TERM $pid 2>/dev/null; wait $pid 2>/dev/null; return 2
    fi
    sleep 30
  done
  wait $pid
}

if [[ ! -e $RUNS/smoke-passed ]]; then
  say "a short wiring run first (about 30 to 75 minutes)"
  guarded $RUNS/smoke.log .venv/bin/python tools/run_subject_core.py --quick $COMMON --out $RUNS/smoke \
    || { say "the wiring run failed; send $RUNS/smoke.log"; exit 1; }
  touch $RUNS/smoke-passed
  say "wiring run passed"
fi
say "seed 101's campaign, as on the reference machines (about nine hours; progress: zsh $0 status)"
guarded $RUNS/campaign.log nice -n 5 .venv/bin/python tools/run_subject_core.py --rounds 370 --trials 6 $COMMON \
  --out $RUNS/campaign || { say "the campaign failed; send $RUNS/campaign.log"; exit 1; }
run=$(ls -td $RUNS/campaign/run_*(N) | head -1)
python3 $KIT/compare_with_reference.py $run --out $RUNS
say "done (peak memory $(cat $RUNS/peak_memory.txt)). Send $RUNS/replication_report.md, or paste it into a"
say "replication issue on GitHub. $RUNS/replication_report.json has the same in full."
