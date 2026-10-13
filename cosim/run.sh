#!/usr/bin/env bash
# Closed-loop co-simulation (DESIGN.md §7.3). Python (cosim/host.py) runs
# the pendulum and the host side of the planner, Icarus Verilog
# (cosim/tb_cosim.v) runs the RTL, and the RTL's best_idx picks every action
# applied to the pendulum. Every episode return must equal the golden
# model's exactly.
#
#   PYTHON=python3 bash cosim/run.sh               20 episodes, seed 0, H = 15
#   EPISODES=1 PYTHON=python3 bash cosim/run.sh    one episode, a quick check
#   LANES=8 PYTHON=python3 bash cosim/run.sh       8 lanes (4 passes, slower)
#
# Works from any directory. Builds into sim/cosim/ (not committed) and saves
# the report to sim/cosim/cosim.log. Exit status 0 means PASS.
set -euo pipefail

cd "$(dirname "$0")/.."            # the repo root: rom/ and rtl/ are found from here
PYTHON=${PYTHON:-python3}
EPISODES=${EPISODES:-20}
LANES=${LANES:-32}
DIR=sim/cosim
LOG=$DIR/cosim.log

mkdir -p "$DIR"

# 1. compile the testbench around the engine
iverilog -g2005 -Wall -I. -DCOSIM_DIR="\"$DIR\"" -Ptb_cosim.LANES="$LANES" \
    -o "$DIR/tb_cosim" rtl/rollout_lane.v rtl/rollout_engine.v cosim/tb_cosim.v

# 2. two named pipes, one per direction. cleanup runs however this script
#    ends (PASS, FAIL or Ctrl-C): it stops the simulator if it is still
#    running and deletes the pipes.
vvp_pid=""
cleanup() {
    if [ -n "$vvp_pid" ]; then kill "$vvp_pid" 2>/dev/null || true; fi
    rm -f "$DIR/to_rtl" "$DIR/from_rtl"
}
trap cleanup EXIT
rm -f "$DIR/to_rtl" "$DIR/from_rtl"
mkfifo "$DIR/to_rtl" "$DIR/from_rtl"

# 3. start the simulator in the background. It waits inside $fopen until
#    the host opens the pipes.
vvp -n "$DIR/tb_cosim" > "$DIR/vvp.log" &
vvp_pid=$!

# 4. run the host. -u: print each episode's line as soon as it is done,
#    even though the output goes through tee.
status=0
if "$PYTHON" -u cosim/host.py --episodes "$EPISODES" --pipes "$DIR" | tee "$LOG"; then
    # the host sent "0": the testbench prints its summary and ends by itself
    wait "$vvp_pid" || status=1
else
    status=1
    # the simulator may be stuck waiting for a host that is gone
    kill "$vvp_pid" 2>/dev/null || true
    wait "$vvp_pid" 2>/dev/null || true
fi
vvp_pid=""

echo "--- simulator output (vvp) ---" | tee -a "$LOG"
tee -a "$LOG" < "$DIR/vvp.log"
exit "$status"
