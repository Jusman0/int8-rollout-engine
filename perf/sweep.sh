#!/usr/bin/env bash
# Cycle-count sweep (DESIGN.md §7.5). Runs the bit-exact testbench for
# LANES = 32, 16, 8 and H = 15, 5 on 200 decisions each, and records the
# clocks per decision and whether all 200 decisions matched the golden model.
#
# Run from the repo root:   PYTHON=python3 bash perf/sweep.sh
# Writes:
#   results/sweep.log    summary table, then every command's full output
#   results/cycles.log   perf/cycles.py: formula vs measured, multiplier use
# Scratch (sim/ is gitignored):
#   sim/perf/h15, sim/perf/h5   decision vectors and compiled testbenches
#
# Each H needs its own decision vectors: the plans are H actions long, and
# the closed-loop episode that produces them plays out differently. They come
# from the golden model's own command line, 200 decisions each (--count 200 =
# the first episode, seed 0). The H = 15 ones are the same 200 decisions as
# the first 200 in sim/, the vectors "make test" uses.
# The testbench is built and run with the normal Makefile, so this sweep
# tests exactly what "make test" tests, only with other LANES/VEC/N.
#
# golden.py rewrites rom/ every time it runs. The contents never change (the
# weights depend only on model/dynamics_fp32.npz, not on H), and this script
# stops if they ever do.

set -e
cd "$(dirname "$0")/.."            # make and golden.py expect the repo root
PYTHON=${PYTHON:-python3}
N=200
LOG=results/sweep.log
RAW=sim/perf/raw.log               # every command's output; goes under the table

mkdir -p sim/perf results
: > "$RAW"                         # start the raw log empty
t_start=$SECONDS

# ------------------------------------------------------------ 1. vectors
rom_before=$(cat rom/* | cksum)
for H in 15 5; do
  echo "################################################################" >> "$RAW"
  echo "# vectors: \$PYTHON model/golden.py --H $H --count $N --out sim/perf/h$H" >> "$RAW"
  "$PYTHON" model/golden.py --H $H --count $N --out sim/perf/h$H >> "$RAW" 2>&1 ||
    { echo "golden.py failed, its output is in $RAW" >&2; exit 1; }
done
if [ "$(cat rom/* | cksum)" != "$rom_before" ]; then
  echo "rom/ changed when golden.py ran, so this sweep would not test the committed weights" >&2
  exit 1
fi

# ------------------------------------------------------------ 2. runs
# A failing run must not stop the sweep (set -e would), so make's exit code
# is ignored with "|| true"; the testbench's own PASS/FAIL line decides.
rows=""
fails=0
for H in 15 5; do
  for LANES in 32 16 8; do
    VEC=sim/perf/h$H
    run_log=sim/perf/run_l${LANES}_h$H.log
    t0=$SECONDS
    make test LANES=$LANES VEC=$VEC N=$N PYTHON="$PYTHON" > "$run_log" 2>&1 || true
    secs=$((SECONDS - t0))

    # "clocks per decision: 1281..1281 (...)" -> 1281..1281, the min..max over all decisions
    clocks=$(sed -n 's/^clocks per decision: \([0-9.]*\) .*/\1/p' "$run_log")
    # "costs: 200/200 decisions bit-exact (...)" -> 200/200, and the same for the others
    costs=$(sed -n 's/^costs: \([0-9]*\/[0-9]*\) .*/\1/p' "$run_log")
    states=$(sed -n 's/^final states: \([0-9]*\/[0-9]*\) .*/\1/p' "$run_log")
    best=$(sed -n 's/^best_idx: \([0-9]*\/[0-9]*\) .*/\1/p' "$run_log")
    if grep -qx PASS "$run_log"; then verdict=PASS; else verdict=FAIL; fails=$((fails + 1)); fi

    # one table row per run; += appends to the string
    rows+=$(printf '#  %5s %3s %9s %15s %8s %8s %10s %7s %4ss' \
                   "$LANES" "$H" "$N" "$clocks" "$costs" "$states" "$best" "$verdict" "$secs")
    rows+=$'\n'                    # $'\n' is a newline character

    {
      echo "################################################################"
      echo "# run LANES=$LANES H=$H: make test LANES=$LANES VEC=$VEC N=$N  ($secs s)"
      cat "$run_log"
    } >> "$RAW"
  done
done

# ------------------------------------------------------------ 3. the log
# What was tested: the commit, and whether rtl/ tb/ model/ rom/ or the
# Makefile had uncommitted changes when the sweep ran.
if commit=$(git rev-parse --short HEAD 2>/dev/null); then
  dirty=$(git status --porcelain -- rtl tb model rom Makefile | tr '\n' ' ')
  source_state="commit $commit, rtl/tb/model/rom/Makefile: ${dirty:-no uncommitted changes}"
else
  source_state="not a git checkout"
fi
{
  echo "# Cycle sweep (DESIGN.md §7.5): command  PYTHON=python3 bash perf/sweep.sh"
  echo "# $(date '+%Y-%m-%d %H:%M:%S'), $source_state"
  echo "# $(iverilog -V 2>&1 | head -1); $("$PYTHON" --version 2>&1)"
  echo "# $N decisions per run (episode 0, seed 0). clocks per decision = the engine's cycles"
  echo "# output (start seen -> done raised), min..max over the run; the testbench checks it"
  echo "# against its own count. bit-exact = decisions where all 32 costs / all 32 final"
  echo "# (theta, omega) / best_idx and best_cost equal the golden model's."
  echo "#"
  echo "#  LANES   H decisions    clocks/decis.    costs   states   best_idx verdict wall"
  printf '%s' "$rows"
  echo "#"
  echo "# formula vs measured and multiplier use: results/cycles.log (perf/cycles.py)"
  echo "# total sweep time: $((SECONDS - t_start)) s"
  echo
  cat "$RAW"
} > "$LOG"

# ------------------------------------------------------------ 4. formula
"$PYTHON" perf/cycles.py "$LOG" | tee results/cycles.log
cycles_ok=${PIPESTATUS[0]}       # cycles.py's exit code, not tee's

echo
sed -n '1,/^# total sweep time/p' "$LOG"
if [ "$fails" -ne 0 ] || [ "$cycles_ok" -ne 0 ]; then
  echo "SWEEP FAILED: $fails testbench run(s) failed, cycles.py exit code $cycles_ok"
  exit 1
fi
echo "SWEEP PASSED"
