#!/usr/bin/env bash
# run_mutants.sh: planted-bug (mutation) tests, DESIGN.md §7.4.
#
# A testbench that never fails proves nothing. So: plant one small,
# realistic bug in a COPY of rtl/, run the unchanged testbench on it, and
# check that it FAILS. Then the next bug, and so on. Each bug is a sed
# script, mutants/NN_name.sed. Its first two lines say which file it edits
# ("# file:") and what mistake it models ("# models:").
#
# For each mutant, and for the unmodified design ("00_original"):
#   1. copy rtl/*.v to sim/mutants/<name>/ and apply the sed script there
#   2. check that the file really changed. A pattern that no longer matches
#      the RTL would change nothing, and the "mutant" would PASS: that must
#      be reported as a broken mutant, not as a bug the testbench missed.
#   3. compile the copy with tb/tb_rollout_engine.v (unchanged), once per
#      vector set below
#   4. run N decisions from each set and record PASS or FAIL, and how many
#      decisions failed each of the testbench's checks
#
# The two vector sets, N decisions each:
#   first   the first N decisions in sim/, as model/golden.py recorded them
#   corner  N decisions picked by mutants/pick_corner_vectors.py: every
#           decision that clamps an h, saturates omega or has a tied best
#           cost, then the earliest others
#
# The test passes when the original design PASSES on both sets and every
# mutant FAILS on the corner set. The first set is run as well to show why
# the corner set is needed: some mutants get through it.
#
# Usage, from anywhere (the Makefile's "make mutants" runs this):
#   bash mutants/run_mutants.sh
# Settings, from the environment:
#   PYTHON=python3   a python with numpy and gymnasium (for the corner set)
#   N=300            decisions per run
#   LANES=32         lanes in the engine
#   JOBS=4           simulations run at the same time
#   KEEP=1           keep sim/mutants/ afterwards (deleted by default)
# The report goes to stdout, progress messages to stderr. Exit status 0
# means every mutant was caught.

set -u
cd "$(dirname "$0")/.." || exit 1   # repo root: rom/ and sim/ paths are relative to it

PYTHON=${PYTHON:-python3}
N=${N:-300}
LANES=${LANES:-32}
JOBS=${JOBS:-4}
KEEP=${KEEP:-0}
WORK=sim/mutants
SETS="first corner"
STARTED=$(date '+%Y-%m-%d %H:%M:%S')

if [ ! -f sim/vectors.vh ] || [ ! -f sim/vec_state.hex ]; then
    echo "run_mutants.sh: no recorded vectors in sim/, run make golden first" >&2
    exit 1
fi

# where each vector set lives
vec_dir() {
    if [ "$1" = first ]; then echo sim; else echo "$WORK/vec"; fi
}

rm -rf "$WORK"
mkdir -p "$WORK"

# ------------------------------------------------------------ vector sets
echo "picking the corner-case vectors ..." >&2
if ! "$PYTHON" mutants/pick_corner_vectors.py --n "$N" --out "$WORK/vec" > "$WORK/coverage.txt"; then
    echo "run_mutants.sh: mutants/pick_corner_vectors.py failed" >&2
    exit 1
fi

# 00_original first, then every mutants/*.sed in name order
MUTANTS=00_original
for f in mutants/*.sed; do
    MUTANTS="$MUTANTS $(basename "$f" .sed)"
done

# ------------------------------------------------------- mutate, compile
# A mutant that cannot be tested gets a $dir/status file saying why ("sed
# error", "not applied", "compile error"). The report counts those as
# harness errors, never as caught bugs.
for name in $MUTANTS; do
    dir="$WORK/$name"
    mkdir -p "$dir"
    cp rtl/*.v "$dir/"
    if [ "$name" != 00_original ]; then
        file=$(sed -n 's|^# file: rtl/||p' "mutants/$name.sed")
        if ! sed -f "mutants/$name.sed" "rtl/$file" > "$dir/$file"; then
            echo "sed error" > "$dir/status"
            continue
        fi
        # diff exits with 1 when the files differ, which is what we want
        diff "rtl/$file" "$dir/$file" > "$dir/diff.txt"
        if [ $? -ne 1 ]; then
            echo "not applied" > "$dir/status"
            continue
        fi
    fi
    for set in $SETS; do
        vec=$(vec_dir "$set")
        # the same flags as the Makefile's testbench rule: -I. finds
        # rom/params.vh, -I$vec finds vectors.vh
        if ! iverilog -g2005 -Wall -I. -I"$vec" -DVEC_DIR="\"$vec\"" \
                -Ptb_rollout_engine.LANES="$LANES" -o "$dir/tb_$set" \
                "$dir/rollout_lane.v" "$dir/rollout_engine.v" tb/tb_rollout_engine.v \
                > "$dir/compile_$set.txt" 2>&1; then
            echo "compile error" > "$dir/status"
        fi
    done
done

# ------------------------------------------------------------- simulate
# JOBS simulations at a time: start JOBS in the background, wait for all
# of them, start the next JOBS.
started=0
for name in $MUTANTS; do
    for set in $SETS; do
        tb="$WORK/$name/tb_$set"
        [ -f "$WORK/$name/status" ] && continue
        echo "running $name on the $set $N decisions ..." >&2
        vvp -n "$tb" +n="$N" > "$WORK/$name/run_$set.txt" 2>&1 &
        started=$((started + 1))
        if [ $((started % JOBS)) -eq 0 ]; then wait; fi
    done
done
wait

# --------------------------------------------------------------- results
# failed FILE PREFIX: the testbench prints summary lines such as
#   costs: 287/300 decisions bit-exact (all 32 candidates)
# This prints how many decisions FAILED that check (300 - 287 = 13), or ?
# if the line is missing (the simulation stopped early).
failed() {
    local counts ok total
    counts=$(sed -n "s|^$2.* \([0-9][0-9]*\)/\([0-9][0-9]*\).*|\1 \2|p" "$1")
    if [ -z "$counts" ]; then echo "?"; return; fi
    read ok total <<< "$counts"
    echo $((total - ok))
}

# outcome NAME SET: PASS or FAIL, then the failing-decision counts
# costs/states/best, plus the clock and busy/done checks when they
# fired. "FAIL watchdog" if the engine hung.
outcome() {
    local log="$WORK/$1/run_$2.txt" result extra protocol
    if [ -f "$WORK/$1/status" ]; then cat "$WORK/$1/status"; return; fi
    if grep -q '^FAIL: watchdog' "$log"; then
        echo "FAIL watchdog"; return      # hung: no summary lines to read
    elif grep -q '^PASS$' "$log"; then
        result=PASS
    elif grep -q '^FAIL$' "$log"; then
        result=FAIL
    else
        echo "no result"; return          # the simulation crashed
    fi
    extra=""
    if [ "$(failed "$log" "clocks per decision")" != 0 ]; then
        extra="$extra, clocks $(failed "$log" "clocks per decision")"
    fi
    protocol=$(sed -n 's|^busy/done protocol errors: ||p' "$log")
    if [ -n "$protocol" ] && [ "$protocol" != 0 ]; then
        extra="$extra, busy/done $protocol"
    fi
    echo "$result $(failed "$log" "costs")/$(failed "$log" "final states")/$(failed "$log" "best_idx")$extra"
}

models() {
    if [ "$1" = 00_original ]; then
        echo "nothing planted: the RTL as committed"
    else
        sed -n 's|^# models: ||p' "mutants/$1.sed"
    fi
}

echo "# Planted-bug tests (DESIGN.md §7.4): bash mutants/run_mutants.sh"
echo "# started $STARTED, $(iverilog -V 2>&1 | head -n 1)"
echo "# rtl/ and tb/ from commit $(git rev-parse --short HEAD 2>/dev/null || echo '?')" \
     "($(git status --porcelain rtl tb 2>/dev/null | wc -l | tr -d ' ') uncommitted changes)," \
     "N = $N decisions per run, LANES = $LANES"
echo
echo "Vector sets (corner cases replayed through the golden model):"
sed 's|^|  |' "$WORK/coverage.txt"
echo
echo "Each result column: PASS or FAIL, then how many of the $N decisions failed each"
echo "testbench check: costs / final states / best (best_idx or best_cost wrong)."
echo "A mutant must FAIL."
echo
row='%-20s | %-66s | %-18s | %-18s\n'
printf "$row" "mutant" "what it models" "first $N" "corner $N"
printf "$row" "--------------------" \
    "------------------------------------------------------------------" \
    "------------------" "------------------"
for name in $MUTANTS; do
    printf "$row" "$name" "$(models "$name")" "$(outcome "$name" first)" "$(outcome "$name" corner)"
done
echo

# verdict: the original passes both sets, every mutant fails the corner set
errors=0
caught=0
caught_first=0
total=0
for set in $SETS; do
    case "$(outcome 00_original "$set")" in
        PASS*) ;;
        *) echo "ERROR: the original design does not PASS on the $set set"; errors=$((errors + 1)) ;;
    esac
done
for name in $MUTANTS; do
    [ "$name" = 00_original ] && continue
    total=$((total + 1))
    case "$(outcome "$name" corner)" in
        FAIL*) caught=$((caught + 1)) ;;
        PASS*) echo "NOT CAUGHT: $name passes on the corner set" ;;
        *)     echo "ERROR: $name: $(outcome "$name" corner)"; errors=$((errors + 1)) ;;
    esac
    case "$(outcome "$name" first)" in
        FAIL*) caught_first=$((caught_first + 1)) ;;
    esac
done
echo "first $N decisions:  $caught_first of $total mutants caught"
echo "corner $N decisions: $caught of $total mutants caught"
echo "wall time: $SECONDS s ($JOBS simulations at a time)"
if [ "$errors" -eq 0 ] && [ "$caught" -eq "$total" ]; then
    echo "ALL MUTANTS CAUGHT"
    status=0
else
    echo "MUTANT TEST FAILED"
    status=1
fi

# ------------------------------------------------------------- details
echo
echo "################################################################"
echo "# Details: each mutant's change and the testbench output"
for name in $MUTANTS; do
    echo
    echo "================================================================"
    echo "== $name: $(models "$name")"
    if [ -f "$WORK/$name/diff.txt" ]; then
        echo "-- change (diff rtl/ mutated):"
        cat "$WORK/$name/diff.txt"
    fi
    for set in $SETS; do
        if [ -s "$WORK/$name/compile_$set.txt" ]; then
            echo "-- compiler output, $set set:"
            cat "$WORK/$name/compile_$set.txt"
        fi
        if [ -f "$WORK/$name/run_$set.txt" ]; then
            echo "-- testbench, $set $N decisions:"
            cat "$WORK/$name/run_$set.txt"
        fi
    done
done

if [ "$KEEP" != 1 ]; then rm -rf "$WORK"; fi
exit $status
