#!/usr/bin/env bash
# Synthesize rollout_engine, then place and route it on a Lattice ECP5-45F
# (LFE5U-45F, CABGA381 package) with three placement seeds and report each
# seed's post-route max clock frequency (DESIGN.md §7.5).
#
#   bash synth/timing_ecp5.sh                  32 lanes, then 8 lanes
#   LANES=32 bash synth/timing_ecp5.sh         only the 32-lane engine
#   YOSYS=yowasp-yosys NEXTPNR=yowasp-nextpnr-ecp5 bash synth/timing_ecp5.sh
#
# Needs yosys and nextpnr-ecp5 (OSS CAD Suite, or the YoWASP pip packages
# with the names above), and rom/*.hex from "make golden".
#
# Out-of-context: there is no pin-constraint (.lpf) file, so nextpnr puts
# the 129 I/O pins wherever it likes. The Fmax is an estimate for the engine
# on its own, not for a board.
#
# Writes, for each lane count N:
#   results/synth/stat_lanesN.txt          yosys resource report
#   results/synth/nextpnr_lanesN_sS.log    nextpnr log for seed S, trimmed to
#                                          utilisation + post-route timing
# plus results/synth/summary.txt and results/synth/tool_versions.txt.
# Full logs stay in sim/, which is not committed.

set -euo pipefail                # stop on any failed command, unset variable or failed pipe

YOSYS=${YOSYS:-yosys}
NEXTPNR=${NEXTPNR:-nextpnr-ecp5}
LANES=${LANES:-"32 8"}           # one or more lane counts, each must divide 32
FREQ=50                          # target clock in MHz; nextpnr says PASS or FAIL against it
H=15                             # imagined steps, only for the clocks line in the summary

cd "$(dirname "$0")/.."          # repo root: $readmemh opens rom/*.hex relative to it

# Check everything up front, so a missing tool or ROM stops the script
# before it overwrites any of the committed results.
for tool in "$YOSYS" "$NEXTPNR"; do
    if ! command -v "$tool" > /dev/null; then
        echo "$tool not found: install it, or set YOSYS / NEXTPNR (see the top of this file)" >&2
        exit 1
    fi
done
for f in rom/params.vh rom/w1.hex rom/b1.hex rom/w2.hex rom/b2.hex; do
    if [ ! -f "$f" ]; then
        echo "missing $f: run 'make golden' first" >&2
        exit 1
    fi
done
mkdir -p sim results/synth

# nextpnr prints its version on stderr, hence the 2>&1
{ $YOSYS -V; $NEXTPNR --version 2>&1; } > results/synth/tool_versions.txt

# The count of one cell type in the yosys report $stat (set in the loop
# below): the line whose second word is the cell name, e.g. "  6706   LUT4".
# Prints 0 if the cell type is not there.
cells() { awk -v c="$1" '$2 == c { n = $1 } END { print n + 0 }' "$stat"; }

SUMMARY=results/synth/summary.txt
{
    echo "rollout_engine on a Lattice ECP5-45F (LFE5U-45F, CABGA381)."
    echo "Post-route estimate, out-of-context (no pin constraints). Target clock $FREQ MHz."
    echo "Reproduce from the repo root: YOSYS=$YOSYS NEXTPNR=$NEXTPNR LANES=\"$LANES\" bash synth/timing_ecp5.sh"
    sed 's/^/  /' results/synth/tool_versions.txt
} > "$SUMMARY"

for lanes in $LANES; do
    # ---------------------------------------------------------- synthesis
    # synth/ecp5.ys builds 32 lanes. Make a copy with the lane count swapped:
    # "LANES 32" is its chparam line, "lanes32" is in its two output names.
    ys=sim/ecp5_lanes$lanes.ys
    json=sim/rollout_ecp5_lanes$lanes.json
    stat=results/synth/stat_lanes$lanes.txt
    sed -e "s/LANES 32/LANES $lanes/" -e "s/lanes32/lanes$lanes/g" synth/ecp5.ys > "$ys"
    echo "LANES=$lanes: yosys (full log in sim/yosys_lanes$lanes.log)"
    $YOSYS -q -l "sim/yosys_lanes$lanes.log" -s "$ys"

    {
        echo
        echo "LANES = $lanes"
        echo "  yosys cells ($stat):"
        echo "    LUT4 $(cells LUT4)   CCU2C $(cells CCU2C)   PFUMX $(cells PFUMX)   L6MUX21 $(cells L6MUX21)"
        echo "    TRELLIS_FF $(cells TRELLIS_FF)   MULT18X18D $(cells MULT18X18D)   DP16KD $(cells DP16KD)   TRELLIS_DPR16X4 $(cells TRELLIS_DPR16X4)"
        echo "  post-route Fmax (target $FREQ MHz):"
    } >> "$SUMMARY"

    # ---------------------------------------------------------- place and route
    fmaxes=""
    for seed in 1 2 3; do
        log=sim/nextpnr_lanes${lanes}_s$seed.log
        echo "LANES=$lanes: nextpnr seed $seed (full log in $log)"
        # Without --timing-allow-fail, nextpnr exits nonzero whenever a seed
        # misses the target clock, and we still want that seed's number.
        # A real failure (does not fit, cannot route) still exits nonzero,
        # and set -e stops the script there.
        $NEXTPNR --45k --package CABGA381 --json "$json" --seed "$seed" \
            --freq "$FREQ" --timing-allow-fail --quiet --log "$log"

        # nextpnr prints "Max frequency" twice: an estimate after placement
        # and the real number after routing, so take the last one. Format:
        #   ... Max frequency for clock '<name>': 47.49 MHz (FAIL at 50.00 MHz)
        result=$(sed -E -n "s/.*Max frequency for clock '[^']*': ([0-9.]+) MHz \((PASS|FAIL) at.*/\1 \2/p" "$log" | tail -n 1)
        if [ -z "$result" ]; then
            echo "no Max frequency line in $log" >&2
            exit 1
        fi
        read -r fmax verdict <<< "$result"      # split "47.49 FAIL" into two variables
        echo "    seed $seed: $fmax MHz ($verdict at $FREQ MHz)" | tee -a "$SUMMARY"
        fmaxes="$fmaxes $fmax"

        # Committed copy: the resource tables, then everything after routing
        # (critical path reports, post-route Fmax, slack histogram). The
        # placer and router progress tables in between are left out.
        {
            echo "# nextpnr log trimmed by synth/timing_ecp5.sh: placer and router progress removed."
            echo "# $NEXTPNR --45k --package CABGA381 --json $json --seed $seed --freq $FREQ --timing-allow-fail --quiet --log $log"
            sed -n '/^Info: Logic utilisation before packing:/,/^Info: *Total DFFs:/p' "$log"
            echo
            sed -n '/^Info: Device utilisation:/,/^$/p' "$log"
            sed -n '/^Info: Routing complete/,$p' "$log"
        } > "results/synth/nextpnr_lanes${lanes}_s$seed.log"
    done
    rm "$json"                   # the netlist is about 17 MB; rerunning rebuilds it

    # $fmaxes is left unquoted on purpose: printf gets one number per argument.
    # Sorted, the middle of three values is the median.
    median=$(printf '%s\n' $fmaxes | sort -n | sed -n 2p)
    clocks=$(( 32 / lanes * (83 * H + 2) + 34 ))
    {
        echo "    median: $median MHz"
        echo "  nextpnr utilisation (from results/synth/nextpnr_lanes${lanes}_s1.log; the same for every seed):"
        sed -E -n 's/^Info:[[:space:]]*(Total LUT4s|TRELLIS_COMB|TRELLIS_FF|TRELLIS_RAMW|MULT18X18D|DP16KD|TRELLIS_IO):[[:space:]]*/    \1 /p' \
            "results/synth/nextpnr_lanes${lanes}_s1.log"
        echo "    (Total LUT4s is before packing: logic + carry-chain + LUTRAM LUT4s. TRELLIS_COMB is after.)"
        echo "  clocks per decision at H = $H: $clocks"
        echo "    (formula (32/LANES)*(83H+2)+34, which matches the measured counts in results/config_sweep.log)"
        echo "  one decision at the median Fmax: $clocks / $median MHz =" \
             "$(awk -v c="$clocks" -v f="$median" 'BEGIN { printf "%.1f", c / f }') us"
    } >> "$SUMMARY"
done

echo
cat "$SUMMARY"
