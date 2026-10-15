#!/usr/bin/env bash
# Summarizes Tang Nano 20K engine builds from the nextpnr logs in
# board/build/ (made by "make bit"): resource use, post-route Fmax and the
# timing slack at the board's 27 MHz.
#
#   bash board/report.sh 8 1                 LANES = 8, seed 1
#   bash board/report.sh 32 "1 2"            LANES = 32, seeds 1 and 2
#   bash board/report.sh 8 1 -noalu          the build made with SYNTH_OPTS=-noalu
# "make results" runs these and writes board/results/build_l8.txt and build_l32.txt.
#
# Slack = clock period - the critical path's delay. nextpnr's Fmax is
# 1 / (critical path delay), so slack = 1000/27 ns - 1000/Fmax ns. Positive
# slack means the path settles with that much time to spare at 27 MHz.
set -euo pipefail                  # stop on any failed command or unset variable

cd "$(dirname "$0")"               # board/
lanes=$1
seeds=$2
opts=${3:-}                        # extra synth_gowin options, as given to make SYNTH_OPTS=
# the same file-name suffix the Makefile makes from them: "-noalu" -> "_noalu"
var=$(for o in $opts; do printf '_%s' "${o#-}"; done)
OSS=${OSS:-$HOME/oss-cad-suite/bin}
FREQ=27
period=$(awk -v f="$FREQ" 'BEGIN { printf "%.2f", 1000 / f }')

echo "board/top.v (rollout_engine LANES = $lanes, H = 15, UART wrapper) on the Tang Nano 20K"
echo "(GW2AR-LV18QN88C8/I7), open-source flow: yosys synth_gowin${opts:+ $opts} -> nextpnr-himbaechel -> gowin_pack."
echo "Built with: cd board && make bit LANES=$lanes SEED=<seed>${opts:+ SYNTH_OPTS=$opts}"
echo "Summarized by: bash board/report.sh $lanes \"$seeds\"${opts:+ $opts}"
echo "Tools: $("$OSS/yosys" -V)"
echo "       $("$OSS/nextpnr-himbaechel" --version 2>&1 | head -n 1)"
echo "       apycula (gowin_pack) $(sed -n 's/^Version: //p' "$OSS"/../lib/python3*/site-packages/apycula-*.dist-info/METADATA)"
echo
echo "These are the tools' estimates. These DSP-block builds are not the one that ran on the board (that is the -nodsp build: results/build_nodsp.txt, results/hardware/)."
echo "nextpnr's Gowin timing numbers look optimistic: the ECP5 build of the same engine"
echo "(results/synth/summary.txt) reached 47-54 MHz, and its critical path was the lane"
echo "multiplier. The board does not depend on them: it runs at 27 MHz from the 27 MHz oscillator (U3)."
echo

for seed in $seeds; do
    log=build/pnr_l${lanes}${var}_s$seed.log
    # nextpnr prints "Max frequency" after placement (an estimate) and again
    # after routing; the last one is the post-route number
    fmax=$(sed -E -n "s/.*Max frequency for clock '[^']*': ([0-9.]+) MHz.*/\1/p" "$log" | tail -n 1)
    split=$(grep -E "ns logic, .* ns routing" "$log" | tail -n 1 | sed 's/^Info: //')
    slack=$(awk -v p="$period" -v f="$fmax" 'BEGIN { printf "%.2f", p - 1000 / f }')
    echo "seed $seed: post-route Fmax $fmax MHz; critical path $(awk -v f="$fmax" 'BEGIN { printf "%.2f", 1000 / f }') ns ($split)"
    echo "        slack at $FREQ MHz ($period ns period): +$slack ns"
done

first=$(echo "$seeds" | awk '{ print $1 }')
log=build/pnr_l${lanes}${var}_s$first.log
echo
echo "nextpnr device utilisation (seed $first; placement does not change the cell counts), unused types left out:"
sed -n '/^Info: Device utilisation:/,/^Info: *$/p' "$log" | grep -E '[0-9]+/' | grep -v -E ' 0/ ' | sed 's/^Info:[[:space:]]*/  /'
# An ALU cell (one bit of a carry chain) sits in a LUT's place: in the placed
# designs here no slice ever uses ALUk and LUTk with the same k. So the
# logic actually left is 20,736 minus both counts together.
awk '/^Info: Device utilisation:/ { on = 1 }
     on && $2 == "LUT4:" { split($3, a, "/"); lut = a[1]; total = $4 }
     on && $2 == "ALU:"  { split($3, a, "/"); alu = a[1] }
     on && /^Info: *$/   { on = 0 }
     END { printf "  LUT positions used (LUT4 + ALU, which share them): %d of %d (%.0f%%)\n",
                  lut + alu, total, 100 * (lut + alu) / total }' "$log"
echo
echo "Critical path, seed $first (from nextpnr's post-route report). Names are yosys's: the"
echo "part before the first _DFF/_LUT/_ALU/_MUX is the Verilog signal (u_eng. = inside the engine)."
# the last "Critical path report" section: its first Source is where the path
# starts (a flip-flop), its last Sink where it ends; count the logic stages
awk '/Critical path report for clock/ { n = 0; src = ""; on = 1; next }
     on && /Source/ && src == "" { src = $NF }
     on && /Sink/                { sink = $NF }
     on && $2 == "logic"         { n++ }
     on && /ns logic,/           { on = 0; s = src; k = sink; l = n; t = $0 }
     END { sub(/^Info: /, "", t)
           printf "  from %s\n  to   %s\n  %d logic stages; %s\n", s, k, l, t }' "$log"
