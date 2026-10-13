"""
Clocks per rollout_engine decision: a formula read off the state machine in
rtl/rollout_engine.v, checked against what the testbench measured, and how
busy the multipliers are (DESIGN.md §6, §7.5).

Run from the repo root (perf/sweep.sh runs it and saves results/cycles.log):
  python3 perf/cycles.py results/sweep.log

The formula, state by state. Every clock the FSM is in exactly one state, and
no state ever waits for data (no branch depends on a cost, a weight or an
action), so every decision with the same LANES and H takes the same number
of clocks:

  state  clocks        where it comes from in rtl/rollout_engine.v
  INIT   1 per pass    one clock, then always L1
  L1     48 per step   i = 0..2 inside j = 0..15: one MAC per (neuron, input)
  L2     32 per step   j = 0..15 inside i = 0..1: one MAC per (output, hidden)
  COST    3 per step   i = 0..2: theta'^2, omega'^2, u^2; then the next step's
                       L1, or STORE after step H-1
  STORE  1 per pass    then INIT for the next pass, or DRAIN after the last
  DRAIN  1             the last STORE lands one clock after it is issued
  ARG    32            n = 0..31: one candidate's cost compared per clock
  DONE   1             raise done, latch cycles

  PASSES = 32 / LANES                  (each pass imagines LANES candidates)
  cycles = PASSES * (1 + H * (48 + 32 + 3) + 1) + 1 + 32 + 1
         = PASSES * (83 * H + 2) + 34

Multiplier use. Each lane has one multiplier. A decision needs
32 candidates x H steps x 80 network MACs (48 + 32), and the same multiplier
also does the 3 cost squares per step. Utilization = useful MACs divided by
(multipliers x clocks): the fraction of multiplier-clocks that did a
network MAC.
"""

import re
import sys

N_CAND = 32                  # candidates per decision
N_IN, N_HID, N_OUT = 3, 16, 2
COST_TERMS = 3               # theta'^2, omega'^2, u^2
MACS_PER_STEP = N_HID * N_IN + N_OUT * N_HID      # 48 + 32 = 80 network MACs
CLOCKS_PER_STEP = MACS_PER_STEP + COST_TERMS      # 83: one multiply per clock
STATES = ["INIT", "L1", "L2", "COST", "STORE", "DRAIN", "ARG", "DONE"]

TB_CLOCK_PS = 10_000         # tb/tb_rollout_engine.v: always #5 clk = ~clk, so 10 ns
# Clocks the testbench spends per decision besides the engine's cycles and
# the 32 x H action bytes: 1 raising start, 1 lowering it, 1 checking that
# done fell after one clock, and 3 that pass during the 32 x #1 (1 ns)
# debug-port reads. A property of the testbench, not of the engine.
TB_OTHER = 6


def predict(lanes, H):
    """Clocks the FSM spends in each state during one decision."""
    passes = N_CAND // lanes
    return {
        "INIT":  passes,                         # once per pass
        "L1":    passes * H * N_HID * N_IN,      # 48 per imagined step
        "L2":    passes * H * N_OUT * N_HID,     # 32 per imagined step
        "COST":  passes * H * COST_TERMS,        # 3 per imagined step
        "STORE": passes,                         # once per pass
        "DRAIN": 1,                              # once per decision
        "ARG":   N_CAND,                         # once per decision, 1 clock per candidate
        "DONE":  1,                              # once per decision
    }


def read_runs(path):
    """One dict per testbench run in a log written by perf/sweep.sh.

    Each run starts with a line "# run LANES=.. H=..", followed by the
    testbench's own output, which this reads back:
      clocks in each FSM state, decision 0: INIT 1, L1 720, ...
      config: LANES = 32, H = 15, vectors from sim/perf/h15
      decisions run: 200
      clocks per decision: 1281..1281 (...)
      PASS
      tb/tb_rollout_engine.v:185: $finish called at 353400022000 (1ps)
    """
    runs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("# run "):
                runs.append({})
                continue
            if not runs:                         # the header and table come first
                continue
            r = runs[-1]
            m = re.match(r"clocks in each FSM state, decision 0: (.*)", line)
            if m:
                # "INIT 1, L1 720, ..." -> {"INIT": 1, "L1": 720, ...}
                r["states"] = {name: int(v) for name, v in re.findall(r"(\w+) (\d+)", m[1])}
            m = re.match(r"config: LANES = (\d+), H = (\d+)", line)
            if m:
                r["lanes"], r["H"] = int(m[1]), int(m[2])
            m = re.match(r"decisions run: (\d+)", line)
            if m:
                r["n"] = int(m[1])
            m = re.match(r"clocks per decision: (\d+)\.\.(\d+)", line)
            if m:
                r["min"], r["max"] = int(m[1]), int(m[2])
            if line in ("PASS", "FAIL"):
                r["verdict"] = line
            m = re.search(r"\$finish called at (\d+) \(1ps\)", line)
            if m:
                r["finish_ps"] = int(m[1])
    return runs


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/sweep.log"
    runs = read_runs(path)
    # a run that crashed or hit the watchdog never prints these lines
    complete = [r for r in runs if "states" in r and "min" in r and "lanes" in r]
    ok = len(runs) > 0 and len(complete) == len(runs)
    if not runs:
        print(f"no '# run' blocks in {path}: is it a log from perf/sweep.sh?")
    elif not ok:
        print(f"{len(runs) - len(complete)} of {len(runs)} runs in {path} did not finish")

    print(f"Clocks per decision: formula vs measured ({path})")
    print("  formula: cycles = (32 / LANES) * (83 * H + 2) + 34, read off rtl/rollout_engine.v")
    print("  measured: the engine's cycles output (min..max over the run), and the clocks")
    print("  per FSM state that the testbench counted for decision 0\n")
    print("  LANES   H  passes  predicted  measured  difference  8 states match  testbench")
    for r in complete:
        passes = N_CAND // r["lanes"]
        p = predict(r["lanes"], r["H"])
        pred = sum(p.values())
        # the state-by-state sum and the closed form are the same formula
        assert pred == passes * (CLOCKS_PER_STEP * r["H"] + 2) + 34
        same = r["min"] == r["max"] == pred and r["states"] == p
        ok = ok and same and r.get("verdict") == "PASS"
        if r["min"] == r["max"]:
            measured = str(r["min"])
        else:
            measured = f"{r['min']}..{r['max']}"
        print(f"  {r['lanes']:5d} {r['H']:3d} {passes:7d} {pred:10d} {measured:>9} "
              f"{r['max'] - pred:11d} {'yes' if r['states'] == p else 'NO':>15} "
              f"{r.get('verdict', '?'):>10}")
        if r["states"] != p:
            for s in STATES:
                print(f"      {s:5s} predicted {p[s]:5d}, measured {r['states'].get(s)}")

    print("\nWhere the clocks go, per FSM state (the formula; checked against decision 0 above)")
    print("  LANES   H " + "".join(f"{s:>7}" for s in STATES) + "    total")
    for r in complete:
        p = predict(r["lanes"], r["H"])
        print(f"  {r['lanes']:5d} {r['H']:3d} " + "".join(f"{p[s]:7d}" for s in STATES)
              + f" {sum(p.values()):8d}")

    print("\nMultiplier utilization = useful MACs / (multipliers x clocks)")
    print("  useful MACs = 32 candidates x H steps x 80 network MACs (48 layer 1 + 32 layer 2)")
    print("  multipliers = LANES (one per lane); clocks = measured clocks per decision")
    print("  'busy' also counts the 3 cost squares per step, which use the same multiplier\n")
    print("  LANES   H  multipliers  clocks  useful MACs  mult-clocks  utilization    busy")
    for r in complete:
        useful = N_CAND * r["H"] * MACS_PER_STEP
        squares = N_CAND * r["H"] * COST_TERMS
        mult_clocks = r["lanes"] * r["max"]
        print(f"  {r['lanes']:5d} {r['H']:3d} {r['lanes']:12d} {r['max']:7d} {useful:12d} "
              f"{mult_clocks:12d} {100 * useful / mult_clocks:11.2f}% "
              f"{100 * (useful + squares) / mult_clocks:6.2f}%")
    print("  The multipliers sit idle in INIT and STORE (2 clocks per pass) and in DRAIN, ARG")
    print("  and DONE (34 clocks per decision). Fewer lanes means more passes over the same")
    print("  34-clock tail, so utilization goes up as LANES goes down, even though each")
    print("  decision takes longer.")

    print("\nThe floor that H sets: one candidate's H imagined steps run one after another")
    print("  LANES   H  one candidate = 83*H  decision  decision - floor")
    for r in complete:
        floor = CLOCKS_PER_STEP * r["H"]
        print(f"  {r['lanes']:5d} {r['H']:3d} {floor:20d} {r['max']:9d} {r['max'] - floor:17d}")
    clocks = {(r["lanes"], r["H"]): r["max"] for r in complete}
    for H in sorted({r["H"] for r in complete}, reverse=True):
        if (8, H) in clocks and (32, H) in clocks:
            print(f"  H = {H}: LANES 8 -> 32 takes {clocks[8, H]} -> {clocks[32, H]} clocks, "
                  f"{clocks[8, H] / clocks[32, H]:.2f}x fewer for 4x the lanes")

    # Before every start the testbench writes the 32 x H actions, one byte per
    # clock. The engine's cycles output does not count that, but the
    # simulation time does: total time / clock period / decisions. The 20 ns
    # of reset and the 2 ns after the last readback add up to less than one
    # clock per run, and round() removes them.
    print("\nTestbench clocks per decision, including the host writing the actions")
    print("  LANES   H  testbench  =  cycles  +  32*H action bytes  +  testbench other")
    for r in complete:
        if "finish_ps" not in r or "n" not in r:
            continue
        tb = round(r["finish_ps"] / TB_CLOCK_PS / r["n"])
        load = N_CAND * r["H"]
        other = tb - r["max"] - load
        ok = ok and other == TB_OTHER
        print(f"  {r['lanes']:5d} {r['H']:3d} {tb:10d}  = {r['max']:7d}  + {load:18d}  + {other:16d}"
              + ("" if other == TB_OTHER else f"   (expected {TB_OTHER})"))
    print(f"  testbench other = {TB_OTHER}: 1 raising start, 1 lowering it, 1 checking that done fell,")
    print("  and 3 that pass during the 32 one-nanosecond debug-port reads.")

    print("\nRESULT: " + ("PASS, the formula matches every run exactly" if ok else "FAIL, see above"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
