# Study guide

How to learn this project well enough to explain every line without help. Work through it with the repo open, in order. The ten questions at the end are the gate: answer them out loud or on paper, without AI and without notes. If you can't answer at least 8 of them clearly, go back to the reading.

## 1. Reading order

1. **[DESIGN.md](DESIGN.md) §2, number formats** (20 min). For θ, ω, u and h, say what one LSB means in real units and what happens at the edge of the range. Check that 127 × 8/127 rad/s is exactly Pendulum-v1's speed limit of 8 rad/s, and that 256 × π/128 is exactly one turn.
2. **DESIGN.md §4, integer inference** (45 min). Read the pseudocode until you can write it from memory. Then §4.1, folding the scales: follow one weight from the float model to `rom/w1.hex`. Then §4.2, the cost and where 21/32 and 1/2048 come from.
3. **DESIGN.md §6, microarchitecture** (30 min). Lanes, weight broadcast, the action memory, the argmin, and "the key latency fact". Skip the Changelog for now.
4. **[rtl/rollout_lane.v](rtl/rollout_lane.v)** (1 h). Top to bottom. For each `wire`, say which clock its value belongs to; for each `always @(posedge clk)`, say what gets registered and when. The two `always @*` blocks are multiplexers, not registers.
5. **The FSM in [rtl/rollout_engine.v](rtl/rollout_engine.v)** (1 h). Start with the `case (state)` block near the end, then the broadcast block (`ex_*` and `wb_*` registers) that turns each FSM state into lane control bits one clock later, then the STORE loop and the argmin. Keep the header comment's clock table next to you.
6. **[tb/tb_rollout_engine.v](tb/tb_rollout_engine.v)** (30 min). How one decision is driven (write actions, pulse start, wait for done with a watchdog) and what is compared.
7. **[model/golden.py](model/golden.py)**: `fold()`, `int_step_one_lane()` and `argmin_first()` (30 min). The same math as the RTL, in plain Python integers.
8. **The logs** (30 min): [results/test.log](results/test.log), [results/mutants.log](results/mutants.log) (read the header notes), [results/cosim.log](results/cosim.log), [results/planner_quality.txt](results/planner_quality.txt), [results/synth/summary.txt](results/synth/summary.txt).

## 2. Hand calculation: one imagined step

Trace lane 0's first imagined step in decision 0 by hand, then check yourself against the "Hand check" block in [results/golden.log](results/golden.log) (or run `make golden` and read its printout). Use a calculator for multiplications, not for the shifts.

1. **Inputs.** The measured state is θ = −7, ω = 10, and lane 0's first action is u = −85. Convert each to real units with the LSB sizes from DESIGN.md §2.
2. **Neuron 0.** B1[0] = 643 and W1[0] = (29, 29, 0). Compute acc1[0] = B1[0] + 29·θ + 29·ω + 0·u, then `acc1[0] >>> 7`, then clamp to 0..127. You should get h[0] = 5.
3. **Neuron 2.** B1[2] = −2076 and W1[2] = (43, 1, 0). Compute acc1[2] and shift it right by 7. Is the shift result −18 or −19? Why? What does the clamp do with it?
4. **All 16.** Do the other 14 neurons from the weights in the log, and write h[0..15] in a row.
5. **Layer 2.** acc2[0] = B2[0] + Σ W2[0][j]·h[j], with B2[0] = −22 and the W2 row printed in the log. Shift by S2[0] = 8 to get d_θ. Then acc2[1] with B2[1] = −99, shifted by S2[1] = 7, to get d_ω. If you divided acc2[1] by 128 on a calculator and dropped the fraction, you'd get a different answer from the hardware. Which, and why?
6. **State update and cost.** θ′ = θ + d_θ (any wrap?), ω′ = clamp(ω + d_ω, −127, 127) (any clamp?), then c_step = θ′² + ((21·ω′²) >> 5) + (u² >> 11). You should get 55.
7. **Sanity check.** The log also shows what the float model and the real physics predict for the same step, in LSBs. How far off is the integer step?
8. **Next step.** Check your θ′, ω′, h[] and c_step against the first row of `sim/trace.txt` (written by `make golden`). Its second row is lane 0's step 1: do one value of it by hand.
9. **Timing.** Draw the clocks for neuron 0 and the start of neuron 1: in which clock does the FSM *issue* each MAC, in which does the lane *execute* it, and in which is h[0] *written back*? What is the multiplier doing in the clock h[0] is written?

## 3. Ten questions (the gate)

1. **Wrap vs saturate.** θ is allowed to overflow from +127 to −128, but ω is clamped at ±127. Why is wrapping right for θ and wrong for ω? Point to the lines in `rtl/rollout_lane.v` that do each. One planted bug swaps the ω rule: why did the first 300 recorded decisions not catch it?
2. **Folding the scales.** A usual int8 layer requantizes with `acc · M >> shift`. Why does this engine need only "add bias, arithmetic shift, clamp"? Where did the scale factors go, what does the `+2^(S−1)` inside each bias do, what approximation is left, and why must the shift be arithmetic (`>>>`)? tiny-tensor-core shares one requantizer multiplier among all its neurons: why wouldn't that work here?
3. **Accumulator widths.** Using the worst cases in `results/golden.log` (|acc1| ≤ 16,957, |acc2| ≤ 85,570, at most 26,975 cost per step), show that neither the 24-bit signed accumulator nor the 24-bit unsigned cost accumulator can overflow, even at H = 16. What are the narrowest widths that would still be safe? Why does `golden.py` work out the worst case from the weights instead of using the largest value seen in simulation?
4. **Clock breakdown.** Derive 1,281 clocks for LANES = 32, H = 15, state by state. Why is an imagined step exactly 83 clocks, with no extra clocks for writing h[j], θ′ and ω′ back? What is the DRAIN clock for?
5. **Lanes vs latency.** Going from 8 to 32 lanes cuts a decision from 5,022 to 1,281 clocks, 3.92× and not 4×. Why not 4×? Why can't more lanes bring a decision below 1,245 clocks at H = 15? Name two ways to cut latency and what each one costs.
6. **Argmin tie-break.** The argmin scans n = 0..31 with strict `<`. Tied candidates have the same cost, so why does it matter which one wins? What would go wrong in the co-simulation if the RTL used `<=`? How often are there ties in the recorded data?
7. **Verification strategy.** The testbench checks the 32 costs, the 32 final states and best_idx/best_cost separately. Why not just best_idx? Use the "no 21/32 weight" and "ω′ written back late" planted bugs in your answer. Why does a planted bug that PASSES not prove the bug is harmless, and how did this project deal with that?
8. **int8 vs float.** At H = 15 the int8 planner returned −146.6 and the float planner −147.2; the paired difference is +0.6 with a 95% CI of −7.7 to +9.6. What can you conclude, and what can't you? What happened at H = 5? What is one "seed" in that table?
9. **Co-simulation vs replay.** `make test-full` already replays 4,000 decisions bit-exact. What does `make cosim` test that the replay can't? Why are the episode returns compared with float `==` and no tolerance?
10. **What this does NOT show.** Name five things an interviewer might assume this project shows but it doesn't, and the evidence (or the missing evidence) for each.

## 4. Stretch (not part of the gate)

**Where does the area go?** The 32-lane build uses 13,804 LUT4s but only 33 DSP multipliers. Open [results/synth/stat_lanes32.txt](results/synth/stat_lanes32.txt) and account for the big items yourself: which registers and adders in `rtl/rollout_lane.v` become `CCU2C` carry chains, what `TRELLIS_DPR16X4` (LUT-based RAM) is storing, and how much comes from the operand multiplexers and the 32-entry result buffer. Compare with [stat_lanes8.txt](results/synth/stat_lanes8.txt): which parts scale with LANES and which don't? This is a likely interview follow-up to "33 multipliers."
