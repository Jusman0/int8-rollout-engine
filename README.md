# int8-rollout-engine

[![tests](https://github.com/Jusman0/int8-rollout-engine/actions/workflows/test.yml/badge.svg)](https://github.com/Jusman0/int8-rollout-engine/actions/workflows/test.yml)

A 32-lane int8 Verilog engine that, every control tick, imagines 32 possible futures of a pendulum through a small learned physics model, scores each one, and hands back the best plan. It's sampling-based model-predictive control with the model rollouts in hardware, checked bit-exact against a Python golden model and in a closed loop where the RTL picks every action. Everything runs in simulation (Icarus Verilog), with synthesis estimates for an ECP5 FPGA.

## Results

Every result here comes from a log in [`results/`](results/), and the command next to it makes that log again. Runs: Apple M2, Icarus Verilog 13.0, Python 3.14, NumPy 2.5.4, gymnasium 1.4.0. (The DreamZero and GR00T figures further down come from the sources linked there, not from this repo.)

| Result | Log | Reproduce |
|---|---|---|
| **Bit-exact replay**: for each of **4,000 / 4,000** recorded planner decisions, all 32 candidate costs, all 32 final states (θ, ω) and the chosen plan (best_idx, best_cost) match the golden model. 32 lanes, H = 15 imagined steps | [`test_full.log`](results/test_full.log) | `make test-full` |
| The same 4,000 / 4,000 at 16 and 8 lanes, and at 32, 16 and 8 lanes with H = 5 | [`config_sweep.log`](results/config_sweep.log) | `make test-full LANES=8`; for H = 5, `python3 model/golden.py --H 5 --out sim/h5` then `make test VEC=sim/h5 N=4000 LANES=8` |
| **Closed loop, the RTL in charge**: the RTL picks every action of 20 pendulum episodes. **20 / 20** episode returns identical to the golden model's (float `==`), **4,000 / 4,000** choices the same | [`cosim.log`](results/cosim.log) | `make cosim` |
| **Clocks per decision**: **1,281** at 32 lanes, H = 15. All 6 configurations (below) equal (32 / LANES)·(83·H + 2) + 34 exactly | [`sweep.log`](results/sweep.log), [`cycles.log`](results/cycles.log) | `make sweep` |
| **int8 vs float planner**: no measurable difference at H = 15, **+0.6** return (95% CI −7.7 to +9.6, 100 episodes). At H = 5, int8 is worse by 142 | [`planner_quality.txt`](results/planner_quality.txt) | `python3 model/golden.py --eval` |
| **Planted bugs**: **9 / 9** caught on the corner-case decisions (only 6 / 9 on the first 300 decisions, see below) | [`mutants.log`](results/mutants.log) | `make mutants` |
| **ECP5-45F, 32 lanes**: 13,804 LUT4s before packing (31%), 33 of 72 DSP multipliers. **47.49 MHz** post-route estimate, out-of-context, median of 3 seeds, so it **misses** its 50 MHz target | [`synth/summary.txt`](results/synth/summary.txt) | `make timing` |
| The learned model's one-step error on held-out data: 0.108 LSB in θ, 0.484 LSB in ω | [`train_dynamics.log`](results/train_dynamics.log) | `python3 model/train_dynamics.py` |

### Clocks per decision

From [`results/sweep.log`](results/sweep.log) and [`results/cycles.log`](results/cycles.log) (`make sweep`):

| LANES | passes over the 32 candidates | H = 15 | H = 5 |
|---|---|---|---|
| 32 | 1 | **1,281** | 451 |
| 16 | 2 | 2,528 | 868 |
| 8 | 4 | 5,022 | 1,702 |

Every measured count equals `(32 / LANES)·(83·H + 2) + 34`, and so does every FSM state's share. At 32 lanes and H = 15 the multipliers do a network MAC on 93.7% of clocks. `cycles` runs from `start` to `done`; writing the 32 × H action bytes through the 8-bit host port beforehand takes another 32·H clocks (480 at H = 15), which the engine doesn't count.

### Planner quality

From [`results/planner_quality.txt`](results/planner_quality.txt) (`python3 model/golden.py --eval`). Mean return over 5 seeds × 20 episodes of 200 steps; higher is better, a good controller scores around −150:

| Planner | H = 15 | H = 5 |
|---|---|---|
| float32 model, float cost | −147.2 | −968.6 |
| int8 model, integer cost (exactly what the RTL computes) | **−146.6** | −1,111.0 |
| no torque at all, for scale | −1,121.3 | −1,121.3 |

| Paired difference (bootstrap 95% CI, 10,000 resamples) | Mean | 95% CI |
|---|---|---|
| int8 − float, H = 15 | +0.6 | −7.7 to +9.6 |
| int8 − float, H = 5 | −142.4 | −170.9 to −118.0 |

A "seed" here is a different set of 20 start states and planner random numbers, not a different model: there is one trained model. At H = 5 (a 0.25 s look-ahead) both planners do badly: the float one is only somewhat better than no torque, and the int8 one is about as bad as no torque. I haven't found out why int8 does worse there.

### Planted bugs

From [`results/mutants.log`](results/mutants.log) (`make mutants`). Each bug is a small `sed` edit (one to three lines) to a copy of `rtl/`, run through the unchanged testbench. A bug counts as caught if the testbench says FAIL.

| Planted bug | First 300 decisions | Corner-case 300 |
|---|---|---|
| θ saturates at ±127 instead of wrapping | caught | caught |
| ω wraps instead of saturating | **missed** | caught |
| argmin uses `<=`, so a tie picks the last candidate | **missed** | caught |
| logical `>>` instead of arithmetic `>>>` when requantizing | caught | caught |
| hidden values clamp at 255 instead of 127 | **missed** | caught |
| ω cost term without its 21/32 weight | caught | caught |
| the step loop stops one imagined step early | caught | caught |
| layer 2's accumulator never restarts from the bias | caught | caught |
| ω′ written back one clock late, so its cost squares the old ω | caught | caught |

The three misses aren't harmless bugs. The first 300 decisions never saturate ω, never clamp a hidden value and never have two candidates tied for the lowest cost, so those bugs compute exactly what the correct design computes on that data. The corner-case set takes all 71 of the 4,000 recorded decisions that hit one of those cases, plus the earliest others, and catches all nine. Two bugs (the missing 21/32 and the late ω′) leave every final state right; only the cost check sees them, which is why the testbench checks costs and states separately.

### Synthesis

From [`results/synth/`](results/synth/) (`make timing YOSYS=... NEXTPNR=...`; Yosys 0.69, nextpnr-ecp5 0.11.1, LFE5U-45F). All MHz and µs figures are **post-route estimates, out-of-context** (no pin constraints), from the median of 3 placement seeds. Nothing has been run on a board.

| | 32 lanes | 8 lanes |
|---|---|---|
| LUT4s (before packing) | 13,804 of 43,848 (31%) | 6,338 (14%) |
| DSP multipliers (MULT18X18D) | 33 of 72: one per lane, plus one | 9 of 72 |
| Flip-flops | 3,751 | 2,025 |
| Post-route Fmax, seeds 1 / 2 / 3 | 47.49 / 48.94 / 47.26 MHz | 53.53 / 51.54 / 51.20 MHz |
| Median vs the 50 MHz target | **47.49 MHz, misses on every seed** | 51.54 MHz, meets on every seed |
| Clocks per decision (H = 15) | 1,281 | 5,022 |
| One decision at the median Fmax | 27.0 µs | 97.4 µs |

In all six runs the slowest path is the same single clock: a broadcast control register → the operand multiplexer → the lane's multiplier → the 21/32 shift-and-add → the 24-bit cost adder. About 60% of it is routing (8.4 ns of logic, 12.1 to 12.8 ns of routing at 32 lanes), because one register fans out to all 32 lanes. Registering the product before the cost add is the obvious next step (about one extra clock per pass); I haven't tried it, so I don't know what it gains.

## How it works

```
         host: measured state (θ, ω) and 32 candidate plans of H actions each
            │                                    │
            │ theta0, omega0, start              │ act_* port, one byte per clock
            │                                    ▼
            │                  ACT memory: ACT[t] = 32 actions x 8 bits
            │                                    │ lane n reads byte n of ACT[t]
            ▼                                    ▼
         ┌────────┬────────┬────────┬── ... ──┬─────────┐   each lane: its own θ, ω,
         │ lane 0 │ lane 1 │ lane 2 │         │ lane 31 │   h[0..15], one signed 8x8
         └────────┴────────┴────────┴── ... ──┴─────────┘   multiplier, 24-bit acc, cost
             ▲        ▲        ▲                   ▲
             └────────┴────────┴──────── ... ──────┘
                              │  one weight (and bias) per clock,
                              │  the same for all 32 lanes
         weight ROM: W1 48, B1 16, W2 32, B2 2       FSM: issue → execute → write back

   after H imagined steps:  32 costs ──► argmin, strict < (32 clocks) ──► best_idx, best_cost
```

Each lane imagines one candidate plan. Every clock the FSM reads one weight from ROM and broadcasts it, and each lane multiplies it by its own operand (θ, ω, u or a hidden value): one weight fetch feeds 32 MACs, the same idea as a GPU warp sharing one instruction. One imagined step for one lane ([DESIGN.md §4](DESIGN.md)):

```
for j in 0..15:  h[j] = clamp((B1[j] + W1[j]·(θ, ω, u)) >>> 7, 0, 127)     layer 1, 48 clocks
for k in 0..1:   d[k] = (B2[k] + W2[k]·h) >>> S2[k]                          layer 2, 32 clocks
θ' = θ + d[0]                    8-bit add: overflow wraps +π around to −π
ω' = clamp(ω + d[1], −127, 127)  saturates: a wrapped speed would flip sign
cost += θ'² + ((21·ω'²) >> 5) + (u² >> 11)                                    3 clocks
```

| Phase (LANES = 32, H = 15) | Clocks | Work |
|---|---|---|
| INIT | 1 | every lane loads the measured state, clears its cost |
| Layer 1 | 48 × 15 = 720 | 16 neurons × 3 inputs, one MAC per lane per clock |
| Layer 2 | 32 × 15 = 480 | 2 outputs × 16 hidden values |
| Cost | 3 × 15 = 45 | θ′², ω′², u² on the same multiplier |
| STORE, DRAIN | 2 | copy costs and final states to a result buffer |
| Argmin | 32 | one candidate compared per clock |
| DONE | 1 | |
| **Total** | **1,281** | measured per state in [`results/test.log`](results/test.log) |

**No requantization multiplier.** A typical int8 layer rescales with `acc · M >> shift`, which would need a second multiplier in every lane (tiny-tensor-core uses a 24×17 one). Here every scale factor is folded into the integer weights offline, and each layer's scale is rounded to a power of two, so requantizing is just add bias, arithmetic shift, clamp. The bias also carries +2^(S−1), which turns the floor shift into round-half-up for free. The write-back happens in the clock after a dot product's last MAC, while the multiplier already works on the next one, so it adds no clocks.

## The latency lesson

More lanes buy more candidates, not a shorter decision: going from 8 to 32 lanes cuts a decision from 5,022 to 1,281 clocks only because 8 lanes need four passes over the same 32 candidates. Inside one candidate, step t + 1 needs the state from step t, so its H imagined steps run one after another at 83 clocks each, a floor of 1,245 clocks at H = 15 that the 32-lane engine is only 36 clocks above ([`results/cycles.log`](results/cycles.log)). What's left is fewer imagined steps (H = 5 takes 451 clocks, but plans badly, see above) or a cheaper step, which means more multipliers working on one candidate. [DreamZero](https://arxiv.org/abs/2602.15922) shows the same pressure at scale: its latency is denoising steps × one forward pass of a 14-billion-parameter model, and it cut 16 steps to an effective 4 and then to 1 on the way from about 5.7 s to 150 ms per action chunk.

## Why this connects to robotics

In his talk at [Sequoia AI Ascent 2026](https://sequoiacap.com/article/ai-ascent-2026), Jim Fan argued that robots will act by imagining the future with learned world models. That turns control into an inference-latency problem, because every decision costs candidates × imagined steps × one forward pass, and the steps can't run in parallel. NVIDIA's GR00T N1.7, a 3B-parameter robot model, takes 39.9 ms per inference on a Jetson AGX Thor (MAXN power mode, 4 denoising steps, 1 camera, batch 1, TensorRT with mixed NVFP4 and FP8 layers), per the [Jetson AI Lab tutorial](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/). This project is the same loop shrunk to a 98-parameter model and a 2-number state, small enough to check every bit. Its clock counts and MHz estimates say nothing about real robot models and aren't comparable to GR00T's numbers.

## Limitations

- **A toy task.** The state is two numbers, the model is 3 → 16 → 2 (98 parameters), and the task is gymnasium's Pendulum-v1.
- **This task doesn't need the hardware's speed.** Pendulum-v1 advances 50 ms per step, and the NumPy golden model already makes a decision in about 0.35 ms on the same M2 ([`results/cosim.log`](results/cosim.log): 4,000 decisions in 1.4 s).
- **Simulation only.** The RTL runs in Icarus Verilog. There's no FPGA board; the Fmax figures are post-route estimates, out-of-context, and the 32-lane build misses its 50 MHz target.
- **What int8 costs.** At H = 15 nothing measurable (+0.6, 95% CI −7.7 to +9.6). At H = 5 the int8 planner is 142 worse than the float one, for a reason I haven't found.
- **The cost is approximated.** The real cost in integer units has weights 0.6587 on ω² and 0.000412 on u². The hardware uses 21/32 = 0.65625 and 1/2048 = 0.000488, so that they're shifts and adds, and each term is rounded down.
- **One trained model**, seed 0. The planted bugs were run at 32 lanes and H = 15 only, and the `cycles` output leaves out loading the actions (above).
- CI (`.github/workflows/test.yml`) runs `make golden && make test-full` with Ubuntu's Icarus Verilog package. It hasn't run yet; every log here is from Icarus 13.0 on macOS.

## Run it

Needs [Icarus Verilog](https://steveicarus.github.io/iverilog/) (`brew install icarus-verilog`) and Python 3 (3.14 was used) with the packages in [`requirements.txt`](requirements.txt) (`pip install -r requirements.txt`; if they're in a virtualenv, add `PYTHON=path/to/venv/bin/python` to the make command). Yosys and nextpnr-ecp5 ([OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build), or the YoWASP pip packages with `YOSYS=yowasp-yosys NEXTPNR=yowasp-nextpnr-ecp5`) are only for synthesis. Times are from an Apple M2 that was running other jobs.

```bash
make test        # 1,000 decisions, about 2 minutes; ends with ALL TESTS PASSED
make test-full   # all 4,000 decisions, about 11 minutes (what CI runs)
make cosim       # the RTL drives 20 pendulum episodes, about 8 to 12 minutes
make mutants     # 9 planted bugs must all FAIL, about 4 to 5 minutes
make sweep       # clocks for LANES 8/16/32 x H 5/15, about 2 to 3 minutes
make synth       # yosys resource report only, 32 lanes, about 15 s
make timing      # place and route on an ECP5-45F, 3 seeds, LANES 32 and 8, about 6 to 7 minutes
```

```
decision 0: state (-7, 10), best 30, cost 397, 1281 clocks
...
clocks in each FSM state, decision 0: INIT 1, L1 720, L2 480, COST 45, STORE 1, DRAIN 1, ARG 32, DONE 1
config: LANES = 32, H = 15, vectors from sim
decisions run: 4000
costs: 4000/4000 decisions bit-exact (all 32 candidates)
final states: 4000/4000 decisions bit-exact (all 32 theta and omega)
best_idx: 4000/4000 decisions match (best_cost too)
clocks per decision: 1281..1281 (cycles output matches the testbench count on 4000/4000)
busy/done protocol errors: 0
PASS
```

## Layout

```
DESIGN.md                 the spec: number formats, integer math, microarchitecture, and a changelog
STUDY.md                  reading order, a hand calculation and ten questions
rtl/rollout_lane.v        one lane: one candidate's state, h[0..15], one 8x8 multiplier, cost
rtl/rollout_engine.v      ROMs, action memory, LANES lanes, FSM, result buffer, argmin
tb/tb_rollout_engine.v    replays recorded decisions, bit-exact checks, watchdog
model/train_dynamics.py   trains the 3 -> 16 -> 2 dynamics model (NumPy, float32)
model/golden.py           folds it to int8, the integer golden model, rom/ and test vectors
model/planner.py          host side of the planner, shared by golden.py and the co-simulation
cosim/                    closed loop: Python pendulum <-> named pipes <-> Icarus
mutants/                  planted bugs (sed scripts) and the corner-case vector picker
perf/                     clock sweep and the clock formula check
synth/                    Yosys script and ECP5 place-and-route
rom/                      int8 weights, 24-bit biases and shifts for the hardware
results/                  every log the numbers above come from
```

## Series

Each project computes the same neural-network operation, the weighted sum, one step closer to a GPU:

1. [analog-neuron](https://github.com/Jusman0/analog-neuron): with resistors
2. [tinyml-mnist-uno](https://github.com/Jusman0/tinyml-mnist-uno): in int8 on an 8-bit CPU, one MAC at a time
3. [dp4a-unit](https://github.com/Jusman0/dp4a-unit): as dedicated hardware, four MACs per clock
4. [tiny-tensor-core](https://github.com/Jusman0/tiny-tensor-core): the whole network in hardware, 32 MACs per clock
5. **int8-rollout-engine**: the network inside a control loop, 32 lanes each imagining a different future

Built with an AI pair programmer. The design decisions and their reasons are in DESIGN.md.
