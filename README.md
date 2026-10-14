# int8-rollout-engine

[![tests](https://github.com/Jusman0/int8-rollout-engine/actions/workflows/test.yml/badge.svg)](https://github.com/Jusman0/int8-rollout-engine/actions/workflows/test.yml)

A 32-lane int8 Verilog engine that, every control tick, imagines 32 possible futures of a pendulum through a small learned physics model, scores each one, and hands back the best plan. It's sampling-based model-predictive control with the model rollouts in hardware, checked bit-exact against a Python golden model and in a closed loop where the RTL picks every action. Everything runs in simulation (Icarus Verilog), with synthesis estimates for an ECP5 FPGA. A CUDA port of the same integer math runs bit-exact on an RTX 3060 ([cuda/README.md](cuda/README.md)).

Board bring-up on a Tang Nano 20K: in progress, see [board/README.md](board/README.md). Nothing has run on hardware yet.

## Results

Every result here comes from a log in [`results/`](results/), and the command next to it reproduces the numbers in that log (the logs also carry notes and wall times added when they were made). Runs: Apple M2, Icarus Verilog 13.0, Python 3.14, NumPy 2.5.4, gymnasium 1.4.0 ([`machine.txt`](results/machine.txt)); the three CUDA rows ran on a Windows 11 PC with an RTX 3060, CuPy 14.2.0 and NVRTC 13.3 (versions at the top of each log in [`results/cuda/`](results/cuda/)). The DreamZero and GR00T figures further down come from the sources linked there, not from this repo.

| Result | Log | Reproduce |
|---|---|---|
| **Bit-exact replay**: for each of **4,000 / 4,000** recorded planner decisions, all 32 candidate costs, all 32 final states (θ, ω) and the chosen plan (best_idx, best_cost) match the golden model. 32 lanes, H = 15 imagined steps | [`test_full.log`](results/test_full.log) | `make test-full` |
| The same 4,000 / 4,000 at 16 and 8 lanes, and at 32, 16 and 8 lanes with H = 5 | [`config_sweep.log`](results/config_sweep.log) | `make test-full LANES=8`; for H = 5, `python3 model/golden.py --H 5 --out sim/h5` then `make test VEC=sim/h5 N=4000 LANES=8` |
| **Closed loop, the RTL in charge**: the RTL picks every action of 20 pendulum episodes. **20 / 20** episode returns identical to the golden model's (float `==`), **4,000 / 4,000** choices the same | [`cosim.log`](results/cosim.log) | `make cosim` |
| **Clocks per decision**: **1,281** at 32 lanes, H = 15. All 6 configurations (below) equal (32 / LANES)·(83·H + 2) + 34 exactly | [`sweep.log`](results/sweep.log), [`cycles.log`](results/cycles.log) | `make sweep` |
| **int8 vs float planner**: no measurable difference at H = 15, **+0.6** return (95% CI −7.7 to +9.6, 100 episodes). At H = 5, int8 is worse by 142 | [`planner_quality.txt`](results/planner_quality.txt) | `python3 model/golden.py --eval` |
| **Planted bugs**: **10 / 10** caught on the corner-case decisions (only 7 / 10 on the first 300 decisions, see below) | [`mutants.log`](results/mutants.log) | `make mutants` |
| **CUDA port, bit-exact**: on an RTX 3060, the 4,000 decisions `make test-full` replays and two more sets of 4,000 made on the PC (H = 15 and H = 5), each set in one kernel launch: every cost, final state and best index equal the golden model's; random decisions with up to 4,096 candidates too; in all, **918,528** candidate results compared (459,264 distinct candidates, each checked in two runs), **0** wrong. **11 / 11** planted kernel bugs caught | [`cuda/bitexact.log`](results/cuda/bitexact.log) | on the GPU machine: `python cuda/test_bitexact.py --vec sim --vec sim/mac --vec sim/h5` ([cuda/README.md](cuda/README.md)) |
| **CUDA closed loop, the GPU in charge**: the kernel picks every action of the 20 episodes. **20 / 20** returns identical (float `==`) to the golden model's on the same PC, **4,000 / 4,000** choices the same | [`cuda/closed_loop.log`](results/cuda/closed_loop.log) | `python cuda/closed_loop.py` |
| **CUDA timing** (provisional, see [cuda/README.md](cuda/README.md)): one decision (32 candidates, H = 15) is **4.25 µs** of GPU time (`rollout_warp`); with `rollout_blocks`, 5.21 µs at 32 and 5.87 µs at 4,096 candidates, 26.98 µs at 65,536; each imagined step adds about 0.20 µs. A round trip from Python, actions in and best index out, takes 62.95 µs on this Windows PC | [`cuda/bench.log`](results/cuda/bench.log) | `.venv\Scripts\python.exe cuda\bench.py --shared-gpu`, the logged command; the flag overrides the script's refusal to run while another Python process holds the GPU |
| **ECP5-45F, 32 lanes**: 13,804 LUT4s before packing (31%), 33 of 72 DSP multipliers (one per lane, plus one for a ROM address, see below). **47.49 MHz** post-route estimate, out-of-context, median of 3 seeds, so it **misses** its 50 MHz target | [`synth/summary.txt`](results/synth/summary.txt) | `make timing` |
| The learned model's one-step RMS error on held-out data: 0.108 LSB in θ, 0.484 LSB in ω | [`train_dynamics.log`](results/train_dynamics.log) | `python3 model/train_dynamics.py` |

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
| the lanes read the θ0, ω0 input ports during the decision, not the copies made at `start` | caught | caught |

The three misses aren't harmless bugs. The first 300 decisions never saturate ω, never clamp a hidden value and never have two candidates tied for the lowest cost, so those bugs compute exactly what the correct design computes on that data. The corner-case set takes all 71 of the 4,000 recorded decisions that hit one of those cases, plus the earliest others, and catches all ten. Two bugs (the missing 21/32 and the late ω′) leave every final state right: only the cost check, and the best_cost comparison, see them; the state check can't. That is why the testbench checks costs and states separately. The last bug is caught only because the testbench changes θ0 and ω0 while the engine is busy: with the inputs held steady, reading the ports gives the same numbers as reading the copies.

### Synthesis

From [`results/synth/`](results/synth/) (`make timing YOSYS=... NEXTPNR=...`; Yosys 0.69, nextpnr-ecp5 0.11.1, LFE5U-45F). All MHz and µs figures are **post-route estimates, out-of-context** (no pin constraints), from the median of 3 placement seeds. Nothing has been run on a board. Since these runs the RTL has only gained comments and simulation-only checks, and Yosys still builds the identical netlist ([`netlist_check.txt`](results/synth/netlist_check.txt)).

| | 32 lanes | 8 lanes |
|---|---|---|
| LUT4s (before packing) | 13,804 of 43,848 (31%) | 6,338 (14%) |
| DSP multipliers (MULT18X18D) | 33 of 72: one per lane, plus one (below) | 9 of 72 |
| Flip-flops | 3,751 | 2,025 |
| Post-route Fmax, seeds 1 / 2 / 3 | 47.49 / 48.94 / 47.26 MHz | 53.53 / 51.54 / 51.20 MHz |
| Median vs the 50 MHz target | **47.49 MHz, misses on every seed** | 51.54 MHz, meets on every seed |
| Clocks per decision (H = 15) | 1,281 | 5,022 |
| One decision at the median Fmax, start to done (not counting the 480 clocks of action loading) | 27.0 µs | 97.4 µs |

In all six runs the slowest path is the same single clock: a broadcast register (`ex_src` or `ex_j`) → the hidden-value read and the operand multiplexer → the lane's DSP multiplier → the 21/32 shift-and-add → the 24-bit cost adder. Its logic delay is 8.4 ns in every run, 3.93 ns of it the multiplier. The rest is routing: 12.1 to 12.8 ns at 32 lanes (about 60% of the path) and 10.3 to 11.1 ns at 8 lanes (55 to 57%). The hop out of the broadcast register is only part of that routing, 1.04 to 3.62 ns at 32 lanes and 1.15 to 2.25 ns at 8 lanes; the rest is spread over the other hops of the path. So these logs don't show that fanning one register out to 32 lanes is what makes the path slow. Finding out would take another run, for example with a copy of the broadcast register for each group of lanes. Registering the product before the cost add is the obvious next step. It would lengthen the lane pipeline, so the cost beats and write-backs would need re-timing; I haven't tried it, so I don't know what it gains or costs in clocks. (All figures: the "Critical path report for clock" section of each `nextpnr_*.log`.)

The 33rd DSP multiplier isn't in a lane. Yosys built it for `3 * j`, the W1 ROM address `3 * j + i` in `rtl/rollout_engine.v` ([`dsp_cells_lanes32.txt`](results/synth/dsp_cells_lanes32.txt), `make synth`). Multiplying a 4-bit counter by 3 only needs a shift and an add (`2j + j`); I've left the RTL as it is, so the numbers above stay the ones that were placed and routed.

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

**No requantization multiplier.** A typical int8 layer rescales with `acc · M >> shift`. tiny-tensor-core shares one 24×17 multiplier for that among all its neurons, which works because its 32 neurons finish layer 1 together and are requantized one per clock in a 32-clock phase. Here every lane finishes a hidden neuron every 3 clocks, so a shared requantizer would stall the lanes, and each lane would need a second multiplier of its own. Instead every scale factor is folded into the integer weights offline ([DESIGN.md §4.1](DESIGN.md)): each layer's weights are scaled up by a power of two, 2^S, and the weights absorb the rest of the scale, so requantizing is just add bias, arithmetic shift (by S), clamp. What the power of two costs is weight resolution: S is the largest shift that keeps every weight within ±127, so the largest weight can land anywhere from 64 to 127 (in layer 1 it's 76, [`results/golden.log`](results/golden.log)). The bias also carries +2^(S−1), which turns the floor shift into round-half-up for free. The write-back happens in the clock after a dot product's last MAC, while the multiplier already works on the next one, so it adds no clocks.

## The latency lesson

More lanes buy more candidates, not a shorter decision: going from 8 to 32 lanes cuts a decision from 5,022 to 1,281 clocks only because 8 lanes need four passes over the same 32 candidates. Inside one candidate, step t + 1 needs the state from step t, so its H imagined steps run one after another at 83 clocks each, a floor of 1,245 clocks at H = 15 that the 32-lane engine is only 36 clocks above ([`results/cycles.log`](results/cycles.log)). What's left is fewer imagined steps (H = 5 takes 451 clocks, but plans badly, see above) or a cheaper step, which means more multipliers working on one candidate. The CUDA port shows the same shape on a GPU: 5.21 µs for 32 candidates, 5.87 µs for 4,096 (`rollout_blocks`, including the best_key fill kernel), and about 0.20 µs more for every imagined step ([`results/cuda/bench.log`](results/cuda/bench.log), provisional). [DreamZero](https://arxiv.org/abs/2602.15922) shows the same pressure at scale: its latency is denoising steps × one forward pass of a 14-billion-parameter model, and it cut 16 steps to an effective 4 and then to 1 on the way from about 5.7 s to 150 ms per action chunk.

## CUDA port

[`cuda/`](cuda/) runs the same integer rollouts on an NVIDIA GPU, as a port of the RTL's ideas: one thread per candidate (a lane), one warp of 32 threads per decision (the 32 lanes in lockstep), the weights in `__constant__` memory so that one read serves the whole warp (the weight broadcast), `__dp4a` for four int8 multiply-adds in one instruction, and a five-round warp-shuffle argmin instead of the 32-clock scan. The kernels are CUDA C++ compiled at run time by NVRTC through CuPy, so the Windows PC they ran on needed no CUDA Toolkit. They match the golden model bit for bit on every recorded decision and in the closed loop (the CUDA rows above). The RTL-to-CUDA mapping, the timing and what it doesn't show are in [cuda/README.md](cuda/README.md); the study questions are in [STUDY.md](STUDY.md#5-cuda-port).

## Why this connects to robotics

In his talk at [Sequoia AI Ascent 2026](https://sequoiacap.com/article/ai-ascent-2026), Jim Fan argued that robots will act by imagining the future with learned world models. That makes control an inference-latency problem. How the cost adds up depends on the method. In sampling-based MPC, this project's method, every decision costs candidates × imagined steps × one forward pass, and the imagined steps can't run in parallel. DreamZero samples no candidates: its cost is denoising steps × one forward pass (above). For a sense of what one robot-model inference costs today: NVIDIA's GR00T N1.7, a vision-language-action model, takes 39.9 ms per inference on a Jetson AGX Thor (MAXN power mode, 4 denoising steps, 1 camera, batch 1, TensorRT with mixed NVFP4 and FP8 layers), per the [Jetson AI Lab tutorial](https://www.jetson-ai-lab.com/tutorials/groot_n17_on_thor/). This project is a sense-plan-act loop with a learned model inside, shrunk to a 98-parameter model and a 2-number state, small enough to check every bit. Its clock counts and MHz estimates say nothing about real robot models and aren't comparable to GR00T's numbers.

## Limitations

- **A toy task.** The state is two numbers, the model is 3 → 16 → 2 (98 parameters), and the task is gymnasium's Pendulum-v1.
- **This task doesn't need the hardware's speed.** Pendulum-v1 advances 50 ms per step, and the NumPy golden model already makes a decision in about 0.6 ms on the same M2, while it was busy with other simulations ([`results/cosim.log`](results/cosim.log): 4,000 decisions in 2.5 s).
- **Simulation only.** The RTL runs in Icarus Verilog. There's no FPGA board; the Fmax figures are post-route estimates, out-of-context, and the 32-lane build misses its 50 MHz target.
- **What int8 costs.** At H = 15 nothing measurable (+0.6, 95% CI −7.7 to +9.6). At H = 5 the int8 planner is 142 worse than the float one, for a reason I haven't found.
- **The cost is approximated.** The real cost in integer units has weights 0.6587 on ω² and 0.000412 on u². The hardware uses 21/32 = 0.65625 and 1/2048 = 0.000488, so that they're shifts and adds, and each term is rounded down.
- **One trained model**, seed 0. The planted bugs were run at 32 lanes and H = 15 only, and the `cycles` output leaves out loading the actions (above).
- **The CUDA port ran on one GPU**, an RTX 3060 under Windows (WDDM), timed with CUDA events only: no profiler, the machine code not disassembled, clocks not locked. The benchmark is provisional: it ran while another program's Python process held a GPU context (a process that was idle when the run started), with `--shared-gpu` overriding the script's own refusal ([cuda/README.md](cuda/README.md)).
- CI (`.github/workflows/test.yml`) runs `make golden && make test-full` on Ubuntu with its Icarus Verilog package, and passed on the first push. The logs in `results/` are from Icarus 13.0 on macOS.

## Run it

Needs [Icarus Verilog](https://steveicarus.github.io/iverilog/) (`brew install icarus-verilog`) and Python 3 (3.14 was used) with the packages in [`requirements.txt`](requirements.txt) (`pip install -r requirements.txt`; if they're in a virtualenv, add `PYTHON=path/to/venv/bin/python` to the make command). Yosys and nextpnr-ecp5 ([OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build), or the YoWASP pip packages with `YOSYS=yowasp-yosys NEXTPNR=yowasp-nextpnr-ecp5`) are only for synthesis.

```bash
make test        # the first 1,000 decisions, a quick check; ends with ALL TESTS PASSED
make test-full   # all 4,000 decisions (what CI runs)
make cosim       # the RTL drives 20 pendulum episodes
make mutants     # 10 planted bugs must all FAIL
make sweep       # clocks for LANES 8/16/32 x H 5/15
make synth       # yosys resource report and the DSP-cell listing, 32 lanes
make timing      # place and route on an ECP5-45F, 3 seeds, LANES 32 and 8
```

The CUDA port needs an NVIDIA GPU and a virtualenv with CuPy; its setup and commands are in [cuda/README.md](cuda/README.md).

Wall times, as recorded in the logs: `make test` 273 s, `make test-full` 625 s, `make cosim` 773 s, `make mutants` 456 s, `make sweep` 223 s. They were measured with all of these running at once on the M2, so a run on its own is faster. `make synth` and `make timing` weren't timed in a log.

`make test` is only a quick check: its 1,000 decisions never saturate ω and never clamp a hidden value ([`results/coverage.log`](results/coverage.log)). Those cases are in `make test-full` (20 and 32 of its 4,000 decisions) and in the corner-case set of `make mutants`.

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
cuda/                     CUDA port: kernels (compiled by NVRTC through CuPy), bit-exact test, closed loop, benchmark
mutants/                  planted bugs (sed scripts) and the corner-case vector picker
perf/                     clock sweep and the clock formula check
synth/                    Yosys scripts (resources, DSP-cell listing) and ECP5 place-and-route
rom/                      int8 weights, 24-bit biases and shifts for the hardware
results/                  every log the numbers above come from
```

## Series

Each project computes the same neural-network operation, the weighted sum, one step closer to a GPU:

1. [analog-neuron](https://github.com/Jusman0/analog-neuron): with resistors
2. [tinyml-mnist-uno](https://github.com/Jusman0/tinyml-mnist-uno): in int8 on an 8-bit CPU, one MAC at a time
3. [dp4a-unit](https://github.com/Jusman0/dp4a-unit): as dedicated hardware, four MACs per clock
4. [tiny-tensor-core](https://github.com/Jusman0/tiny-tensor-core): the whole network in hardware, 32 MACs per clock
5. **int8-rollout-engine**: the network inside a control loop, 32 lanes each imagining a different future, and in [`cuda/`](cuda/) the same rollouts on a GPU, with `__dp4a` doing what dp4a-unit does in hardware

Built with an AI pair programmer. The design decisions and their reasons are in DESIGN.md.
