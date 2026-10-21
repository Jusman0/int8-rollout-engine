# int8-rollout-engine: design spec (v1)

A 32-lane int8 Verilog engine that, every control tick, **imagines 32 candidate futures** of a pendulum through a
small learned dynamics model, scores each future, and returns the best one. Sampling-based model-predictive control
(MPC), with the model rollouts in hardware.

This document is the single source of truth. The Python golden model (`model/`) and the RTL (`rtl/`) both implement
exactly what is written here, and the testbench checks them against each other bit-for-bit.

## 0. Why this exists (the one-paragraph story)

In a 2026 talk at Sequoia AI Ascent, Jim Fan argued that robots will act by imagining the future with learned
world models (NVIDIA's DreamZero, arXiv 2602.15922, predicts future video frames together with the actions to take).
For sampling-based model-predictive control, this project's method, the hardware cost of imagining is
**candidates × imagined steps × one forward pass**, and the imagined steps are *sequential*: step t+1 needs the
state from step t. (DreamZero's cost has the same sequential shape without the candidates: denoising steps × one
forward pass.) This project measures that cost on a toy you can hold in your head: a 2-number state
(angle, angular velocity), a 3→16→2 network, 32 candidates, 15 steps. It is **not** a world action model. It is RTL verified in
simulation; an 8-lane build also runs bit-exact on a Tang Nano 20K FPGA (Stage 7).

## 1. The task: gymnasium `Pendulum-v1`

- Observation `[cos θ, sin θ, ω]`, action torque `τ ∈ [-2, 2]`, `dt = 0.05 s`, 200 steps per episode.
- Reward per step `-(θ² + 0.1·ω² + 0.001·τ²)`, θ normalized to `[-π, π)`. Higher return is better; a good controller
  scores around −150, a bad one −1000 or worse.

## 2. Number formats (all int8)

| Quantity | Stored as | 1 LSB means | Range | Overflow rule |
|---|---|---|---|---|
| angle θ | `theta_q` signed 8-bit | π/128 rad | −π … +π·127/128 | **wraps** (two's-complement overflow from +127 to −128 *is* angle wrap from +π to −π) |
| angular velocity ω | `omega_q` signed 8-bit | 8/127 rad/s | −8 … +8 rad/s | **saturates** to [−127, +127] (wrapping a velocity would flip its sign) |
| action u | `u_q` signed 8-bit | 1/127 | −1 … +1 | host only produces [−127, +127]; torque τ = 2·u |
| hidden activation h | unsigned, stored 8-bit | s_h (calibrated) | 0 … 127 | clamped to [0, 127] (ReLU + saturate) |

Measured state → int8 (host side):
- `theta_q = ((round(θ / (π/128)) + 128) mod 256) − 128`
- `omega_q = clip(round(ω / (8/127)), −127, 127)`

## 3. The learned dynamics model

- Float MLP: inputs `(θ, ω, u)` → 16 ReLU → 2 outputs, predicting the state change `(Δθ, Δω)` over one step.
- Trained in NumPy (fp32), fixed seeds, on 30,000 random-action transitions (episodes reset every 25 steps). Inputs
  and targets are standardized (`xm, xs, ym, ys`); `Δθ` is wrapped to `[-π, π)` before training.

## 4. Integer inference: what the hardware computes (multiplier-free requantization)

One imagined step for one candidate, given `theta_q, omega_q` and that step's action `u_q`:

```
x = (theta_q, omega_q, u_q)                                   # three signed int8 inputs

for j in 0..15:                                               # layer 1
    acc1_j = B1[j] + Σ_i W1[j][i] · x[i]                      # W1 signed int8, B1 wide int
    h[j]   = clamp(acc1_j >>> S1, 0, 127)                     # arithmetic shift right = floor

for k in 0..1:                                                # layer 2
    acc2_k = B2[k] + Σ_j W2[k][j] · h[j]                      # W2 signed int8
    d[k]   = acc2_k >>> S2[k]

theta_q' = low 8 bits of (theta_q + d[0])                     # wrap
omega_q' = clamp(omega_q + d[1], −127, 127)                   # saturate

c_step   = theta_q'² + ((21 · omega_q'²) >> 5) + (u_q² >> 11)  # unsigned
```

**Why no requantization multipliers:** a typical int8 network needs `acc · M >> shift` per output
(tiny-tensor-core uses a 24×17 multiply). Here every scale factor is **folded into the integer weights offline**, so
requantization is just *add bias, arithmetic shift, clamp*. That saves one multiplier per lane (32 total). The cost:
the scale applied to each layer's weights is a power of two, 2^S, and the weights absorb the rest of the scale. S is
the largest shift that keeps every weight within ±127 (§4.1), so the largest weight lands between 64 and 127 rather
than exactly at 127: the power of two can cost up to one bit of weight resolution. The approximations are rounding
the weights and biases to integers, rounding hidden values and state changes to whole LSBs (round-half-up, from the
+2^(S−1) in each bias), and clamping h at 127; §4.2 covers the cost.

### 4.1 Folding the scales (offline, in `model/golden.py`)

With input LSBs `L = (π/128, 8/127, 1/127)` and the float model `a = W1·((L∘x_q − xm)/xs) + b1`:

- Layer 1 real weights `W1'[j][i] = W1[j][i] · L[i] / xs[i] / s_h`, real bias `b1'[j] = (b1[j] − Σ_i W1[j][i]·xm[i]/xs[i]) / s_h`.
  `s_h` = (largest pre-activation seen on the training transitions) / 127.
- `S1` = the largest integer with `max|W1'| · 2^S1 ≤ 127`. Then `W1 = round(W1' · 2^S1)` and
  `B1 = round(b1' · 2^S1) + 2^(S1−1)`. The `+2^(S1−1)` turns the floor-shift into round-half-up.
- Layer 2, per output k with state LSB `Lk ∈ {π/128, 8/127}`: `W2'[k][j] = ys[k]·W2[k][j]·s_h / Lk`,
  `b2'[k] = (ys[k]·b2[k] + ym[k]) / Lk`. `S2[k]` = largest integer with `max_j |W2'[k][j]| · 2^S2[k] ≤ 127`.
  `W2 = round(W2' · 2^S2[k])` and `B2 = round(b2' · 2^S2[k]) + 2^(S2[k]−1)`.
- The golden model **asserts** `S1, S2[k] ≥ 1` and that every accumulator fits the RTL width (|acc| < 2^(ACC_W−1)).

### 4.2 The cost, in integer units

The real cost `θ² + 0.1ω² + 0.004u²` divided by `(π/128)²` gives `θ_q² + 0.6587·ω_q² + 0.000412·u_q²`.
Approximations: `0.6587 ≈ 21/32 = 0.65625` and `0.000412 ≈ 1/2048 = 0.000488`. These are shift-and-add constants, so no
multiplier is needed for the coefficients. Max per step: 16384 + 10584 + 7 = 26,975; over 15 steps: 404,625 < 2^19, so a
**24-bit unsigned cost accumulator** has plenty of headroom.

## 5. The planner (host ↔ engine split)

Every control tick:
1. **Host** builds N = 32 candidate plans of H = 15 int8 actions.
   - First tick of an episode: uniform random in [−127, 127].
   - Afterwards: candidate 0 is last tick's best plan shifted left by one step, with its last action repeated.
     Candidates 1–31 are candidate 0 plus Gaussian noise (σ = 0.5 in u units), clipped to [−1, 1] and quantized
     to /127. The host RNG is NumPy `default_rng`, seeded per episode, so every run is reproducible.
2. **Engine** rolls all 32 candidates forward H steps from the measured `(theta_q, omega_q)`, accumulating
   `cost[n] = Σ_t c_step`. It returns all costs plus `best = argmin cost`, taking the lowest index on ties
   (strict `<` while scanning n = 0..31).
3. **Host** applies torque `τ = 2·u_q[best][0]/127` and remembers `plan[best]` for the next tick's warm start.

Re-planning from the *real* measured state every tick is what stops the imagined errors from compounding. DreamZero
does the same thing when it swaps its dreamed frames for real camera frames.

## 6. Microarchitecture (RTL)

- **Lane-per-candidate SIMD.** LANES = 32 lanes. Each lane owns one candidate's state `(theta, omega)`, a 16-entry
  hidden register file, one signed 8×8 multiplier, an `ACC_W = 24` accumulator, and a 24-bit cost accumulator.
- **Weight broadcast.** Each clock the sequencer reads one int8 weight from ROM and broadcasts it to all lanes. Each
  lane multiplies it by *its own* operand (θ, ω, u, or h[j]). One weight fetch feeds 32 MACs. That is the same reason
  a GPU warp shares one instruction across 32 threads.
- **Action memory.** `ACT[t]` is one 32×8 = 256-bit word per imagined step t, so each lane reads its own action in
  one access. Host writes it byte by byte before `start`.
- **Requantize in-line.** When a hidden neuron's three MACs finish, the lane adds/shifts/clamps and writes `h[j]`
  while the next neuron's MACs start. This costs zero extra clocks.
- **Cost.** After the state update, the lane's multiplier computes θ'², ω'², u² (operand muxed to the lane's own
  value). The constants 21/32 and 1/2048 are shift-and-add.
- **Argmin.** After the last step, a scan over the 32 lane costs with strict `<` returns `best_idx` and `best_cost`.
- **LANES < 32** (8 or 16) is supported by running ⌈32/LANES⌉ passes; costs are stored and the argmin spans
  all passes.

Expected cycle count (measured value goes in the README): per imagined step ≈ 48 (layer 1) + 32 (layer 2) + 3
(cost) + a few overhead clocks, so per decision ≈ ⌈32/LANES⌉ · H · ~85 + 32 (argmin) + overhead.

**The key latency fact:** more lanes buy more *candidates*, not a shorter decision. The H imagined steps form a
sequential chain, because step t+1 needs step t's state. To cut latency you must cut H, or make each step cheaper
by putting more multipliers on one candidate's math. That is the same pressure that made DreamZero cut its denoising
steps from 16 to 4 to 1.

### Ports (top module `rollout_engine`)
```
clk, rst
act_we, act_t[3:0], act_n[4:0], act_data[7:0]        host writes ACT[t][n]
theta0[7:0], omega0[7:0]                              measured state (signed)
start → busy, done (1-clock pulse)
best_idx[4:0], best_cost[23:0], cycles[15:0]
dbg_n[4:0] → dbg_cost[23:0], dbg_theta[7:0], dbg_omega[7:0]   readback for verification
```

## 7. Verification plan

1. **Golden model** (`model/golden.py`) implements §4–5 in pure-integer Python and runs closed-loop episodes. It
   writes `rom/*.hex`, `rom/params.vh`, and decision vectors (inputs, all 32 costs, final lane states, best index).
2. **Bit-exact replay** (`make test`): the testbench feeds recorded decisions (the first 1,000; `make test-full` runs all 4,000) through the RTL and compares
   all 32 costs, all 32 final states, and `best_idx`. It prints clocks per decision. A watchdog turns a hung FSM
   into a FAIL.
3. **Closed-loop co-simulation** (`make cosim`): Python runs the pendulum physics and the host planner, and
   Icarus runs the RTL, connected through named pipes. The *RTL's* chosen action drives the pendulum. Episode
   returns must equal the golden model's exactly.
4. **Planted bugs** (`make mutants`): saturate θ instead of wrapping; wrap ω instead of saturating; tie-break with
   `<=`; logical instead of arithmetic shift; clamp h at 255. Every one must FAIL.
5. **Measurements:** clocks per decision for LANES ∈ {8, 16, 32} × H ∈ {5, 15}, compared with the formula. Closed-loop
   return for fp32 vs int8 planners and H = 5 vs 15. ECP5-45F synthesis resources and post-route Fmax (3 seeds).

## 8. Claims policy

The README states only measured results, each with the command that reproduces it. It never says: world action
model, DreamZero on hardware, real-time robot, or faster than a GPU/Jetson. It says "runs on an FPGA" and
"hardware-in-the-loop" only for what `board/results/hardware/` logs (since 2026-10-14: the 8-lane `-nodsp` build
on a Tang Nano 20K, at 27 MHz). It also never gives an MHz/µs figure without committed synthesis logs, and it says "post-route estimate,
out-of-context" for any that are given.

*Built with an AI pair programmer. The design decisions and their reasons are documented here so they can be checked
and explained.*

## Changelog

- **Stage 1, Python golden model.** No rule in §1–§5 was changed. These are the details the spec left
  open, written down so the RTL, testbench and co-simulation follow the same choices:
  - **Training (§3):** seed 0, 60 epochs, Adam (lr 1e-3), batch 256, He init. Everything stays float32 (the
    feasibility prototype had silently promoted its weights to float64). `model/dynamics_fp32.npz` also stores
    `a1_max`, the largest layer-1 pre-activation over the 30,000 training inputs, so §4.1's `s_h = a1_max / 127`.
  - **Rounding (§2, §4.1, §5):** `round()` / `np.round`, which send an exact .5 to the even neighbour. Measured floats
    practically never land on .5, and every host imports `model/planner.py`, so all hosts agree.
  - **Planner (§5):** the first tick uses `rng.integers(-127, 128)`; later ticks draw (N−1)×H standard normals for
    candidates 1–31 (candidate 0 gets no noise). Seeds (from the prototype): `env.reset(seed = 7000 + 100·seed +
    episode)` and `default_rng(10000 + 100·seed + episode)`.
  - **fp32 reference planner (§7.5):** the float32 model, run from the unquantized measured state, with θ wrapped to
    [−π, π), ω not clipped, and the float cost θ² + 0.1ω² + 0.004u². It uses the same int8 candidate plans and the
    same strict-`<` argmin as the integer planner.
  - **ROM layout (§6, §7.1):** one int8 weight per line, in broadcast order (`w1.hex` address 3j + i, `w2.hex`
    address 16k + j). Biases are 24-bit two's complement. The shifts are in `rom/params.vh` as `S1`, `S2_0` and
    `S2_1`. The decision-vector format is documented at the top of `model/golden.py`.
- **Stage 2, RTL and bit-exact testbench.** No rule in §1–§5 was changed and `model/golden.py` was not
  touched. One layout change to §6 (the action memory when LANES < 32, below); the rest are details §6 left open:
  - **Schedule (§6):** an imagined step is exactly 48 (L1) + 32 (L2) + 3 (COST) = 83 clocks, with no overhead clocks.
    A lane writes a finished dot product back one clock after its last MAC, reading the accumulator while its
    multiplier already does the next beat: h[j] during neuron j+1's first MAC (h[15] during layer 2's first MAC),
    θ′ during d_ω's first MAC, ω′ during the θ′² clock. The cost terms go into the cost accumulator one per clock
    (θ′², then (21·ω′²)>>5, then u²>>11). That is the same total as c_step, because each term is floored on its own.
  - **Pipeline:** FSM issue → registered broadcast (execute) → write-back, one clock apart. The ROMs and the action
    memory are read synchronously (address this clock, data next clock).
  - **Clock count:** `cycles` counts every clock outside IDLE, i.e. from the edge that sees `start` (exclusive) to the
    edge that raises `done` (inclusive): PASSES·(1 INIT + 83·H + 1 STORE) + 1 DRAIN + 32 ARG + 1 DONE
    = PASSES·(83·H + 2) + 34, where PASSES = 32 / LANES. DRAIN is the one clock the pipeline costs: the last STORE
    lands one clock after the FSM issues it, and the argmin must not read the buffer before that.
  - **Action memory when LANES < 32 (changes §6):** §6 describes one 256-bit word per step. The RTL uses words of
    LANES×8 bits instead. Word 16·p + t holds candidates p·LANES … p·LANES + LANES − 1 at step t, so each pass reads
    one word exactly as wide as its lanes. With LANES = 32 this is §6's ACT[t]. The host port is unchanged.
  - **Result buffer:** after every pass, the lanes' costs and final (θ, ω) are copied into a 32-entry buffer. This also
    happens when LANES = 32, so there is one code path. It costs 32 × 40 flip-flops. The argmin scan and the dbg port
    read only this buffer. `dbg_*` are combinational reads. `best_idx`, `best_cost`, `cycles` and `dbg_*` are valid
    from `done` until the next `start`.
  - **Inputs and parameters:** `theta0`/`omega0` are copied when `start` is seen. ACT may only be written while
    `busy` is low. LANES must divide 32 and 1 ≤ H ≤ 16; a simulation-time check enforces both. Besides LANES, N_CAND,
    H and ACC_W, the engine takes W1_FILE, B1_FILE, W2_FILE and B2_FILE (ROM paths, as in tiny-tensor-core).
  - **Testbench (§7.2):** `make test VEC=<dir> LANES=<n>` picks the vector directory (the testbench includes
    `<dir>/vectors.vh` for VEC_COUNT and VEC_H) and the lane count (`iverilog -P`). Each (VEC, LANES) pair gets its
    own compiled testbench. Besides the 32 costs, the 32 final states and best_idx/best_cost, the testbench checks:
    `cycles` against its own clock count, that `busy` rises with `start` and falls with `done`, and that `done`
    lasts one clock. A 10,000-clock watchdog turns a hung FSM into a FAIL.
- **Stage 3, closed-loop co-simulation.** No rule in §1–§6 was changed; `rtl/`, `tb/`, `model/` and the
  Makefile were not touched. These are the details §7.3 left open:
  - **Files:** `cosim/host.py` (Python: the pendulum, the host planner from `model/planner.py`, the golden
    reference), `cosim/tb_cosim.v` (Icarus: `rollout_engine` behind two named pipes) and `cosim/run.sh` (compiles,
    makes the pipes with `mkfifo`, starts the simulator then the host, and always deletes the pipes and stops the
    simulator when it ends). The compiled testbench, the pipes and the run's log live in `sim/cosim/` (not
    committed). `make cosim` runs `PYTHON=$(PYTHON) LANES=$(LANES) bash cosim/run.sh`.
  - **Pipe protocol:** `to_rtl` carries host → RTL and `from_rtl` carries RTL → host, as lines of signed decimal
    numbers. Both programs open `to_rtl` first and `from_rtl` second, so opening cannot deadlock. The testbench
    first sends `LANES H`, and the host stops if H differs from its own. Each tick the host sends
    `1 theta_q omega_q` and then the 32 × H actions in ACT[t][n] order (step outer, candidate inner). The testbench
    writes them through the normal `act_*` port, one byte per clock as `tb_rollout_engine` does, pulses `start` and
    answers `best_idx best_cost cycles`. `0` ends the run. No `$fscanf` format ends in whitespace: a trailing `"\n"`
    makes `$fscanf` wait for the next tick's first character, which the host only sends after this tick's answer
    (tried in a scratch test: it hangs).
  - **Reference and checks:** the reference is the golden model's integer-planner closed loop, run inside
    `host.py` with `golden.int_engine` and `planner.run_episode`, exactly as `golden.py` runs it. PASS needs all
    of: every episode return equal with float `==`, the RTL's choice equal to golden's at every tick, and every RTL
    answer (best_idx and best_cost) equal to the golden model's on the same inputs. The last check names the first
    tick that differs.
  - **Hangs:** the host gives the simulator 60 s to connect and 60 s per tick (`SIGALRM`), then fails. The
    testbench keeps the 10,000-clock watchdog and stops with FAIL if the host's input is short or the pipe closes.
- **Stage 3, planted-bug tests.** No rule in §1–§6 was changed; `rtl/`, `tb/`, `model/` and the
  Makefile were not touched. The unmodified RTL passes and all 9 planted bugs of that run FAILED (a 10th was added after review; `results/mutants.log` now holds the 10-bug run). These
  are the details §7.4 left open, and one deviation:
  - **Mutants:** §7.4's five, the dropped 21/32 weight on the ω cost, and three slips of my own: the step loop
    stops one imagined step early (fencepost), layer-2 dot products never restart from the bias (accumulator not
    cleared), and ω′ is written back one clock late (a read-after-write hazard: the ω′² cost term squares the old
    ω, while every final state stays right). "θ saturates" clamps to ±127, the ω clamp copied over; "logical
    shift" changes all three requantization shifts. Each mutant is one sed script, `mutants/NN_name.sed`, applied
    to a copy of `rtl/` in `sim/mutants/<name>/` and compiled with the unchanged testbench. A script that changes
    nothing, or a copy that does not compile, is reported as an error, never as a caught bug.
  - **Which decisions (deviation from "run the first N"):** each mutant runs N = 300 decisions from two sets. On
    the first 300 recorded decisions, 3 mutants PASS (ω wraps, h clamps at 255, tie-break `<=`): those decisions
    never saturate ω, never clamp an h and never tie, so the mutated logic computes exactly what the correct logic
    does. The set that decides the result is 300 decisions picked by `mutants/pick_corner_vectors.py` from the
    4,000 in `sim/`: all 71 that clamp an h, saturate ω or tie, then the earliest others. The lines are copied
    unchanged, so the expected values are still `golden.py`'s. The first-300 results stay in the log to show why.
  - **Pass rule:** the original RTL must PASS on both sets and every mutant must FAIL on the corner set. The
    command is `PYTHON=$(PYTHON) bash mutants/run_mutants.sh`; `make mutants` runs the
    same command.
- **Stage 3, cycle sweep.** No rule in §1–§6 was changed. `perf/sweep.sh` runs the unchanged
  testbench (`make test LANES=… VEC=… N=200`) for LANES ∈ {32, 16, 8} × H ∈ {15, 5}, on 200 decisions per
  configuration (episode 0, seed 0) made by `golden.py --count 200 --out sim/perf/h15` and `--H 5 … sim/perf/h5`.
  golden.py rewrites `rom/` each time; the script checksums `rom/` and stops if the contents changed.
  `perf/cycles.py` derives `cycles = (32/LANES)·(83H + 2) + 34` state by state from the FSM and checks it against the
  measured totals and the per-state counts of decision 0: all six match exactly (`results/cycles.log`). These
  200-decision sets contain no h clamp, ω saturation or θ wrap (the H = 15 set has no tie either), so they measure
  clocks, not corner-case coverage; that comes from the 4,000-decision runs and the planted-bug tests.
- **Stage 3, ECP5 synthesis (2026-10-12).** No rule in §1–§6 was changed and the RTL was not touched. Yosys 0.69
  and nextpnr-ecp5 0.11.1 (YoWASP builds), LFE5U-45F in the CABGA381 package, LANES = 32 and 8. "Out-of-context"
  means no pin-constraint (.lpf) file; nextpnr's experimental `--out-of-context` flag was not used, because it also
  turns off the global clock network. `--freq 50 --timing-allow-fail`, seeds 1–3; the post-route "Max frequency"
  line (the last one in each log) is the one reported, and the median of the three is the headline. Committed
  nextpnr logs are trimmed to utilisation plus everything after routing (`synth/timing_ecp5.sh`). Result
  (`results/synth/summary.txt`): LANES = 32 misses 50 MHz on all three seeds (median 47.49 MHz); LANES = 8 meets
  it on all three (median 51.54 MHz). The critical path runs from a broadcast register through the operand mux,
  the unregistered multiplier and the 21/32 shift-and-add into the 24-bit cost adder. Not fixed here.
- **Stage 4, integration and documentation.** No rule in §1–§7 was changed; `rtl/`, `tb/` and `model/`
  were not touched.
  - **Makefile:** targets `cosim`, `mutants`, `sweep`, `synth` and `timing` run each stage's script with the command
    that stage used. `YOSYS` and `NEXTPNR` pick the
    tool names. `cosim`, `mutants` and `sweep` depend on `rom/params.vh`, so the ROM matches the model that the
    Python side folds. Each target was run once at commit c188ae0 and reproduced its committed log: the same 20
    co-simulation returns, the same mutant table, the same sweep counts (`results/cycles.log` byte-identical), the
    same `stat_lanes32.txt`, and nextpnr logs identical except the "Router1 time" lines. The logs from the stage-3
    runs were kept.
  - **CI:** `.github/workflows/test.yml` runs `make golden && make test-full` on ubuntu-latest with Python 3.14 and
    Ubuntu's `iverilog` package, which may be older than the Icarus 13.0 behind every log. `make mutants` is left out,
    to keep CI to the one bit-exact check, and because its sed scripts and parallel runner have only been run on
    macOS (BSD sed), never on Ubuntu's GNU sed. (Corrected after review: this bullet first gave a ~3 minute time
    budget as the reason, which `make test-full`, at about 10 minutes (`results/test_full.log`), already exceeds.) CI has not run yet; nothing
    is pushed.
  - **requirements.txt** pins numpy 2.5.4 and gymnasium 1.4.0. cocotb is not used (the co-simulation uses named
    pipes).
  - **README:** §0 named the talk "Robotics: the Endgame"; that published title could not be verified, so the
    README says "his talk at Sequoia AI Ascent 2026" (§0 now says the same). Figures that are only in stage reports,
    not in a committed log, are left out of the README: the "about 8 clocks per step with 16 multipliers per lane"
    estimate (never built) and the area savings tried in a scratch copy of the RTL.
  - **External figures (deviation from §8, "the README states only measured results"; added after review):** the
    README's latency and robotics sections cite figures that were not measured here, as context: DreamZero
    (arXiv 2602.15922) cutting 16 denoising steps to an effective 4 and then 1, and going from about 5.7 s to 150 ms
    per action chunk; GR00T N1.7 taking 39.9 ms per inference on a Jetson AGX Thor (Jetson AI Lab tutorial). Each is
    sourced inline with a link, and the README's Results section says these come from the linked sources, not from
    this repo.
- **Review fixes.** No rule in §1–§5 was changed. §0 and §4 were reworded (below); the synthesizable
  RTL is unchanged: Yosys builds the same netlist as before, byte for byte, at LANES = 32 and 8
  (`results/synth/netlist_check.txt`), so the place-and-route logs still apply and were not re-run.
  - **§0:** the quote "dreams a couple of seconds ahead and acts accordingly" was not from any source in the
    research notes, and the talk title "Robotics: the Endgame" could not be verified; §0 now names the event
    (Sequoia AI Ascent 2026) and cites DreamZero by its arXiv number. It also says the candidates × imagined steps ×
    forward pass cost is specific to sampling-based MPC; DreamZero's is denoising steps × forward pass.
  - **§4:** "the only approximation is rounding the weights themselves to int8" was too narrow. Hidden values and
    state changes are also rounded to whole LSBs and h clamps at 127. The power-of-two scale itself costs weight
    resolution: the largest weight lands between 64 and 127 (76 in layer 1, `results/golden.log`).
  - **Testbench (§7.2):** while busy is high, the testbench drives `theta0`/`omega0` to the bitwise inverse of the
    measured state, so a design that reads the ports instead of the copies made at `start` fails. It prints FAIL
    when `+n` asks for 0 decisions or more than the vector directory holds (before, it ran the decisions it had and
    could print PASS). The decision printout shows the state from the vectors, not the (changed) ports.
  - **Planted bug 10 (§7.4):** the lanes read the live `theta0`/`omega0` ports. It FAILs on both sets; all 10 are
    caught on the corner set, 7 of 10 on the first 300 (`results/mutants.log`).
  - **RTL, simulation only:** the parameter check now accepts only LANES = 8, 16 or 32 (the stage-2 rule was "LANES
    must divide 32"; LANES = 4 at H = 15 needs 10,010 clocks, more than the 10,000-clock watchdog). A new check prints
    an error when `act_we` is high while `busy` is high. The hardware still does not block such a write; enforcing
    "ACT may only be written while busy is low" stays the host's job. Both checks sit between `synthesis
    translate_off` and `translate_on`.
  - **Synthesis evidence:** `synth/dsp_cells.ys` (run by `make synth`) lists the source of every DSP multiplier
    (`results/synth/dsp_cells_lanes32.txt`): 32 are the lanes' multipliers and the 33rd is `3 * j` in the W1 ROM
    address. The RTL was not changed to remove it, so the logged resource and timing numbers stay valid.
  - **Logs re-run** with the new testbench, at commit 56fb632: `make test`, `make test-full`, the configuration
    sweep, `make sweep`, `make mutants` and `make cosim` (LANES = 32, and LANES = 8 for one episode), all PASS.
    `results/cycles.log` came out byte-identical. New logs: `results/coverage.log` (the corner cases in the first
    1,000 decisions that `make test` replays: no ω saturation, no h clamp, 6 ties), `results/machine.txt`.
  - **Stage-4 bullets corrected:** the CI bullet now gives the real reason `make mutants` is not in CI, and a new
    bullet records the README's external figures (DreamZero, GR00T) as a deviation from §8.
- **CUDA port (2026-10-13).** No rule in §1–§7 was changed; `rtl/`, `tb/`, `model/` and the Makefile were not
  touched. New: `cuda/` (kernels, host engine, tests, benchmark, `cuda/README.md`) and `results/cuda/`. These are
  the details §4–§6 leave open on a GPU:
  - **Toolchain:** CUDA C++ compiled at run time by NVRTC 13.3 through `cupy.RawModule` (CuPy 14.2.0), with
    `--std=c++20`, so `>>` on a negative `int` is guaranteed arithmetic. NVRTC emits sm_86 machine code (a cubin)
    directly, with PTX as its intermediate step. No nvcc, no CUDA Toolkit. Run on a Windows 11 PC with an RTX 3060.
  - **Mapping:** one thread per candidate (a lane), one 32-thread block, i.e. one warp, per decision
    (`rollout_warp`), all recorded decisions in one launch. W1, B1, W2, B2, S1 and S2 in `__constant__` memory. W1
    rows are packed as four int8 per int32 with a zero 4th byte, so layer 1 is one `__dp4a` per neuron on
    (θ, ω, u, 0); h is packed four per word for layer 2's four `__dp4a` per output (valid as signed bytes because
    h ≤ 127). θ wraps with `((v + 128) & 255) - 128`.
  - **Argmin:** five `__shfl_down_sync` rounds on (cost, index) pairs, comparing the index too: a tree does not
    compare in index order, so strict `<` alone would not give §5's lowest index on a tie; the pair comparison
    gives exactly the scan's result. For N > 32 candidates (`rollout_blocks`): shared memory across the warps of a
    block, then one 64-bit `atomicMin` on (cost << 32 | index) across blocks, the keys set to 2^64 − 1 first.
  - **Widths:** accumulators and costs are 32-bit `int`, wider than the RTL's 24 bits; both hold every value the
    golden model can produce (worst-case |acc| 85,570; the cost fits up to H = 79,607, since a step costs at most
    26,976 with u = −128, which the kernel accepts; §4.2's 26,975 is for the planner's −127..127). H is not limited
    to 16.
  - **Checks:** the same `exp_*.hex` vectors as `tb/tb_rollout_engine.v`, random sets against
    `golden.int_rollout` and `golden.argmin_first`, eleven planted bugs (ten kernel edits and one wrong start value
    of best_key; the five in code only `rollout_blocks` has must fail on its N = 4,096, 256-thread-block run), and
    the closed loop against the golden model on the same machine. Logs made on the PC and copied back; only their
    line endings were changed, plus one edit in `bench.log` that its header records.
  - **Benchmark provisional:** `results/cuda/bench.log` was made with `bench.py --shared-gpu` while another
    program's Python process held a GPU context. The rule for that PC is to skip benchmarks then, and the override
    was used without the OK the rule asks for. The timing figures stay provisional until a run without the flag
    replaces the log, or the run is accepted. `bench.py` keeps its default refusal and now re-checks before each
    section.
  - **Vectors made on the PC differ from the Mac's:** the PC's closed-loop episodes come out slightly different (the
    cause was not pinned down), so its `sim/` holds a different set of decisions. On the PC, `golden.int_rollout`
    reproduces every expected value in the Mac's files, and the GPU passes both sets.

### Stage 7: on a real FPGA

- **Board:** Sipeed Tang Nano 20K (GW2AR-LV18QN88C8/I7), 27 MHz oscillator, no PLL, wrapped by `board/top.v` (UART
  protocol in `board/README.md`); open-source flow Yosys 0.69+286 `synth_gowin`, nextpnr-himbaechel 0.11.1-59,
  apicula `gowin_pack`, openFPGALoader 1.1.1. The engine RTL is unchanged.
- **Deviation: the board build makes the multipliers from LUTs (`synth_gowin -nodsp`), now the board Makefile's
  default.** With the GW2A's DSP blocks in use the chip returned the right clock count (5,022) and wrong costs and
  states on every decision; `-noalu` (carry chains off, DSP on) failed the same way; `-nodsp` was bit-exact. With
  this flow and this design, the DSP-block builds compute wrong while the same RTL without DSP blocks is bit-exact,
  as are RTL simulation and the CUDA port; which stage of the flow is at fault is not isolated yet. A simple signedness error does not explain the wrong values (`board/diag/emulate_multiplier.py`).
- **Results:** 4,000/4,000 recorded decisions bit-exact and 20/20 closed-loop episodes identical to the golden model
  (`board/results/hardware/`). LANES = 8 only; a LANES = 16 `-nodsp` build fits (91% of LUT4s) but has not run.
- **Measured:** with a logic analyzer on the busy pin, one decision takes 186.000 µs on the chip, exactly 5,022
  clocks at 27 MHz, matching the RTL's cycle count (`board/results/hardware/2026-10-15_busy_pulse.txt`).
