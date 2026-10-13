# int8-rollout-engine: design spec (v1)

A 32-lane int8 Verilog engine that, every control tick, **imagines 32 candidate futures** of a pendulum through a
small learned dynamics model, scores each future, and returns the best one. Sampling-based model-predictive control
(MPC), with the model rollouts in hardware.

This document is the single source of truth. The Python golden model (`model/`) and the RTL (`rtl/`) both implement
exactly what is written here, and the testbench checks them against each other bit-for-bit.

## 0. Why this exists (the one-paragraph story)

Jim Fan's 2026 "Robotics: the Endgame" talk argues robots will act by imagining the future with learned world
models (NVIDIA's DreamZero "dreams a couple of seconds ahead and acts accordingly"). The hardware cost of
imagining is **candidates × imagined steps × one forward pass**, and the imagined steps are *sequential*: step t+1
needs the state from step t. This project measures that cost on a toy you can hold in your head: a 2-number state
(angle, angular velocity), a 3→16→2 network, 32 candidates, 15 steps. It is **not** a world action model and does not run
on an FPGA; it is RTL verified in simulation, with synthesis estimates.

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
each layer's scale is rounded to a power of two, and the weights absorb the rest of the scale, so the only
approximation is rounding the weights themselves to int8.

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
2. **Bit-exact replay** (`make test`): the testbench feeds every recorded decision through the RTL and compares
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
model, DreamZero on hardware, hardware-in-the-loop, runs on an FPGA, real-time robot, or faster than a GPU/Jetson.
It also never gives an MHz/µs figure without committed synthesis logs, and it says "post-route estimate,
out-of-context" for any that are given.

*Built with an AI pair programmer. The design decisions and their reasons are documented here so they can be checked
and explained.*

## Changelog

- **Stage 1, Python golden model (2026-10-12).** No rule in §1–§5 was changed. These are the details the spec left
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
