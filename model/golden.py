"""
Golden model for int8-rollout-engine: the integer math the RTL must match
bit for bit (DESIGN.md §2-§5), the closed-loop planner, and the test vectors.

  1. Folds the float dynamics model (model/dynamics_fp32.npz) into int8
     weights, wide biases and one power-of-two shift per layer (§4.1),
     checks that every accumulator fits ACC_W = 24 bits, writes rom/.
  2. Prints one imagined step for one lane with every intermediate value,
     so the RTL can be debugged by hand.
  3. Runs the integer planner in closed loop on Pendulum-v1 and writes every
     decision the engine has to make (inputs and expected outputs) to sim/.
  4. With --eval: compares planners that imagine with the float model and
     with the integer model, for H = 15 and H = 5, over 5 seeds x 20
     episodes, and writes results/planner_quality.txt.

Everything the hardware computes is done here with integers only (Python
ints or NumPy int64). ">>" on a negative integer rounds toward minus
infinity (floor), exactly like Verilog's ">>>" on a signed value.

Run from the repo root:
  python3 model/golden.py                        rom/ and sim/ (seed 0, H = 15, 4,000 decisions)
  python3 model/golden.py --count 1000           only the first 1,000 decisions
  python3 model/golden.py --H 5 --out sim/h5     H = 5 vectors for the cycle-count sweep
  python3 model/golden.py --eval                 also the fp32 vs int8 closed-loop table

ROM files (rom/, committed, read with $readmemh; signed values are two's
complement)
  w1.hex     48 lines x 2 hex digits: W1[j][i] at address 3*j + i, the order
             the sequencer broadcasts them (neuron 0's three weights first)
  b1.hex     16 lines x 6 hex digits: B1[j], signed 24-bit
  w2.hex     32 lines x 2 hex digits: W2[k][j] at address 16*k + j
  b2.hex      2 lines x 6 hex digits: B2[k], signed 24-bit
  params.vh  localparams S1, S2_0, S2_1 (the shifts) and BIAS_W

Decision vectors (sim/ or --out, not committed). Decision d = 0..COUNT-1,
lane n = 0..31, imagined step t = 0..H-1. A wide word holds lane 0 in its
lowest bits, so in the hex text lane 31 comes first (the same packing as
tiny-tensor-core). Every line of a file has the same width.
  vec_state.hex  line d        16 bits: {omega0[7:0], theta0[7:0]}
  vec_act.hex    line d*H + t  256 bits: lane n's action u[n][t] in bits
                               [8n+7:8n], exactly the engine's ACT[t] word
  exp_cost.hex   line d        768 bits: lane n's total cost in [24n+23:24n]
  exp_theta.hex  line d        256 bits: lane n's theta after H steps in [8n+7:8n]
  exp_omega.hex  line d        256 bits: lane n's omega after H steps in [8n+7:8n]
  exp_best.hex   line d        32 bits: {best_idx[7:0], best_cost[23:0]}
  vectors.vh     localparams VEC_COUNT and VEC_H for the testbench
  returns.txt    each episode's total reward (the co-simulation must match)
  trace.txt      every lane and step of decision 0, with h[] and acc2
"""

import argparse
import math
import platform
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
import planner  # noqa: E402  (the shared host code, model/planner.py)

ACC_W = 24       # lane accumulator width in the RTL (DESIGN.md §6)
COST_W = 24      # cost accumulator width
H_MAX = 16       # act_t[3:0] can address 16 imagined steps
N_IN, N_HID, N_OUT = 3, 16, 2

# LSB size of each input (theta, omega, u) and output (d_theta, d_omega)
L_IN = np.array([planner.THETA_LSB, planner.OMEGA_LSB, planner.U_LSB])
L_OUT = np.array([planner.THETA_LSB, planner.OMEGA_LSB])


def to_hex(v, bits):
    """Two's complement in a fixed number of hex digits."""
    return f"{int(v) & ((1 << bits) - 1):0{bits // 4}x}"


def pack(values, bits):
    """values[0] in the lowest bits, so the hex string lists the last first."""
    return "".join(to_hex(v, bits) for v in reversed(list(values)))


# ---------------------------------------------------------------- folding --

def load_float_model():
    d = np.load(ROOT / "model" / "dynamics_fp32.npz")
    return {k: d[k] for k in d.files}


def largest_shift(max_abs):
    """The largest S with max_abs * 2^S <= 127, so the biggest weight still
    fits in a signed byte. Returns 0 if even S = 1 does not fit; the caller
    asserts S >= 1."""
    s = 0
    while max_abs * 2 ** (s + 1) <= 127:
        s += 1
    return s


def fold(m):
    """Fold every scale factor into integer weights (DESIGN.md §4.1).

    Afterwards one imagined step needs only multiply-adds, an arithmetic
    shift and a clamp: no requantization multiplier. Done in float64; the
    results are plain integers.
    """
    w1, b1, w2, b2 = (m[k].astype(np.float64) for k in ("W1", "b1", "W2", "b2"))
    xm, xs, ym, ys = (m[k].astype(np.float64) for k in ("xm", "xs", "ym", "ys"))

    # one hidden LSB = s_h, chosen so the largest pre-activation seen in
    # training lands exactly on 127
    s_h = float(m["a1_max"]) / 127

    # layer 1: the float model computes W1 @ ((L * x_q - xm) / xs) + b1;
    # expand it into (weight per input LSB) * x_q + (constant), in hidden LSBs
    w1_real = w1 * L_IN / xs / s_h                        # [j][i]
    b1_real = (b1 - (w1 * xm / xs).sum(axis=1)) / s_h
    s1 = largest_shift(np.abs(w1_real).max())
    W1 = np.round(w1_real * 2 ** s1).astype(np.int64)
    # + 2^(S1-1) is half an output LSB: it turns the floor shift into
    # round-half-up, for free, because it is folded into the bias
    B1 = np.round(b1_real * 2 ** s1).astype(np.int64) + 2 ** (s1 - 1)

    # layer 2: output k = ys[k] * (W2[k] @ h_real + b2[k]) + ym[k], with
    # h_real = s_h * h; measured in state LSBs (pi/128 or 8/127)
    w2_real = ys[:, None] * w2 * s_h / L_OUT[:, None]     # [k][j]
    b2_real = (ys * b2 + ym) / L_OUT
    s2 = [largest_shift(np.abs(w2_real[k]).max()) for k in range(N_OUT)]
    W2 = np.stack([np.round(w2_real[k] * 2 ** s2[k]) for k in range(N_OUT)]).astype(np.int64)
    B2 = np.array([np.round(b2_real[k] * 2 ** s2[k]) + 2 ** (s2[k] - 1)
                   for k in range(N_OUT)]).astype(np.int64)

    assert s1 >= 1 and min(s2) >= 1, f"shift < 1: S1 = {s1}, S2 = {s2}"
    assert np.abs(W1).max() <= 127 and np.abs(W2).max() <= 127
    return {"W1": W1, "B1": B1, "S1": s1, "W2": W2, "B2": B2, "S2": tuple(s2), "s_h": s_h}


def worst_case_acc(p):
    """Largest |accumulator| any input could produce, not just our data.

    Layer 1 inputs are int8, so |x| <= 128. Hidden values are 0..127. An
    accumulator starts at the bias and adds one product per clock, so every
    partial sum is also inside this bound.
    """
    acc1 = int((np.abs(p["B1"]) + 128 * np.abs(p["W1"]).sum(axis=1)).max())
    acc2 = int((np.abs(p["B2"]) + 127 * np.abs(p["W2"]).sum(axis=1)).max())
    return acc1, acc2


def check_widths(p):
    acc1, acc2 = worst_case_acc(p)
    assert acc1 < 2 ** (ACC_W - 1), f"layer-1 accumulator can overflow: {acc1}"
    assert acc2 < 2 ** (ACC_W - 1), f"layer-2 accumulator can overflow: {acc2}"
    # cost: theta'^2 <= 128^2, (21*omega'^2) >> 5 with |omega'| <= 127, u^2 >> 11
    step_max = 128 * 128 + ((21 * 127 * 127) >> 5) + ((127 * 127) >> 11)
    assert H_MAX * step_max < 2 ** COST_W
    return acc1, acc2, step_max


def write_rom(p):
    rom = ROOT / "rom"
    rom.mkdir(exist_ok=True)
    # .ravel() reads W1[j][i] row by row, giving address 3*j + i (and 16*k + j)
    (rom / "w1.hex").write_text("".join(to_hex(w, 8) + "\n" for w in p["W1"].ravel()))
    (rom / "b1.hex").write_text("".join(to_hex(b, ACC_W) + "\n" for b in p["B1"]))
    (rom / "w2.hex").write_text("".join(to_hex(w, 8) + "\n" for w in p["W2"].ravel()))
    (rom / "b2.hex").write_text("".join(to_hex(b, ACC_W) + "\n" for b in p["B2"]))
    acc1, acc2 = worst_case_acc(p)
    params = [("S1", p["S1"], "h[j] = clamp((B1[j] + sum_i W1[j][i]*x[i]) >>> S1, 0, 127)"),
              ("S2_0", p["S2"][0], "d_theta = (B2[0] + sum_j W2[0][j]*h[j]) >>> S2_0"),
              ("S2_1", p["S2"][1], "d_omega = (B2[1] + sum_j W2[1][j]*h[j]) >>> S2_1"),
              ("BIAS_W", ACC_W, "width of each entry in b1.hex and b2.hex")]
    (rom / "params.vh").write_text(
        "// generated by model/golden.py from model/dynamics_fp32.npz\n"
        + "".join(f"localparam {name:<6} = {value:2d};  // {note}\n" for name, value, note in params)
        + f"// worst-case |acc1| = {acc1}, |acc2| = {acc2}: both fit a signed {ACC_W}-bit accumulator\n")


# ------------------------------------------------------- integer engine --

def int_step(theta, omega, u, p):
    """One imagined step (DESIGN.md §4) for every lane at once.

    theta, omega, u: int64 arrays with one entry per lane (candidate).
    Returns the new theta and omega, this step's cost, and the intermediate
    values (for the trace and the corner-case counts).
    """
    x = np.stack([theta, omega, u])                  # (3, lanes)
    acc1 = p["B1"][:, None] + p["W1"] @ x            # (16, lanes): bias + 3 MACs
    h = np.clip(acc1 >> p["S1"], 0, 127)             # floor shift, ReLU, saturate
    acc2 = p["B2"][:, None] + p["W2"] @ h            # (2, lanes): bias + 16 MACs
    d_theta = acc2[0] >> p["S2"][0]
    d_omega = acc2[1] >> p["S2"][1]

    theta_sum = theta + d_theta
    low8 = theta_sum & 0xFF                          # keep the low 8 bits ...
    theta_new = np.where(low8 >= 128, low8 - 256, low8)   # ... read as signed: wrap
    omega_sum = omega + d_omega
    omega_new = np.clip(omega_sum, -127, 127)        # saturate

    cost = theta_new * theta_new + ((21 * omega_new * omega_new) >> 5) + ((u * u) >> 11)
    inner = {"acc1": acc1, "h": h, "acc2": acc2,
             "theta_sum": theta_sum, "omega_sum": omega_sum}
    return theta_new, omega_new, cost, inner


def int_step_one_lane(theta, omega, u, p):
    """The same step for a single lane with plain Python ints and loops, so it
    reads line for line like DESIGN.md §4. Used for the hand check, the trace,
    and to cross-check int_step()."""
    x = (theta, omega, u)
    acc1, h = [], []
    for j in range(N_HID):
        acc = int(p["B1"][j])
        for i in range(N_IN):
            acc += int(p["W1"][j][i]) * x[i]
        acc1.append(acc)
        h.append(min(max(acc >> p["S1"], 0), 127))
    acc2, d = [], []
    for k in range(N_OUT):
        acc = int(p["B2"][k])
        for j in range(N_HID):
            acc += int(p["W2"][k][j]) * h[j]
        acc2.append(acc)
        d.append(acc >> p["S2"][k])
    theta_new = (theta + d[0]) & 0xFF
    if theta_new >= 128:
        theta_new -= 256
    omega_new = min(max(omega + d[1], -127), 127)
    cost = theta_new * theta_new + ((21 * omega_new * omega_new) >> 5) + ((u * u) >> 11)
    return theta_new, omega_new, cost, {"acc1": acc1, "h": h, "acc2": acc2, "d": d}


class Corners:
    """Counts the corner cases the RTL tests need to exercise (DESIGN.md §7.4)
    and the largest accumulators actually seen."""

    def __init__(self):
        self.max_acc1 = self.max_acc2 = 0
        self.lane_steps = 0
        self.h_clamped = 0      # hidden values with acc1 >> S1 > 127
        self.theta_wraps = 0    # theta + d_theta outside -128..127
        self.omega_sats = 0     # omega + d_omega outside -127..127
        self.decisions = 0
        self.ties = 0           # decisions where 2+ lanes share the lowest cost

    def add_step(self, inner, p):
        self.max_acc1 = max(self.max_acc1, int(np.abs(inner["acc1"]).max()))
        self.max_acc2 = max(self.max_acc2, int(np.abs(inner["acc2"]).max()))
        self.lane_steps += inner["acc1"].shape[1]
        self.h_clamped += int(((inner["acc1"] >> p["S1"]) > 127).sum())
        ts, os_ = inner["theta_sum"], inner["omega_sum"]
        self.theta_wraps += int(((ts < -128) | (ts > 127)).sum())
        self.omega_sats += int(((os_ < -127) | (os_ > 127)).sum())

    def add_decision(self, cost):
        self.decisions += 1
        self.ties += int((cost == cost.min()).sum() > 1)


def int_rollout(theta0, omega0, plans, p, corners=None):
    """Imagine every candidate H steps ahead from the same int8 state.
    Returns each lane's total cost and final (theta, omega)."""
    n_cand, horizon = plans.shape
    theta = np.full(n_cand, theta0, np.int64)
    omega = np.full(n_cand, omega0, np.int64)
    cost = np.zeros(n_cand, np.int64)
    for t in range(horizon):
        theta, omega, c, inner = int_step(theta, omega, plans[:, t], p)
        cost += c
        if corners is not None:
            corners.add_step(inner, p)
    return cost, theta, omega


def argmin_first(cost):
    """Scan n = 0..N-1 with strict <, like the RTL: a tie keeps the lower index."""
    best = 0
    for n in range(1, len(cost)):
        if cost[n] < cost[best]:
            best = n
    return best


def int_engine(p, record=None, corners=None):
    """An engine for planner.run_episode() that imagines with the integer
    model. If record is a list, every decision is appended to it."""
    def decide(theta, omega, plans):
        theta_q, omega_q = planner.quantize_state(theta, omega)
        cost, theta_f, omega_f = int_rollout(theta_q, omega_q, plans, p, corners)
        best = argmin_first(cost)
        if corners is not None:
            corners.add_decision(cost)
        if record is not None:
            record.append({"theta0": theta_q, "omega0": omega_q, "plans": plans.copy(),
                           "cost": cost, "theta": theta_f, "omega": omega_f, "best": best})
        return best
    return decide


# --------------------------------------------------------- float engine --

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def float_engine(m):
    """An engine that imagines with the float32 model and the exact float
    cost theta^2 + 0.1 omega^2 + 0.004 u^2, starting from the unquantized
    measured state: the "no quantization anywhere" reference. theta is
    wrapped to [-pi, pi) like a real angle; omega is not clipped, because
    the +-8 rad/s limit only comes from the int8 format. It gets the same
    candidate plans (int8 actions) and the same argmin rule as the integer
    engine."""
    w1, b1, w2, b2 = m["W1"], m["b1"], m["W2"], m["b2"]
    xm, xs, ym, ys = m["xm"], m["xs"], m["ym"], m["ys"]

    def decide(theta0, omega0, plans):
        n_cand, horizon = plans.shape
        theta = np.full(n_cand, theta0, np.float32)
        omega = np.full(n_cand, omega0, np.float32)
        u_all = plans.astype(np.float32) / 127
        cost = np.zeros(n_cand, np.float32)
        for t in range(horizon):
            u = u_all[:, t]
            xn = (np.stack([theta, omega, u], axis=1) - xm) / xs
            h = np.maximum(xn @ w1.T + b1, 0)
            y = (h @ w2.T + b2) * ys + ym
            theta = wrap_angle(theta + y[:, 0])
            omega = omega + y[:, 1]
            cost += theta ** 2 + 0.1 * omega ** 2 + 0.004 * u ** 2
        return argmin_first(cost)
    return decide


# ------------------------------------------------------------ reporting --

def fold_report(p, corners=None):
    acc1, acc2, step_max = check_widths(p)
    lines = [
        f"hidden scale s_h = {p['s_h']:.6f} (largest training pre-activation / 127)",
        f"S1 = {p['S1']}, S2 = ({p['S2'][0]}, {p['S2'][1]})   (all >= 1: OK)",
        f"max |W1| = {np.abs(p['W1']).max()}, max |W2| = {np.abs(p['W2']).max()}, "
        f"B1 in [{p['B1'].min()}, {p['B1'].max()}], B2 = {[int(b) for b in p['B2']]}",
        f"worst-case |acc1| = {acc1:,}, |acc2| = {acc2:,}  "
        f"(ACC_W = {ACC_W} holds |acc| < {2 ** (ACC_W - 1):,}: OK)",
        f"worst-case cost: {step_max:,} per step, {H_MAX * step_max:,} for H = {H_MAX} "
        f"(< 2^{COST_W}: OK)",
    ]
    if corners is not None:
        lines.append(f"observed  |acc1| = {corners.max_acc1:,}, |acc2| = {corners.max_acc2:,}  "
                     f"(over {corners.lane_steps:,} imagined lane-steps)")
    return lines


def hand_check(rec, p, m):
    """Print one lane's first imagined step with every intermediate value."""
    th, om, u = rec["theta0"], rec["omega0"], int(rec["plans"][0, 0])
    th_new, om_new, c, inner = int_step_one_lane(th, om, u, p)
    W1, B1, S1 = p["W1"], p["B1"], p["S1"]
    print(f"\nHand check: decision 0, lane 0, imagined step 0")
    print(f"  x = (theta, omega, u) = ({th}, {om}, {u})")
    print(f"  layer 1:  acc1[j] = B1[j] + W1[j][0]*{th} + W1[j][1]*{om} + W1[j][2]*{u};"
          f"  h[j] = clamp(acc1[j] >> {S1}, 0, 127)")
    for j in range(N_HID):
        print(f"    j={j:2d}: {inner['acc1'][j]:7d} = {B1[j]:6d} + ({W1[j][0]:4d})*{th}"
              f" + ({W1[j][1]:4d})*{om} + ({W1[j][2]:4d})*{u}"
              f"   >> {S1} = {inner['acc1'][j] >> S1:5d}   h = {inner['h'][j]}")
    for k, name in enumerate(("d_theta", "d_omega")):
        terms = " + ".join(f"({p['W2'][k][j]})*{inner['h'][j]}" for j in range(N_HID))
        print(f"  acc2[{k}] = {p['B2'][k]} + {terms}")
        print(f"          = {inner['acc2'][k]};  {name} = acc2[{k}] >> {p['S2'][k]} = {inner['d'][k]}")
    print(f"  theta' = low 8 bits of ({th} + {inner['d'][0]}) = {th_new}")
    print(f"  omega' = clamp({om} + {inner['d'][1]}, -127, 127) = {om_new}")
    print(f"  c_step = ({th_new})^2 + ((21*({om_new})^2) >> 5) + (({u})^2 >> 11)"
          f" = {th_new * th_new} + {(21 * om_new * om_new) >> 5} + {(u * u) >> 11} = {c}")

    # the same step through the float model and through the real physics,
    # starting from the int8 state converted back to real units
    real = np.array([th * planner.THETA_LSB, om * planner.OMEGA_LSB, u / 127])
    h_f = np.maximum(((real - m["xm"]) / m["xs"]) @ m["W1"].T + m["b1"], 0)
    y_f = (h_f @ m["W2"].T + m["b2"]) * m["ys"] + m["ym"]
    # Pendulum-v1 physics (gymnasium source): g = 10, m = l = 1, dt = 0.05,
    # torque = 2u, |omega| <= 8, and the angle moves by the *new* omega * dt
    om_next = np.clip(real[1] + (15 * np.sin(real[0]) + 3 * 2 * real[2]) * 0.05, -8, 8)
    print(f"  in LSBs: float model d = ({y_f[0] / L_OUT[0]:+.3f}, {y_f[1] / L_OUT[1]:+.3f}),"
          f" real physics d = ({om_next * 0.05 / L_OUT[0]:+.3f}, {(om_next - real[1]) / L_OUT[1]:+.3f}),"
          f" integer step d = ({inner['d'][0]:+d}, {inner['d'][1]:+d})")


def cross_check(records, p, n_decisions=100):
    """The vectorized int_step() and the loop version must agree exactly."""
    for rec in records[:n_decisions]:
        for n in range(rec["plans"].shape[0]):
            th, om, total = rec["theta0"], rec["omega0"], 0
            for u in rec["plans"][n]:
                th, om, c, _ = int_step_one_lane(th, om, int(u), p)
                total += c
            assert (total, th, om) == (rec["cost"][n], rec["theta"][n], rec["omega"][n])
        assert rec["best"] == int(np.argmin(rec["cost"]))   # argmin returns the first minimum


# ---------------------------------------------------------------- vectors --

def write_vectors(records, returns, p, out, horizon, seed):
    out.mkdir(parents=True, exist_ok=True)
    (out / "vec_state.hex").write_text(
        "".join(pack([r["theta0"], r["omega0"]], 8) + "\n" for r in records))
    (out / "vec_act.hex").write_text(
        "".join(pack(r["plans"][:, t], 8) + "\n" for r in records for t in range(horizon)))
    (out / "exp_cost.hex").write_text("".join(pack(r["cost"], 24) + "\n" for r in records))
    (out / "exp_theta.hex").write_text("".join(pack(r["theta"], 8) + "\n" for r in records))
    (out / "exp_omega.hex").write_text("".join(pack(r["omega"], 8) + "\n" for r in records))
    (out / "exp_best.hex").write_text(
        "".join(to_hex(r["best"], 8) + to_hex(r["cost"][r["best"]], 24) + "\n" for r in records))
    (out / "vectors.vh").write_text(
        f"// generated by model/golden.py --seed {seed} --H {horizon} --count {len(records)}\n"
        f"localparam VEC_COUNT = {len(records)};\n"
        f"localparam VEC_H     = {horizon};\n")
    (out / "returns.txt").write_text(
        f"# seed {seed}, H = {horizon}: total reward of each integer-planner episode\n"
        + "".join(f"{e} {r!r}\n" for e, r in enumerate(returns)))

    # trace of decision 0: every lane, every imagined step
    rec = records[0]
    lines = [f"decision 0: theta0 = {rec['theta0']}, omega0 = {rec['omega0']}, "
             f"S1 = {p['S1']}, S2 = {p['S2']}",
             "lane  t     u  theta omega  acc2[0]  acc2[1]  c_step     cost  h[0..15]"]
    for n in range(rec["plans"].shape[0]):
        th, om, total = rec["theta0"], rec["omega0"], 0
        for t in range(horizon):
            u = int(rec["plans"][n, t])
            th, om, c, inner = int_step_one_lane(th, om, u, p)
            total += c
            lines.append(f"{n:4d} {t:2d} {u:5d} {th:6d} {om:5d} {inner['acc2'][0]:8d} "
                         f"{inner['acc2'][1]:8d} {c:7d} {total:8d}  {inner['h']}")
    lines.append(f"best_idx = {rec['best']}, best_cost = {rec['cost'][rec['best']]}")
    (out / "trace.txt").write_text("\n".join(lines) + "\n")


def dump(p, m, args):
    """Closed loop with the integer planner; record every decision."""
    episodes = args.episodes
    count = args.count if args.count is not None else episodes * planner.EPISODE_STEPS
    needed = math.ceil(count / planner.EPISODE_STEPS)
    assert needed <= episodes, "--count needs more episodes than --episodes"
    env = gym.make("Pendulum-v1")
    records, returns = [], []
    engine = int_engine(p, record=records)
    for e in range(needed):
        returns.append(planner.run_episode(env, engine, args.seed, e, args.H))
    records = records[:count]

    # corner cases in exactly the decisions written (replaying them also
    # re-checks that every recorded output is reproducible)
    corners = Corners()
    for r in records:
        cost, theta, omega = int_rollout(r["theta0"], r["omega0"], r["plans"], p, corners)
        assert (cost == r["cost"]).all() and (theta == r["theta"]).all() and (omega == r["omega"]).all()
        corners.add_decision(cost)

    hand_check(records[0], p, m)
    cross_check(records, p)
    out = ROOT / args.out
    write_vectors(records, returns, p, out, args.H, args.seed)

    print(f"\nInteger closed loop, seed {args.seed}, H = {args.H}: {needed} episodes, "
          f"mean return {np.mean(returns):.1f}")
    print(f"wrote {len(records)} decisions to {out.relative_to(ROOT)}/ "
          f"(int_step and the loop version agree on the first 100)")
    print(f"corner cases in these {corners.decisions} decisions "
          f"({corners.lane_steps:,} lane-steps): h clamped at 127: {corners.h_clamped}, "
          f"theta wraps: {corners.theta_wraps}, omega saturations: {corners.omega_sats}, "
          f"decisions with a tied best cost: {corners.ties}")
    print(f"observed |acc1| = {corners.max_acc1:,}, |acc2| = {corners.max_acc2:,}")


# ------------------------------------------------------------- evaluation --

def bootstrap_ci(diff, rng, n_boot=10_000):
    """Mean of the paired differences and a percentile-bootstrap 95% CI."""
    diff = np.asarray(diff).ravel()
    means = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return diff.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def zero_torque_return(env, seed, episode):
    env.reset(seed=planner.env_seed(seed, episode))
    return sum(float(env.step(np.zeros(1, np.float32))[1]) for _ in range(planner.EPISODE_STEPS))


def evaluate(p, m, seeds=range(5), episodes=20):
    t0 = time.time()
    env = gym.make("Pendulum-v1")
    corners = Corners()
    configs = [("fp32", 15), ("int8", 15), ("fp32", 5), ("int8", 5)]
    ret = {c: np.zeros((len(seeds), episodes)) for c in configs}
    zero = np.zeros((len(seeds), episodes))
    for name, horizon in configs:
        engine = float_engine(m) if name == "fp32" else int_engine(p, corners=corners)
        for si, seed in enumerate(seeds):
            for e in range(episodes):
                ret[(name, horizon)][si, e] = planner.run_episode(env, engine, seed, e, horizon)
        print(f"  {name} H={horizon}: done ({time.time() - t0:.0f} s)", flush=True)
    for si, seed in enumerate(seeds):
        for e in range(episodes):
            zero[si, e] = zero_torque_return(env, seed, e)

    lines = [
        "# generated by: python3 model/golden.py --eval",
        f"# {time.strftime('%Y-%m-%d %H:%M')}, Python {platform.python_version()}, "
        f"NumPy {np.__version__}, gymnasium {gym.__version__}",
        "",
        "Integer model (folded from model/dynamics_fp32.npz):",
        *("  " + s for s in fold_report(p, corners)),
        "",
        "Closed-loop return on gymnasium Pendulum-v1 (sum of 200 rewards per episode;",
        "higher is better). Each planner imagines N = 32 candidate plans per tick",
        "(warm start, sigma = 0.5) with either the float32 model and float cost (fp32)",
        "or the integer model and integer cost of DESIGN.md §4 (int8). For a given seed",
        "every planner sees the same 20 start states and the same host RNG seeds.",
        "",
        f"{'seed':>4}  {'fp32 H=15':>10}  {'int8 H=15':>10}  {'fp32 H=5':>10}  "
        f"{'int8 H=5':>10}  {'zero torque':>11}",
    ]
    for si, seed in enumerate(seeds):
        lines.append(f"{seed:>4}  " + "  ".join(f"{ret[c][si].mean():10.1f}" for c in configs)
                     + f"  {zero[si].mean():11.1f}")
    lines.append(f"{'mean':>4}  " + "  ".join(f"{ret[c].mean():10.1f}" for c in configs)
                 + f"  {zero.mean():11.1f}")

    rng = np.random.default_rng(0)
    n = len(seeds) * episodes
    lines += ["", f"Paired differences over all {n} episodes "
                  "(mean, percentile-bootstrap 95% CI, 10,000 resamples):"]
    for label, a, b in [("int8 - fp32, H=15", ("int8", 15), ("fp32", 15)),
                        ("int8 - fp32, H=5 ", ("int8", 5), ("fp32", 5)),
                        ("H=5 - H=15, fp32", ("fp32", 5), ("fp32", 15)),
                        ("H=5 - H=15, int8", ("int8", 5), ("int8", 15))]:
        mean, lo, hi = bootstrap_ci(ret[a] - ret[b], rng)
        lines.append(f"  {label}: {mean:+7.1f}   [{lo:+7.1f}, {hi:+7.1f}]")
    lines += ["",
              f"int8 corner cases over {corners.decisions:,} decisions: h clamped at 127: "
              f"{corners.h_clamped:,}, theta wraps: {corners.theta_wraps:,}, "
              f"omega saturations: {corners.omega_sats:,}, tied best cost: {corners.ties:,}",
              f"runtime: {time.time() - t0:.0f} s"]

    text = "\n".join(lines) + "\n"
    print("\n" + text)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "planner_quality.txt").write_text(text)
    print("wrote results/planner_quality.txt")


# ------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=0, help="seed for the vector dump (default 0)")
    ap.add_argument("--H", type=int, default=15, help="imagined steps per plan (default 15)")
    ap.add_argument("--episodes", type=int, default=20, help="episodes to run (default 20)")
    ap.add_argument("--count", type=int, default=None,
                    help="decisions to write (default: all, episodes x 200)")
    ap.add_argument("--out", default="sim", help="vector directory (default sim)")
    ap.add_argument("--eval", action="store_true",
                    help="also run the fp32 vs int8 comparison -> results/planner_quality.txt")
    args = ap.parse_args()
    assert 1 <= args.H <= H_MAX

    t0 = time.time()
    m = load_float_model()
    p = fold(m)
    print("Integer model (DESIGN.md §4.1):")
    for line in fold_report(p):
        print("  " + line)
    write_rom(p)
    print("wrote rom/w1.hex, b1.hex, w2.hex, b2.hex, params.vh")

    dump(p, m, args)
    print(f"({time.time() - t0:.1f} s)")
    if args.eval:
        print("\nClosed-loop evaluation, 5 seeds x 20 episodes x 4 planners:")
        evaluate(p, m)


if __name__ == "__main__":
    main()
