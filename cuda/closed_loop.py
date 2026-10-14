"""
Closed loop with the GPU in charge: the CUDA kernel picks every action of
the 20 Pendulum-v1 episodes (seed 0, H = 15), through the same host code as
the golden model and the RTL co-simulation (model/planner.run_episode).

The reference is the golden model's own integer closed loop, run on this
same machine first, exactly as model/golden.py runs it. PASS needs all of:
  - every episode return equal to golden's with float ==, no tolerance
  - the GPU's chosen plan equal to golden's at every tick
  - every GPU answer (best_idx and best_cost) equal to golden's on the same
    inputs, checked the moment it arrives (the log names the first tick
    that differs, if one does)
Also printed, not part of PASS: the returns against sim/returns.txt (what
golden.py wrote on this machine) and, if present, sim/mac/returns.txt
(written on the Mac that made the RTL logs). Those can differ even when every
integer decision is right: the two machines' episodes already diverge in the
float pendulum physics (most likely the last bits of sin/cos; not verified).

Run from the repo root on the machine with the GPU:
  .venv\\Scripts\\python.exe cuda\\closed_loop.py       (Windows)
  python3 cuda/closed_loop.py                          (Linux)
Exits 0 on PASS, 1 on FAIL.
"""

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cuda"))
sys.path.insert(0, str(ROOT / "model"))
import cuda_engine  # noqa: E402  (sets the CuPy cache directory before importing cupy)
import golden       # noqa: E402
import planner      # noqa: E402


class GpuEngine:
    """An engine for planner.run_episode() whose decisions come from the GPU,
    checked against the golden model on the same inputs at every tick."""

    def __init__(self, eng, p):
        self.eng = eng
        self.p = p
        self.chosen = []        # the GPU's best_idx at every tick, in order
        self.mismatches = []    # ticks where the GPU and golden disagree
        self.gpu_seconds = 0.0  # time spent in eng.decide (transfers + kernel)

    def __call__(self, theta, omega, plans):
        theta_q, omega_q = planner.quantize_state(theta, omega)
        t0 = time.perf_counter()
        r = self.eng.decide(theta_q, omega_q, plans)
        self.gpu_seconds += time.perf_counter() - t0

        # the golden model on exactly the same inputs
        cost, theta_f, omega_f = golden.int_rollout(theta_q, omega_q, plans, self.p)
        g_best = golden.argmin_first(cost)
        same = (r.best_idx == g_best and r.best_cost == int(cost[g_best])
                and np.array_equal(r.cost, cost) and np.array_equal(r.theta, theta_f)
                and np.array_equal(r.omega, omega_f))
        if not same:
            self.mismatches.append(
                f"tick {len(self.chosen)}: state ({theta_q}, {omega_q}): GPU best {r.best_idx} "
                f"(cost {r.best_cost}), golden best {g_best} (cost {int(cost[g_best])})")
        self.chosen.append(r.best_idx)
        return r.best_idx


def read_returns(path):
    """returns.txt as golden.py writes it: '# comment' then 'episode repr(return)'."""
    if not path.exists():
        return None
    out = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            e, r = line.split()
            out[int(e)] = r
    return out


def main():
    ap = argparse.ArgumentParser(description="closed loop with the CUDA kernel choosing every action")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--H", type=int, default=15)
    args = ap.parse_args()

    print("CUDA closed loop: gymnasium Pendulum-v1, seed "
          f"{args.seed}, H = {args.H}, {args.episodes} episodes x {planner.EPISODE_STEPS} ticks")
    print("command: " + " ".join([Path(sys.executable).name] + sys.argv))
    for line in cuda_engine.versions():
        print(line)
    print(f"gymnasium {gym.__version__}")
    print("GPU: cuda/rollout_kernel.cu (rollout_warp, one launch per tick); its best_idx picks "
          "every applied action")
    print("golden: model/golden.py integer planner on this machine; both use model/planner.py")

    p = golden.fold(golden.load_float_model())
    eng = cuda_engine.CudaEngine()

    # 1. the reference: golden's own closed loop, as golden.py runs it
    env = gym.make("Pendulum-v1")
    g_returns, g_chosen = [], []
    t0 = time.perf_counter()
    for e in range(args.episodes):
        records = []
        g_returns.append(planner.run_episode(env, golden.int_engine(p, record=records),
                                             args.seed, e, args.H))
        g_chosen.append([r["best"] for r in records])
    t_golden = time.perf_counter() - t0

    # 2. the same episodes with the GPU choosing every action
    gpu = GpuEngine(eng, p)
    gpu_returns = []
    print()
    print(f"{'episode':>7}  {'GPU return':>22}  {'golden return':>22}  {'return':<8}  chosen best_idx")
    t0 = time.perf_counter()
    for e in range(args.episodes):
        first = len(gpu.chosen)
        ret = planner.run_episode(env, gpu, args.seed, e, args.H)
        gpu_returns.append(ret)
        chosen = gpu.chosen[first:]
        agree = sum(a == b for a, b in zip(chosen, g_chosen[e]))
        print(f"{e:>7}  {ret!r:>22}  {g_returns[e]!r:>22}  "
              f"{'MATCH' if ret == g_returns[e] else 'MISMATCH':<8}  "
              f"{agree}/{len(chosen)} equal to golden", flush=True)
    t_gpu = time.perf_counter() - t0

    n = len(gpu.chosen)
    same_returns = sum(a == b for a, b in zip(gpu_returns, g_returns))
    same_ticks = sum(a == b for a, b in zip(gpu.chosen, (c for ep in g_chosen for c in ep)))
    ok = same_returns == args.episodes and same_ticks == n and not gpu.mismatches

    print()
    for line in gpu.mismatches[:5]:
        print("  " + line)
    print(f"decisions made by the GPU: {n} ({planner.N_CAND} candidates x {args.H} imagined steps each)")
    print(f"GPU costs, final states, best_idx and best_cost equal golden's on the same inputs: "
          f"{n - len(gpu.mismatches)}/{n}")
    print(f"GPU chosen best_idx equal to the golden closed loop's: {same_ticks}/{n}")
    print(f"episode returns identical (float ==): {same_returns}/{args.episodes}")
    print(f"mean return: GPU {np.mean(gpu_returns):.1f}, golden {np.mean(g_returns):.1f}")
    print(f"wall time: GPU closed loop {t_gpu:.2f} s ({1000 * t_gpu / n:.3f} ms per tick, of which "
          f"eng.decide {1000 * gpu.gpu_seconds / n:.3f} ms; the rest is the pendulum, the planner "
          f"and the per-tick golden check), golden closed loop {t_golden:.2f} s")

    # not part of PASS: the files golden.py wrote
    for label, path in [("sim/returns.txt (golden.py on this machine)", ROOT / "sim" / "returns.txt"),
                        ("sim/mac/returns.txt (golden.py on the Mac, the RTL logs' vectors)",
                         ROOT / "sim" / "mac" / "returns.txt")]:
        ref = read_returns(path)
        if ref is None:
            print(f"{label}: not present")
            continue
        equal = sum(repr(r) == ref.get(e) for e, r in enumerate(gpu_returns))
        print(f"GPU returns digit for digit equal to {label}: {equal}/{len(gpu_returns)}")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
