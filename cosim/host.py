"""
Closed-loop co-simulation host (DESIGN.md §7.3).

Python runs the pendulum (gymnasium Pendulum-v1) and the host side of the
planner; Icarus Verilog runs the RTL (cosim/tb_cosim.v around
rtl/rollout_engine.v). The two talk through two named pipes. Every control
tick:

  1. the host reads the pendulum, quantizes its state and builds the 32
     candidate plans, with model/planner.py: the same code, the same seeds
     and the same random numbers as model/golden.py
  2. it sends the state and the 32 x H actions to the simulator (to_rtl)
  3. the RTL imagines all 32 futures and sends back best_idx, best_cost
     and its clock count (from_rtl)
  4. the host applies the first action of plan best_idx, so the RTL's
     choice is what moves the pendulum

The reference is the golden model's own closed loop: the same episodes run
with model/golden.py's integer engine, exactly as golden.py runs them. The
check: every episode return is identical, to the last bit of the float,
and the RTL chose the same plan at every tick. The host also has the
golden model at hand, so it checks every RTL answer the moment it arrives
against the golden model on the same inputs; if anything ever differs,
the log says at which tick.

Start it through cosim/run.sh, which compiles the testbench, makes the
pipes and starts the simulator first:
  PYTHON=python3 bash cosim/run.sh
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import gymnasium as gym

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
import golden   # noqa: E402  the integer golden model, model/golden.py
import planner  # noqa: E402  the host code golden.py uses, model/planner.py

# Longest wait for the simulator, in seconds: to open the pipes, or to
# answer one tick (a tick takes 0.1 to 0.2 s). Without a limit, a crashed or
# stuck simulator would leave the host waiting forever; this turns it into
# a FAIL, like the watchdog in the testbenches.
TIMEOUT_S = 60


def timed_out(signum, frame):
    raise TimeoutError(f"the simulator did not answer within {TIMEOUT_S} s")


class RtlEngine:
    """An engine for planner.run_episode() whose decisions come from the RTL.

    planner.run_episode() calls it once per tick with the measured state and
    the 32 plans and applies the plan it returns, the same way it calls the
    golden model's engine.
    """

    def __init__(self, to_rtl, from_rtl, p):
        self.to_rtl = to_rtl
        self.from_rtl = from_rtl
        self.p = p              # the folded integer model, for the per-tick check
        self.chosen = []        # the RTL's best_idx at every tick, in order
        self.cycles = []        # the RTL's clock count at every tick
        self.mismatches = []    # ticks where the RTL and golden disagree

    def __call__(self, theta, omega, plans):
        theta_q, omega_q = planner.quantize_state(theta, omega)

        # plans[n][t] is candidate n's action at step t. The testbench wants
        # ACT[t][n] order (step outer, candidate inner); plans.T is (H, 32),
        # so reading it row by row gives exactly that.
        actions = " ".join(str(int(u)) for u in plans.T.ravel())
        signal.alarm(TIMEOUT_S)
        self.to_rtl.write(f"1 {theta_q} {omega_q} {actions}\n")
        self.to_rtl.flush()                 # send it now, not when a buffer fills
        reply = self.from_rtl.readline()    # waits for the RTL's answer
        signal.alarm(0)
        if not reply:
            raise RuntimeError("the simulator closed its pipe (its output is in the log)")
        best_idx, best_cost, cycles = (int(v) for v in reply.split())

        # the golden model on exactly the same inputs
        cost, _, _ = golden.int_rollout(theta_q, omega_q, plans, self.p)
        g_best = golden.argmin_first(cost)
        if (best_idx, best_cost) != (g_best, int(cost[g_best])):
            self.mismatches.append(
                f"tick {len(self.chosen)}: state ({theta_q}, {omega_q}): RTL best {best_idx} "
                f"(cost {best_cost}), golden best {g_best} (cost {int(cost[g_best])})")

        self.chosen.append(best_idx)
        self.cycles.append(cycles)
        return best_idx


def connect(pipe_dir, horizon):
    """Open both pipes, in the same order as tb_cosim.v (to_rtl, then
    from_rtl). open() on a named pipe waits until the simulator opens the
    other end, so this is also where we wait for it to start."""
    signal.alarm(TIMEOUT_S)
    to_rtl = open(pipe_dir / "to_rtl", "w")
    from_rtl = open(pipe_dir / "from_rtl", "r")
    # the testbench introduces itself with its LANES and H
    lanes, h = (int(v) for v in from_rtl.readline().split())
    signal.alarm(0)
    if h != horizon:
        raise SystemExit(f"the testbench was built with H = {h}, the host plans H = {horizon}")
    return to_rtl, from_rtl, lanes


def golden_closed_loop(p, seed, episodes, horizon):
    """The golden model's integer planner in closed loop, exactly as
    model/golden.py runs it. Returns each episode's return and the plan
    index it chose at every tick."""
    env = gym.make("Pendulum-v1")
    returns, chosen = [], []
    for e in range(episodes):
        records = []
        engine = golden.int_engine(p, record=records)
        returns.append(planner.run_episode(env, engine, seed, e, horizon))
        chosen.append([r["best"] for r in records])
    return returns, chosen


def main():
    ap = argparse.ArgumentParser(description="closed-loop co-simulation host (DESIGN.md §7.3)")
    ap.add_argument("--seed", type=int, default=0, help="seed (default 0, like golden.py)")
    ap.add_argument("--episodes", type=int, default=20, help="episodes (default 20)")
    ap.add_argument("--H", type=int, default=15, help="imagined steps per plan (default 15)")
    ap.add_argument("--pipes", default="sim/cosim",
                    help="directory holding the named pipes to_rtl and from_rtl")
    args = ap.parse_args()
    signal.signal(signal.SIGALRM, timed_out)

    p = golden.fold(golden.load_float_model())

    # 1. the reference: the golden model's own closed loop (a few seconds)
    t0 = time.time()
    g_returns, g_chosen = golden_closed_loop(p, args.seed, args.episodes, args.H)
    t_golden = time.time() - t0

    # 2. the same episodes with the RTL choosing every action
    to_rtl, from_rtl, lanes = connect(ROOT / args.pipes, args.H)
    print(f"Closed-loop co-simulation (DESIGN.md §7.3): gymnasium Pendulum-v1, seed {args.seed}, "
          f"H = {args.H}, {args.episodes} episodes x {planner.EPISODE_STEPS} ticks")
    print(f"RTL: rollout_engine (LANES = {lanes}) in Icarus Verilog, through named pipes; "
          f"its best_idx picks every applied action")
    print("golden: model/golden.py integer planner; both use the host code in model/planner.py")
    print()
    print(f"{'episode':>7}  {'RTL return':>22}  {'golden return':>22}  {'return':<8}  "
          f"chosen best_idx")

    engine = RtlEngine(to_rtl, from_rtl, p)
    env = gym.make("Pendulum-v1")
    rtl_returns, same_returns, same_ticks = [], 0, 0
    t0 = time.time()
    for e in range(args.episodes):
        first = len(engine.chosen)
        ret = planner.run_episode(env, engine, args.seed, e, args.H)
        rtl_returns.append(ret)
        chosen = engine.chosen[first:]
        # == on floats: identical to the last bit, no tolerance
        same = ret == g_returns[e]
        if same:
            same_returns += 1
        agree = sum(1 for a, b in zip(chosen, g_chosen[e]) if a == b)
        same_ticks += agree
        print(f"{e:>7}  {ret!r:>22}  {g_returns[e]!r:>22}  {'MATCH' if same else 'MISMATCH':<8}  "
              f"{agree}/{len(chosen)} equal to golden", flush=True)
    t_rtl = time.time() - t0

    # 3. tell the testbench there are no more ticks; it prints a summary and ends
    to_rtl.write("0\n")
    to_rtl.close()
    from_rtl.close()

    n = len(engine.chosen)
    ok = (same_returns == args.episodes and same_ticks == n and not engine.mismatches)
    print()
    for line in engine.mismatches[:5]:
        print("  " + line)
    print(f"decisions made by the RTL: {n} ({planner.N_CAND} candidates x {args.H} imagined steps each)")
    print(f"RTL best_idx and best_cost equal the golden model's on the same inputs: "
          f"{n - len(engine.mismatches)}/{n}")
    print(f"RTL chosen best_idx equal to the golden closed loop's: {same_ticks}/{n}")
    print(f"episode returns identical (float ==): {same_returns}/{args.episodes}")
    print(f"mean return: RTL {sum(rtl_returns) / len(rtl_returns):.1f}, "
          f"golden {sum(g_returns) / len(g_returns):.1f}")
    print(f"clocks per decision (RTL cycles output): {min(engine.cycles)}..{max(engine.cycles)}")
    print(f"wall time: RTL closed loop {t_rtl:.1f} s ({1000 * t_rtl / n:.1f} ms per decision), "
          f"golden closed loop {t_golden:.1f} s")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
