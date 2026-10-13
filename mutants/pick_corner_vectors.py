"""
Picks the decisions the planted-bug tests run (DESIGN.md §7.4).

Why not just the first 300 decisions in sim/? Because they are too gentle.
The first 300 decisions never clamp a hidden value at 127, never saturate
omega and never have two candidates tied for the lowest cost. A planted bug
in exactly that logic (clamp h at 255, wrap omega, tie-break with <=) would
behave the same as the correct design on them, and PASS. Not because the
bug is harmless, but because the test never asks the question.

So this script reads the 4,000 recorded decisions in sim/, replays each one
through the integer model to see which corner cases it hits, and writes N
of them to a new vector directory:

  1. every decision that clamps an h, saturates omega or has a tied best
     cost (these are rare: under 80 of the 4,000)
  2. then the earliest remaining decisions, to fill up to N

The chosen lines are copied unchanged, so the expected values are still the
golden model's own outputs. The replay also checks that every recorded
expectation is reproduced exactly (the same check golden.py makes when it
writes the vectors).

Run from the repo root (mutants/run_mutants.sh does this):
  python3 mutants/pick_corner_vectors.py --n 300 --out sim/mutants/vec
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
import golden  # noqa: E402  (the integer model the RTL must match)

N_CAND = golden.planner.N_CAND   # 32 candidates per decision

# one line per decision in these files, H lines per decision in vec_act.hex
PER_DECISION = ["vec_state.hex", "exp_cost.hex", "exp_theta.hex", "exp_omega.hex",
                "exp_best.hex"]


def unpack(line, bits, count):
    """The inverse of golden.pack(): field 0 is in the lowest bits. Fields
    are read as signed (two's complement)."""
    word = int(line, 16)
    out = []
    for k in range(count):
        v = (word >> (bits * k)) & ((1 << bits) - 1)
        out.append(v - (1 << bits) if v >= 1 << (bits - 1) else v)
    return out


def read_header(vec_dir):
    """VEC_COUNT and VEC_H from vectors.vh, written by golden.py."""
    text = (vec_dir / "vectors.vh").read_text()
    count = int(re.search(r"VEC_COUNT\s*=\s*(\d+)", text).group(1))
    horizon = int(re.search(r"VEC_H\s*=\s*(\d+)", text).group(1))
    return count, horizon


def corner_table(vec_dir, count, horizon, p):
    """Replay every decision; return one row of corner-case counts per
    decision: [h clamps, theta wraps, omega saturations, tied best (0/1)]."""
    state = (vec_dir / "vec_state.hex").read_text().split()
    act = (vec_dir / "vec_act.hex").read_text().split()
    exp_cost = (vec_dir / "exp_cost.hex").read_text().split()
    exp_theta = (vec_dir / "exp_theta.hex").read_text().split()
    exp_omega = (vec_dir / "exp_omega.hex").read_text().split()

    rows = []
    for d in range(count):
        theta0, omega0 = unpack(state[d], 8, 2)
        # vec_act line d*H + t holds all 32 lanes' actions at step t;
        # int_rollout wants plans[lane][t], hence the transpose
        plans = np.array([unpack(act[d * horizon + t], 8, N_CAND)
                          for t in range(horizon)], dtype=np.int64).T
        corners = golden.Corners()
        cost, theta, omega = golden.int_rollout(theta0, omega0, plans, p, corners)
        # costs are unsigned: unpack reads them signed, so mask back to 24 bits
        assert [c & 0xFFFFFF for c in unpack(exp_cost[d], 24, N_CAND)] == cost.tolist(), d
        assert unpack(exp_theta[d], 8, N_CAND) == theta.tolist(), d
        assert unpack(exp_omega[d], 8, N_CAND) == omega.tolist(), d
        tied = int((cost == cost.min()).sum() > 1)
        rows.append([corners.h_clamped, corners.theta_wraps, corners.omega_sats, tied])
    return np.array(rows)


def coverage(rows):
    """One line: how many decisions hit each corner case (and how often)."""
    hit = rows > 0
    return (f"{len(rows)} decisions: "
            f"h clamped at 127 in {hit[:, 0].sum()} ({rows[:, 0].sum()} times), "
            f"theta wraps in {hit[:, 1].sum()} ({rows[:, 1].sum()} times), "
            f"omega saturates in {hit[:, 2].sum()} ({rows[:, 2].sum()} times), "
            f"tied best cost in {hit[:, 3].sum()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=300, help="decisions to pick (default 300)")
    ap.add_argument("--src", default="sim", help="recorded vectors (default sim)")
    ap.add_argument("--out", default="sim/mutants/vec", help="where to write the picked ones")
    args = ap.parse_args()

    src, out = ROOT / args.src, ROOT / args.out
    count, horizon = read_header(src)
    assert args.n <= count, f"only {count} decisions in {args.src}"

    # the same folding the RTL's rom/ came from
    p = golden.fold(golden.load_float_model())
    rows = corner_table(src, count, horizon, p)

    # 1. every decision with a rare corner case: h clamp, omega saturation, tie
    rare = [d for d in range(count) if rows[d, 0] or rows[d, 2] or rows[d, 3]]
    assert len(rare) <= args.n, "more rare decisions than --n"
    # 2. fill with the earliest other decisions; keep the recorded order
    rare_set = set(rare)
    others = [d for d in range(count) if d not in rare_set]
    picked = sorted(rare + others[:args.n - len(rare)])

    out.mkdir(parents=True, exist_ok=True)
    for name in PER_DECISION:
        lines = (src / name).read_text().split()
        (out / name).write_text("".join(lines[d] + "\n" for d in picked))
    act = (src / "vec_act.hex").read_text().split()
    (out / "vec_act.hex").write_text(
        "".join(act[d * horizon + t] + "\n" for d in picked for t in range(horizon)))
    (out / "vectors.vh").write_text(
        f"// picked by mutants/pick_corner_vectors.py --n {args.n} from {args.src}/\n"
        f"localparam VEC_COUNT = {len(picked)};\n"
        f"localparam VEC_H     = {horizon};\n")

    print(f"recorded vectors ({args.src}/, H = {horizon}), all {coverage(rows)}")
    print(f"first {args.n} ({args.src}/ as recorded): {coverage(rows[:args.n])}")
    print(f"corner {args.n} ({args.out}/): {coverage(rows[picked])}")
    print(f"  = all {len(rare)} decisions with an h clamp, an omega saturation or a tie, "
          f"plus the first {args.n - len(rare)} others (decisions {picked[0]}..{picked[-1]})")


if __name__ == "__main__":
    main()
