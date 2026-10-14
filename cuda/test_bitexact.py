"""
Bit-exact test of the CUDA rollout kernels against the golden model.

  1. Recorded decisions: every decision in each vector directory (default
     sim/ and sim/h5/, the files model/golden.py writes and
     tb/tb_rollout_engine.v replays) runs in ONE kernel launch. Every
     candidate's cost, every final (theta, omega) and best_idx / best_cost
     are compared with exp_*.hex. Both kernels: rollout_warp (one warp per
     decision, the RTL mirror) and rollout_blocks with 32-thread blocks.
  2. Random decisions with many candidates (N = 1024 and 4096, and N = 32
     where every plan appears twice, so every decision has a tie), states and
     actions drawn from the whole int8 range, and H = 1 and H = 30, against
     golden.int_rollout + golden.argmin_first. rollout_blocks with 256- and
     1,024-thread blocks, so the argmin crosses warps (shared memory) and
     blocks (atomicMin).
  3. Planted bugs: edited copies of rollout_kernel.cu (and one wrong
     CudaEngine argument), each run through all of the above. Every one
     must FAIL somewhere; a check that cannot fail proves nothing. The bugs
     in code that only rollout_blocks has (the argmin across warps and
     across blocks, the start value of best_key) must FAIL on one named
     run: the N = 4,096 set with 256-thread blocks, 16 blocks per decision.

Run from the repo root on the machine with the GPU:
  .venv\\Scripts\\python.exe cuda\\test_bitexact.py               (Windows)
  python3 cuda/test_bitexact.py                                  (Linux)
  --vec DIR (repeatable) replaces the default vector directories;
  --no-mutants skips part 3.
Prints PASS and exits 0, or FAIL and exits 1.
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cuda"))
sys.path.insert(0, str(ROOT / "model"))
import cuda_engine  # noqa: E402  (sets the CuPy cache directory before importing cupy)
import golden       # noqa: E402

# ------------------------------------------------------- planted bugs --
# The run that the rollout_blocks-only bugs below must FAIL: 16 blocks of 8
# warps per decision, so level 2 (shared memory, 8 warp winners) and level 3
# (atomicMin over 16 block winners) both decide something.
BLOCKS_RUN = "random N=4096 H=15, every plan 8 times [rollout_blocks x256]"

# (name, text in rollout_kernel.cu, replacement, run that must FAIL or None).
# Each text must occur exactly once. With None, any one failing run is enough.
MUTANTS = [
    ("logical >> instead of arithmetic in all three requantization shifts",
     "return acc >> s;",
     "return (int)((unsigned)acc >> s);", None),
    ("theta saturates at +-127 instead of wrapping",
     "theta = wrap_int8(theta + d_theta);",
     "theta = clamp_int(theta + d_theta, -127, 127);", None),
    ("omega wraps instead of saturating",
     "omega = clamp_int(omega + d_omega, -127, 127);",
     "omega = wrap_int8(omega + d_omega);", None),
    ("h clamped at 255 instead of 127 (h = 128..255 is a negative signed byte to layer 2's __dp4a)",
     "clamp_int(requant_shift(acc1, c_S1), 0, 127)",
     "clamp_int(requant_shift(acc1, c_S1), 0, 255)", None),
    ("argmin tie-break picks the higher index (the RTL's <= bug)",
     "(other_cost == cost && other_idx < idx)",
     "(other_cost == cost && other_idx > idx)", None),
    ("argmin tree compares costs only and keeps its own on a tie",
     "if (other_cost < cost || (other_cost == cost && other_idx < idx)) {",
     "if (other_cost < cost) {", None),
    ("level 2 (shared memory) reads only warp 0's winner",
     "const int n_warps = blockDim.x / WARP_SIZE;",
     "const int n_warps = 1;", BLOCKS_RUN),
    ("level 2: lanes with no warp winner to read get cost 0 instead of 0x7fffffff",
     "best_cost = (lane < n_warps) ? s_cost[lane] : 0x7fffffff;",
     "best_cost = (lane < n_warps) ? s_cost[lane] : 0;", BLOCKS_RUN),
    ("level 3: atomicMin key with the index in the high half, (idx << 32) | cost",
     "((unsigned long long)(unsigned)best_cost << 32) | (unsigned)best_idx;",
     "((unsigned long long)(unsigned)best_idx << 32) | (unsigned)best_cost;", BLOCKS_RUN),
    ("level 3: atomicMin key without the index, cost << 32 only",
     "<< 32) | (unsigned)best_idx;",
     "<< 32);", BLOCKS_RUN),
]

# (name, CudaEngine keyword arguments, run that must FAIL): bugs on the host
# side, with the kernel source left as it is.
HOST_MUTANTS = [
    ("best_key starts at 0 instead of 2^64 - 1 (a memset of 0x00 instead of 0xff): no atomicMin can lower it",
     {"key_init": 0}, BLOCKS_RUN),
]


# -------------------------------------------------------- vector files --

def read_lanes(path, bits, signed):
    """A file of equal-width hex lines, lane 0 in the lowest bits (so last in
    the text), as an int64 array (lines, lanes). split() also drops the '\\r'
    of Windows line endings."""
    lines = Path(path).read_text().split()
    assert len({len(s) for s in lines}) == 1, f"{path}: lines of different widths"
    # bytes.fromhex gives the most significant byte first; reversing a whole
    # line puts lane 0's least significant byte first
    raw = np.frombuffer(bytes.fromhex("".join(lines)), np.uint8).reshape(len(lines), -1)[:, ::-1]
    nb = bits // 8
    raw = raw.reshape(len(lines), -1, nb).astype(np.int64)
    v = sum(raw[..., i] << (8 * i) for i in range(nb))
    if signed:
        v = np.where(v >= 1 << (bits - 1), v - (1 << bits), v)
    return v


def load_vectors(vec_dir):
    """The decision vectors golden.py writes (format: top of model/golden.py)."""
    vh = (vec_dir / "vectors.vh").read_text()
    count = int(re.search(r"VEC_COUNT\s*=\s*(\d+)", vh).group(1))
    horizon = int(re.search(r"VEC_H\s*=\s*(\d+)", vh).group(1))
    state = read_lanes(vec_dir / "vec_state.hex", 8, signed=True)       # (D, 2): theta, omega
    act = read_lanes(vec_dir / "vec_act.hex", 8, signed=True)           # (D*H, 32)
    best = read_lanes(vec_dir / "exp_best.hex", 32, signed=False)[:, 0]
    v = {
        "theta0": state[:, 0], "omega0": state[:, 1],
        # line d*H + t is ACT[t] of decision d -> plans[d][n][t]
        "plans": act.reshape(count, horizon, 32).transpose(0, 2, 1),
        "cost": read_lanes(vec_dir / "exp_cost.hex", 24, signed=False),
        "theta": read_lanes(vec_dir / "exp_theta.hex", 8, signed=True),
        "omega": read_lanes(vec_dir / "exp_omega.hex", 8, signed=True),
        "best_idx": best >> 24,                 # {best_idx[7:0], best_cost[23:0]}
        "best_cost": best & 0xFFFFFF,
        "H": horizon,
    }
    for key in ("theta0", "cost", "theta", "omega", "best_idx"):
        assert len(v[key]) == count, f"{vec_dir}: {key} has {len(v[key])} entries, not {count}"
    return v


# ----------------------------------------------------- reference sets --

def golden_reference(theta0, omega0, plans, p):
    """golden.int_rollout + golden.argmin_first for every decision, and the
    corner cases they hit."""
    corners = golden.Corners()
    cost, theta, omega, best = [], [], [], []
    for d in range(len(plans)):
        c, th, om = golden.int_rollout(int(theta0[d]), int(omega0[d]), plans[d], p, corners)
        corners.add_decision(c)
        cost.append(c)
        theta.append(th)
        omega.append(om)
        best.append(golden.argmin_first(c))
    cost = np.array(cost)
    best = np.array(best)
    return {"theta0": np.asarray(theta0), "omega0": np.asarray(omega0), "plans": plans,
            "cost": cost, "theta": np.array(theta), "omega": np.array(omega),
            "best_idx": best, "best_cost": cost[np.arange(len(best)), best],
            "H": plans.shape[2]}, corners


def random_set(rng, n_dec, n_cand, horizon, copies):
    """Random states and actions over the whole int8 range. With copies > 1,
    n_cand / copies distinct plans each appear `copies` times at random
    positions, so every cost (the lowest included) is shared by several
    candidates: the argmin's tie-break decides every one of them."""
    theta0 = rng.integers(-128, 128, n_dec)
    omega0 = rng.integers(-128, 128, n_dec)
    distinct = rng.integers(-128, 128, (n_dec, n_cand // copies, horizon))
    plans = np.tile(distinct, (1, copies, 1))
    for d in range(n_dec):
        plans[d] = plans[d][rng.permutation(n_cand)]
    return theta0, omega0, plans


def corner_text(c):
    return (f"h clamped at 127: {c.h_clamped:,}, theta wraps: {c.theta_wraps:,}, "
            f"omega saturations: {c.omega_sats:,}, decisions with a tied best cost: {c.ties:,}")


# -------------------------------------------------------------- checks --

def compare(got, want):
    """Per-decision counts, like tb_rollout_engine.v prints them."""
    d = len(want["best_idx"])
    cost_ok = (got.cost == want["cost"]).all(axis=1)
    state_ok = ((got.theta == want["theta"]) & (got.omega == want["omega"])).all(axis=1)
    best_ok = (got.best_idx == want["best_idx"]) & (got.best_cost == want["best_cost"])
    r = {"n": d, "values": int(want["cost"].size),
         "cost": int(cost_ok.sum()), "state": int(state_ok.sum()), "best": int(best_ok.sum()),
         "bad_values": int((got.cost != want["cost"]).sum() + (got.theta != want["theta"]).sum()
                           + (got.omega != want["omega"]).sum())}
    r["ok"] = r["cost"] == r["state"] == r["best"] == d
    if not best_ok.all():
        i = int(np.flatnonzero(~best_ok)[0])
        r["first_best"] = (f"decision {i}: best {got.best_idx[i]} (cost {got.best_cost[i]}), "
                           f"golden {want['best_idx'][i]} (cost {want['best_cost'][i]})")
    return r


def kernels_for(n_cand):
    """(label, kernel, threads per block) to run on a set with N candidates."""
    if n_cand == 32:
        return [("rollout_warp", "warp", None), ("rollout_blocks x32", "blocks", 32)]
    return [("rollout_blocks x256", "blocks", 256), ("rollout_blocks x1024", "blocks", 1024)]


def run_all(engine, sets):
    """Every set through every kernel that fits it. Returns a list of
    (set name, kernel label, compare() result)."""
    results = []
    for name, want in sets:
        for label, kernel, threads in kernels_for(want["plans"].shape[1]):
            got = engine.decide_batch(want["theta0"], want["omega0"], want["plans"], kernel, threads)
            results.append((name, label, compare(got, want)))
    return results


def check_bug(what, bad, sets, must_fail):
    """Run every set through an engine with a planted bug and print where it
    fails. Caught = at least one run FAILs, or, if must_fail names a run,
    that run FAILs."""
    runs = run_all(bad, sets)
    labels = [f"{name} [{label}]" for name, label, _ in runs]
    failed = [(lab, r) for lab, (_, _, r) in zip(labels, runs) if not r["ok"]]
    passed = [lab for lab, (_, _, r) in zip(labels, runs) if r["ok"]]
    if must_fail is None:
        caught = len(failed) > 0
    else:
        assert must_fail in labels, f"no run called {must_fail}"
        caught = must_fail not in passed
    print(f"  {'caught' if caught else 'MISSED'}: {what}: {len(failed)} of {len(runs)} set/kernel runs FAIL"
          + (f"; must FAIL on {must_fail}: {'yes' if caught else 'NO'}" if must_fail else ""))
    if failed:
        # the named run if there is one and it failed, otherwise the first failure
        lab, r = next(((lab, r) for lab, r in failed if lab == must_fail), failed[0])
        n = r["n"]
        print(f"      {'on' if lab == must_fail else 'first'}: {lab}: costs {r['cost']}/{n}, "
              f"final states {r['state']}/{n}, best {r['best']}/{n}")
    if passed:
        print(f"      still PASS on: {', '.join(passed)}")
    return caught


def main():
    ap = argparse.ArgumentParser(description="bit-exact test of cuda/rollout_kernel.cu")
    ap.add_argument("--vec", action="append", default=None,
                    help="vector directory, relative to the repo root (repeatable; default sim and sim/h5)")
    ap.add_argument("--no-mutants", action="store_true", help="skip the planted-bug runs")
    ap.add_argument("--save-ptx", default=None,
                    help="also write the kernels' PTX to this file (relative to the repo root)")
    args = ap.parse_args()
    vec_dirs = args.vec or ["sim", "sim/h5"]
    t_start = time.time()

    print("CUDA bit-exact test: cuda/rollout_kernel.cu against model/golden.py (DESIGN.md section 4)")
    print("command: " + " ".join([Path(sys.executable).name] + sys.argv))
    for line in cuda_engine.versions():
        print(line)

    # the ROM the kernel loads must be what golden.fold() computes
    p = golden.fold(golden.load_float_model())
    eng = cuda_engine.CudaEngine()
    rom = eng.rom
    same = all(np.array_equal(rom[k], p[k]) for k in ("W1", "B1", "W2", "B2")) \
        and rom["S1"] == p["S1"] and tuple(rom["S2"]) == tuple(p["S2"])
    print(f"rom/ equals golden.fold(): {'yes' if same else 'NO'} "
          f"(S1 = {rom['S1']}, S2 = {tuple(rom['S2'])}, B2 = {[int(b) for b in rom['B2']]})")
    if not same:
        print("FAIL")
        return 1

    print("\nPTX the compiler emitted (counts per kernel; 'step loop' = PTX instructions in the loop "
          "over the H imagined steps, its branch included):")
    for kname, counts in cuda_engine.ptx_summary().items():
        print(f"  {kname}: " + ", ".join(f"{pat} {n}" for pat, n in counts.items()))
    if args.save_ptx:
        (ROOT / args.save_ptx).write_text(cuda_engine.ptx_text(), encoding="utf-8", newline="\n")
        print(f"  wrote the PTX to {args.save_ptx}")

    # ---- the sets ----------------------------------------------------
    sets = []
    print("\nRecorded decisions (expected values from the exp_*.hex files):")
    for vd in vec_dirs:
        want = load_vectors(ROOT / vd)
        # replaying them through golden.int_rollout recounts their corner
        # cases and re-checks that the file is this golden model's output
        ref, corners = golden_reference(want["theta0"], want["omega0"], want["plans"], p)
        agrees = all(np.array_equal(ref[k], want[k])
                     for k in ("cost", "theta", "omega", "best_idx", "best_cost"))
        print(f"  {vd}: {len(want['best_idx'])} decisions x 32 candidates, H = {want['H']}; "
              f"golden.int_rollout reproduces the file: {'yes' if agrees else 'NO'}")
        print(f"    {corner_text(corners)}")
        sets.append((vd, want))

    rng = np.random.default_rng(2026_10_13)
    random_specs = [   # (name, decisions, N, H, copies of each plan)
        ("random N=32 H=15, every plan twice", 1000, 32, 15, 2),
        ("random N=32 H=1", 200, 32, 1, 1),
        ("random N=1024 H=15", 16, 1024, 15, 1),
        ("random N=1024 H=30", 4, 1024, 30, 1),
        ("random N=4096 H=15, every plan 8 times", 4, 4096, 15, 8),
    ]
    print("\nRandom decisions (expected values from golden.int_rollout + golden.argmin_first):")
    for name, n_dec, n_cand, horizon, copies in random_specs:
        theta0, omega0, plans = random_set(rng, n_dec, n_cand, horizon, copies)
        want, corners = golden_reference(theta0, omega0, plans, p)
        print(f"  {name}: {n_dec} decisions x {n_cand} candidates")
        print(f"    {corner_text(corners)}")
        sets.append((name, want))

    # ---- the real kernel -----------------------------------------------
    print("\nResults, one launch per (set, kernel); 'decisions' = all N candidates of a decision equal:")
    results = run_all(eng, sets)
    ok = True
    for name, label, r in results:
        n = r["n"]
        print(f"  {name} [{label}]: costs {r['cost']}/{n}, final states {r['state']}/{n}, "
              f"best_idx and best_cost {r['best']}/{n}  {'PASS' if r['ok'] else 'FAIL'}")
        if "first_best" in r:
            print(f"    first wrong best: {r['first_best']}")
        ok = ok and r["ok"]
    # Each candidate is checked once per kernel that runs its set (two
    # kernels per set), so the total counts every candidate twice.
    total = sum(r["values"] for _, _, r in results)
    recorded = sum(r["values"] for name, _, r in results if name in vec_dirs)
    print(f"  candidate results compared (cost, theta and omega each), over all sets and both kernels: "
          f"{total:,} ({sum(want['cost'].size for _, want in sets):,} distinct candidates, each on two "
          f"kernels; recorded sets {recorded:,}, random sets {total - recorded:,}); "
          f"wrong: {sum(r['bad_values'] for _, _, r in results)}")

    # ---- planted bugs ----------------------------------------------------
    if not args.no_mutants:
        print("\nPlanted bugs (each must FAIL at least one check above; where a run is named, that run):")
        source = cuda_engine.KERNEL_PATH.read_text(encoding="utf-8")
        for what, old, new, must_fail in MUTANTS:
            assert source.count(old) == 1, f"mutant text not found exactly once: {old}"
            bad = cuda_engine.CudaEngine(source=source.replace(old, new))
            ok = check_bug(what, bad, sets, must_fail) and ok
        for what, kwargs, must_fail in HOST_MUTANTS:
            bad = cuda_engine.CudaEngine(**kwargs)
            ok = check_bug(what, bad, sets, must_fail) and ok

    print(f"\n({time.time() - t_start:.1f} s)")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
