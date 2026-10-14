"""
Timing of the CUDA rollout kernels, with CUDA events, and of the NumPy
golden model on the same machine and the same inputs.

  (a) one decision, N = 32, H = 15: kernel time; end-to-end time of
      eng.decide() and of a minimal round trip (actions in, best index out)
  (b) the 4,000 recorded decisions of sim/ in one launch: decisions per second
  (c) one decision with N = 32 ... 65,536 candidates: time vs N
  (d) H = 1, 5, 15, 30 imagined steps at fixed N: time vs H
  (e) NumPy golden.int_rollout on the CPU, the same workloads

How the GPU is timed. A CUDA event is a marker the host drops into the
GPU's queue of work; the GPU writes a timestamp when it reaches it, and
cp.cuda.get_elapsed_time(a, b) is the GPU time between two markers. Two
ways of using them:
  "graph"     100 launches are recorded once into a CUDA graph and then
              replayed with one call, between two events; time / 100. The
              host is not involved between the launches, so this is close
              to the kernel's own run time (plus the ~1 us the GPU needs to
              start the next kernel).
  "one launch" an event before and after a single ordinary launch, each
              launch waited for before the next: what one control tick
              sees on the GPU, including the time the driver needs to hand
              the launch to the GPU (on Windows, the WDDM driver model).
Each figure is the median of 20 graph replays or 100 to 200 single
launches, after priming.

The GPU clock is not locked (that needs admin rights), and an idle GPU
drops to a slow power state within a fraction of a second. So every GPU
series is preceded by 50 ms of back-to-back launches of the same kernel
("priming"), and all GPU series run before any CPU timing. After each graph
series, nvidia-smi is asked for the SM clock while that graph replays for
another 0.2 s or so, so the log shows the clock the series ran at (nvidia-smi
samples it; it is not a lock). Readings between series are printed too.

Before anything touches the GPU, the script lists the processes that hold a
GPU context and the GPU's utilization. If the GPU is busy (utilization above
--max-util) it stops. If another Python process holds a context (on the
machine these logs come from, a long-running service may), it also stops,
unless --shared-gpu is given. A run keeps the GPU busy with short kernels
(the longest, 4,000 decisions in one launch, takes under 0.1 ms, and a graph
replays at most 100 launches, a few ms) for about 6 s in total, most of it
priming and clock sampling; the footer prints the totals.

Run from the repo root on the machine with the GPU:
  .venv\\Scripts\\python.exe cuda\\bench.py        (Windows)
  python3 cuda/bench.py                           (Linux)
"""

import argparse
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cuda"))
sys.path.insert(0, str(ROOT / "model"))
import cuda_engine    # noqa: E402  (sets the CuPy cache directory before importing cupy)
import golden         # noqa: E402
import test_bitexact  # noqa: E402  (load_vectors)
import cupy as cp     # noqa: E402

GPU_MS = [0.0]        # total GPU time inside event pairs, for the footer
PRIME_TOTAL_S = [0.0] # total wall time spent priming, for the footer
PRIME_S = 0.05        # seconds of back-to-back launches before each GPU series
GRAPH_K = 100         # launches per CUDA graph


# ------------------------------------------------------------ nvidia-smi --

def nvidia_smi(*args):
    r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=60)
    return r.stdout.strip()


def gpu_state():
    return nvidia_smi("--query-gpu=utilization.gpu,clocks.sm,clocks.mem,pstate,temperature.gpu,"
                      "power.draw,memory.used", "--format=csv,noheader")


def compute_apps():
    """[(pid, process name)] of every process holding a GPU context."""
    out = []
    for line in nvidia_smi("--query-compute-apps=pid,process_name", "--format=csv,noheader").splitlines():
        if "," in line:
            pid, name = line.split(",", 1)
            out.append((int(pid), name.strip()))
    return out


def guard(args):
    """Stop before touching the GPU if someone else is using it."""
    apps = compute_apps()
    util = int(nvidia_smi("--query-gpu=utilization.gpu", "--format=csv,noheader,nounits").split()[0])
    pythons = [(pid, name) for pid, name in apps if "python" in name.lower() and pid != os.getpid()]
    print(f"processes holding a GPU context: {len(apps)} (on Windows this includes the desktop and "
          f"other apps that draw with the GPU); Python ones other than this script: "
          + (", ".join(f"PID {pid} {name}" for pid, name in pythons) or "none"))
    print(f"GPU utilization before the run: {util} %")
    if util > args.max_util:
        print(f"GPU busy ({util} % > {args.max_util} %): not benchmarking")
        sys.exit(2)
    if pythons and not args.shared_gpu:
        print("another Python process holds a GPU context: not benchmarking "
              "(--shared-gpu runs this light benchmark anyway)")
        sys.exit(2)
    if pythons:
        print(f"--shared-gpu: running with the process(es) above resident; utilization was {util} %, "
              "so they were idle at the start, but work of theirs could overlap ours")


def cpu_name():
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


# ----------------------------------------------------------------- timing --

def prime(launch):
    """Keep the GPU busy for PRIME_S seconds so it is at its boost clock."""
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < PRIME_S:
        launch()
        n += 1
        if n % 64 == 0:
            cp.cuda.Device().synchronize()
    cp.cuda.Device().synchronize()
    PRIME_TOTAL_S[0] += time.perf_counter() - t0


def clock_while_busy(launch, seconds=0.2):
    """nvidia-smi's SM clock and P-state read while launch() keeps the GPU
    busy: a second thread asks nvidia-smi a third of the way in, while this
    one goes on launching. (The other readings are taken between series,
    when the GPU may already be dropping to idle clocks.)"""
    reading = {}

    def sample():
        time.sleep(seconds / 3)
        reading["smi"] = nvidia_smi("--query-gpu=clocks.sm,pstate,utilization.gpu",
                                    "--format=csv,noheader")
    sampler = threading.Thread(target=sample)
    sampler.start()
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < seconds or sampler.is_alive():
        launch()
        n += 1
        if n % 16 == 0:
            cp.cuda.Device().synchronize()
    cp.cuda.Device().synchronize()
    sampler.join()
    PRIME_TOTAL_S[0] += time.perf_counter() - t0
    return reading["smi"]


def one_launch_ms(launch, reps):
    """GPU time of single launches, an event pair around each, in ms."""
    prime(launch)
    a, b = cp.cuda.Event(), cp.cuda.Event()
    out = []
    for _ in range(reps):
        a.record()
        launch()
        b.record()
        b.synchronize()        # wait until the GPU has passed marker b
        out.append(cp.cuda.get_elapsed_time(a, b))
    GPU_MS[0] += sum(out)
    return out


def graph_ms(launch, reps=20, k=GRAPH_K, clock=False):
    """GPU time per launch when k launches are replayed from one CUDA graph.
    Stream capture records the launches into a graph instead of running
    them; graph.launch() then hands all k to the GPU in one call. With
    clock=True also returns nvidia-smi's reading while the graph replays."""
    launch()                                   # anything lazy happens outside the capture
    cp.cuda.Device().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:                               # launches inside go to this stream
        stream.begin_capture()
        for _ in range(k):
            launch()
        graph = stream.end_capture()
    prime(lambda: graph.launch(stream))
    a, b = cp.cuda.Event(), cp.cuda.Event()
    out = []
    for _ in range(reps):
        a.record(stream)
        graph.launch(stream)
        b.record(stream)
        b.synchronize()
        total = cp.cuda.get_elapsed_time(a, b)
        GPU_MS[0] += total
        out.append(total / k)
    if clock:
        return out, clock_while_busy(lambda: graph.launch(stream))
    return out


def back_to_back_ms(launch, launches):
    """GPU time per launch with `launches` ordinary launches queued between
    two events. The GPU only runs them without gaps if Python queues them
    faster than the GPU finishes them; otherwise this is Python's rate."""
    prime(launch)
    a, b = cp.cuda.Event(), cp.cuda.Event()
    a.record()
    for _ in range(launches):
        launch()
    b.record()
    b.synchronize()
    total = cp.cuda.get_elapsed_time(a, b)
    GPU_MS[0] += total
    return total / launches


def wall_ms(fn, reps, warmup):
    """Host wall time of fn() (which must wait for its own results), in ms."""
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(1000 * (time.perf_counter() - t0))
    return out


def med(xs):
    return statistics.median(xs)


def us(ms):
    return f"{1000 * ms:,.2f} us"


def random_decision(rng, n_cand, horizon):
    """One decision: (theta0, omega0, plans (1, N, H)) in the host's ranges."""
    return (rng.integers(-128, 128, 1), rng.integers(-127, 128, 1),
            rng.integers(-127, 128, (1, n_cand, horizon)))


# ------------------------------------------------------------------ main --

def main():
    ap = argparse.ArgumentParser(description="CUDA-event timing of the rollout kernels")
    ap.add_argument("--shared-gpu", action="store_true",
                    help="run even if another Python process holds a GPU context")
    ap.add_argument("--max-util", type=int, default=10,
                    help="stop if GPU utilization is above this percentage (default 10)")
    args = ap.parse_args()
    t_start = time.perf_counter()

    print("CUDA rollout benchmark (cuda/bench.py)")
    print("command: " + " ".join([Path(sys.executable).name] + sys.argv))
    print(time.strftime("date: %Y-%m-%d %H:%M:%S") + f", CPU: {cpu_name()}, {platform.platform()}")
    guard(args)
    print("nvidia-smi before (util, SM clock, memory clock, P-state, temp, power, memory used): "
          + gpu_state())

    eng = cuda_engine.CudaEngine()
    p = golden.fold(golden.load_float_model())
    for line in cuda_engine.versions():
        print(line)
    props = cp.cuda.runtime.getDeviceProperties(0)
    n_sm = props["multiProcessorCount"]
    print(f"GPU limits: clockRate {props['clockRate'] / 1e6:.3f} GHz, "
          f"{props['maxThreadsPerMultiProcessor']} threads and {props['maxBlocksPerMultiProcessor']} "
          f"blocks per SM, {props['regsPerMultiprocessor']:,} registers per SM, "
          f"driver model {'TCC' if props['tccDriver'] else 'WDDM'}")
    drv = cp.cuda.driver
    for fn, threads in ((eng.k_warp, 32), (eng.k_blocks, 256)):
        blocks = drv.occupancyMaxActiveBlocksPerMultiprocessor(fn.kernel.ptr, threads, 0)
        print(f"kernel {fn.name}: {fn.num_regs} registers per thread, {fn.local_size_bytes} bytes "
              f"of local memory (spills), {fn.shared_size_bytes} bytes shared; with {threads}-thread "
              f"blocks the driver fits {blocks} blocks = {blocks * threads:,} threads per SM, "
              f"{blocks * threads * n_sm:,} on the GPU at once")

    rng = np.random.default_rng(0)
    work = {}   # name -> (theta0, omega0, plans), reused by the NumPy part

    # warm-up: ~0.3 s of the largest launch, so the GPU leaves its idle clocks
    big = cuda_engine.DeviceBatch(*random_decision(rng, 65536, 15))
    t0 = time.perf_counter()
    n_warm = 0
    while time.perf_counter() - t0 < 0.3:
        eng.launch(big)
        n_warm += 1
        if n_warm % 64 == 0:
            cp.cuda.Device().synchronize()
    cp.cuda.Device().synchronize()
    print(f"warm-up: {n_warm} launches of N = 65,536, H = 15 in {time.perf_counter() - t0:.2f} s; "
          f"nvidia-smi right after: {gpu_state()}")

    # ---- (a) ------------------------------------------------------------
    print("\n(a) one decision, N = 32, H = 15")
    work["a"] = th, om, plans = random_decision(rng, 32, 15)
    db = cuda_engine.DeviceBatch(th, om, plans)
    for label, kernel, threads in (("rollout_warp", "warp", None), ("rollout_blocks x32", "blocks", 32)):
        launch = lambda: eng.launch(db, kernel, threads)  # noqa: E731
        g, smi = graph_ms(launch, clock=True)
        t = one_launch_ms(launch, reps=200)
        b2b = back_to_back_ms(launch, launches=1000)
        extra = " (each with the fill of best_key, a second, tiny kernel)" if kernel == "blocks" else ""
        print(f"  {label}{extra}:")
        print(f"    graph: {us(med(g))} per launch (min {us(min(g))}); one launch: median {us(med(t))} "
              f"(min {us(min(t))}, {len(t)} launches); 1,000 ordinary launches back to back: "
              f"{us(b2b)} per launch")
        print(f"    nvidia-smi while this graph replayed (SM clock, P-state, util): {smi}")
    g_fill = graph_ms(lambda: db.best_key.fill(cuda_engine.KEY_INIT))
    g_memset = graph_ms(lambda: db.best_key.data.memset_async(0xFF, db.best_key.nbytes))
    print(f"  setting best_key to 2^64 - 1 on its own, graph: fill() {us(med(g_fill))}, "
          f"cudaMemsetAsync {us(med(g_memset))} per call (why cuda_engine.py uses fill)")

    # minimal round trip with buffers allocated once: the actions and the
    # state go in, best_idx comes out (4 bytes), nothing else
    act_host = np.ascontiguousarray(plans.transpose(0, 2, 1).astype(np.int8))
    state_host = np.array([[th[0], om[0]]], np.int8)

    def round_trip():
        db.act.set(act_host)          # host -> GPU
        db.state0.set(state_host)
        eng.launch(db, "warp")
        return db.best_idx.get()      # GPU -> host, waits for the kernel
    prime(round_trip)
    t = wall_ms(round_trip, reps=200, warmup=0)
    print(f"  minimal round trip, host wall time (buffers allocated once): copy 480 action bytes and the "
          f"state to the GPU, launch rollout_warp, copy best_idx back: median {us(med(t))} "
          f"(min {us(min(t))}, 200 calls)")
    prime(lambda: eng.decide(int(th[0]), int(om[0]), plans[0]))
    t = wall_ms(lambda: eng.decide(int(th[0]), int(om[0]), plans[0]), reps=200, warmup=0)
    print(f"  end to end, eng.decide() (what closed_loop.py calls every tick), host wall time: repack the "
          f"plans, allocate, copy in, launch, copy back 32 costs, 32 final states and the best "
          f"(5 copies): median {us(med(t))} (min {us(min(t))}, 200 calls)")

    # ---- (b) ------------------------------------------------------------
    print("\n(b) the 4,000 recorded decisions of sim/ (N = 32, H = 15) in one launch of rollout_warp "
          "(4,000 blocks of one warp)")
    v = test_bitexact.load_vectors(ROOT / "sim")
    db4k = cuda_engine.DeviceBatch(v["theta0"], v["omega0"], v["plans"])
    g = graph_ms(lambda: eng.launch(db4k, "warp"), reps=20, k=20)
    t = one_launch_ms(lambda: eng.launch(db4k, "warp"), reps=100)
    print(f"  graph (20 launches): {med(g):.4f} ms per launch = {4000 / med(g) * 1000:,.0f} decisions/s; "
          f"one launch: median {med(t):.4f} ms (min {min(t):.4f} ms, {len(t)} launches) "
          f"= {4000 / med(t) * 1000:,.0f} decisions/s")
    got = eng.decide_batch(v["theta0"], v["omega0"], v["plans"])
    exact = np.array_equal(got.cost, v["cost"]) and np.array_equal(got.best_idx, v["best_idx"])
    prime(lambda: eng.launch(db4k, "warp"))
    t = wall_ms(lambda: eng.decide_batch(v["theta0"], v["omega0"], v["plans"]), reps=20, warmup=2)
    print(f"  end to end, eng.decide_batch(), host wall time: repack and copy 1.92 MB of actions to the "
          f"GPU, launch, copy back 3 x 512 KB of results: median {med(t):.2f} ms (min {min(t):.2f} ms, "
          f"{len(t)} calls) = {4000 / med(t) * 1000:,.0f} decisions/s; results equal exp_*.hex: "
          f"{'yes' if exact else 'NO'}")

    # ---- (c) ------------------------------------------------------------
    sizes = (32, 256, 1024, 4096, 16384, 65536)
    print("\n(c) one decision, H = 15, N candidates: rollout_blocks, min(N, 256)-thread blocks")
    print(f"  {'N':>7} {'blocks':>6} {'graph':>11} {'ns/candidate':>13} {'one launch median':>18}"
          f"   nvidia-smi while the graph replayed")
    for n in sizes:
        work[("c", n)] = th, om, plans = random_decision(rng, n, 15)
        dbn = cuda_engine.DeviceBatch(th, om, plans)
        threads = min(n, 256)
        launch = lambda: eng.launch(dbn, "blocks", threads)  # noqa: E731
        g, smi = graph_ms(launch, clock=True)
        t = one_launch_ms(launch, reps=100)
        print(f"  {n:>7,} {n // threads:>6} {us(med(g)):>11} {1e6 * med(g) / n:>13,.2f} {us(med(t)):>18}"
              f"   {smi}")

    # ---- (d) ------------------------------------------------------------
    print("\n(d) imagined steps H, one decision")
    for n, kernel in ((32, "warp"), (4096, "blocks")):
        label = "rollout_warp" if kernel == "warp" else "rollout_blocks x256"
        print(f"  N = {n:,}, {label}:")
        print(f"    {'H':>3} {'graph':>11} {'one launch median':>18}   nvidia-smi while the graph replayed")
        meds = []
        for horizon in (1, 5, 15, 30):
            work[("d", n, horizon)] = th, om, plans = random_decision(rng, n, horizon)
            dbh = cuda_engine.DeviceBatch(th, om, plans)
            launch = lambda: eng.launch(dbh, kernel)  # noqa: E731
            g, smi = graph_ms(launch, clock=True)
            t = one_launch_ms(launch, reps=100)
            meds.append(med(g))
            print(f"    {horizon:>3} {us(med(g)):>11} {us(med(t)):>18}   {smi}")
        slope, intercept = np.polyfit([1, 5, 15, 30], meds, 1)
        print(f"    least-squares line through the 4 graph figures: {us(slope)} per imagined step "
              f"+ {us(intercept)}")
    print(f"\nnvidia-smi at the end of the GPU part: {gpu_state()}")
    print(f"GPU time inside the event pairs: {GPU_MS[0]:.0f} ms; wall time spent priming and "
          f"clock sampling (back-to-back launches outside the event pairs): {PRIME_TOTAL_S[0]:.2f} s")

    # ---- (e) ------------------------------------------------------------
    print(f"\n(e) NumPy golden.int_rollout on the CPU ({cpu_name()}), host wall time, the same inputs "
          "as above. This is the reference model (int64 arrays, a Python loop over the H steps), not a "
          "tuned CPU implementation.")
    th, om, plans = work["a"]
    t = wall_ms(lambda: golden.int_rollout(int(th[0]), int(om[0]), plans[0], p), reps=200, warmup=10)
    t2 = wall_ms(lambda: golden.argmin_first(golden.int_rollout(int(th[0]), int(om[0]), plans[0], p)[0]),
                 reps=200, warmup=10)
    print(f"  (a) N = 32, H = 15: median {us(med(t))} (min {us(min(t))}, 200 calls); "
          f"with golden.argmin_first, as golden.int_engine does: median {us(med(t2))}")
    per_call = []
    t0 = time.perf_counter()
    for d in range(4000):
        t1 = time.perf_counter()
        golden.int_rollout(int(v["theta0"][d]), int(v["omega0"][d]), v["plans"][d], p)
        per_call.append(1000 * (time.perf_counter() - t1))
    total = time.perf_counter() - t0
    print(f"  (b) the 4,000 decisions of sim/, one call after another: {total:.2f} s "
          f"= {4000 / total:,.0f} decisions/s (median {us(med(per_call))} per call)")
    print(f"  (c) H = 15:  {'N':>7} {'median':>12} {'reps':>5}")
    for n in sizes:
        th, om, plans = work[("c", n)]
        reps = 20 if n <= 4096 else 5
        t = wall_ms(lambda: golden.int_rollout(int(th[0]), int(om[0]), plans[0], p), reps=reps, warmup=1)
        print(f"               {n:>7,} {med(t):>9,.2f} ms {reps:>5}")
    for n in (32, 4096):
        print(f"  (d) N = {n:,}:  {'H':>3} {'median':>12} {'reps':>5}")
        for horizon in (1, 5, 15, 30):
            th, om, plans = work[("d", n, horizon)]
            t = wall_ms(lambda: golden.int_rollout(int(th[0]), int(om[0]), plans[0], p), reps=20, warmup=1)
            print(f"               {horizon:>3} {med(t):>9,.3f} ms {20:>5}")

    print(f"\nnvidia-smi at the end: {gpu_state()}")
    print(f"wall time {time.perf_counter() - t_start:.1f} s")
    print("H = 30 is beyond the RTL's limit of H = 16 (act_t is 4 bits); the kernel takes any H, and its "
          "int32 cost cannot overflow for any H up to 79,610 (2^31 / 26,975, the largest step cost).")


if __name__ == "__main__":
    main()
