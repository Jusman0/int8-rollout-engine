"""
CUDA smoke test: does a CuPy RawKernel compile and run, and do the two GPU
instructions the rollout kernel will lean on give exactly what NumPy gives?

  __dp4a            one instruction: 4 signed int8 x int8 products, summed,
                    plus a 32-bit accumulator
  __shfl_down_sync  a thread reads a register of another thread in its warp,
                    with no shared memory in between

The test: 1,024 int8 dot products of length 512, one per warp. Each of the 32
threads of a warp does 4 __dp4a (16 of the 512 products), then the warp adds
its 32 partial sums with 5 shuffles. Both the per-thread partial sums and the
warp totals are compared with NumPy int64 math, so a wrong __dp4a and a wrong
shuffle reduction show up as different failures.

Run on the machine with the GPU, from the repo root:
  .venv\\Scripts\\python.exe cuda\\smoke_test.py         (Windows)
  python3 cuda/smoke_test.py                             (anywhere else)
  add --mutants to also check that two planted kernel bugs are caught

CuPy compiles the CUDA C++ source below at run time with NVRTC (NVIDIA's
runtime compiler library, here from the nvidia-cuda-nvrtc pip wheel), so no
CUDA Toolkit or C++ compiler is needed. Prints PASS and exits 0, or prints
FAIL and exits 1.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# CuPy keeps compiled kernels in a disk cache, by default ~/.cupy/kernel_cache,
# a folder it creates even before it compiles anything. Keep it inside the repo
# instead: .cupy_cache/ (in .gitignore). This must be set before the import.
os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT / ".cupy_cache"))

import numpy as np   # noqa: E402
import cupy as cp    # noqa: E402

# ------------------------------------------------------------- the kernel --
#
# Words used below, smallest first:
#   thread  one instance of the kernel function. Every thread runs the same
#           code on its own registers; threadIdx and blockIdx tell it who it is.
#   warp    32 threads with consecutive threadIdx.x, which the GPU issues
#           together: one instruction, 32 threads, like the RTL's one weight
#           broadcast feeding 32 lanes. Threads of one warp can read each
#           other's registers with the __shfl_*_sync instructions.
#   block   a group of warps (here 4 warps = 128 threads) that runs on one
#           streaming multiprocessor (SM) and can share fast on-chip memory.
#           The RTX 3060 has 28 SMs.
#   grid    all the blocks of one launch (here 256 blocks).
#
# Memory layout, chosen so the loads are "coalesced": in word w of dot product
# d, thread t reads a4[(d * WORDS + w) * 32 + t]. The 32 threads of a warp
# read 32 neighbouring 4-byte words, 128 contiguous bytes, which the memory
# system serves as one transaction instead of 32 scattered ones.

KERNEL_SOURCE = r"""
// The host fills in WORDS (int32 words per thread) with a #define, so the loop
// below has a compile-time trip count and NVRTC can unroll it.

extern "C" __global__
void dot_int8_dp4a(const int* __restrict__ a4,      // packed int8x4 words
                   const int* __restrict__ b4,      // same layout as a4
                   int*       __restrict__ partial, // one per thread
                   int*       __restrict__ dot,     // one per warp
                   int n_dots)
{
    // Who am I? A global thread number, my warp, and my lane in that warp.
    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = tid / 32;          // one warp per dot product
    const int lane = threadIdx.x % 32;  // 0..31: which 4-byte column I own
                                        // (blockDim.x is a multiple of 32)
    if (warp >= n_dots) return;         // the whole warp leaves together,
                                        // so the shuffles below stay full

    // ---- step 1: each thread, 4 x __dp4a --------------------------------
    // __dp4a(x, y, c) treats each 32-bit int as four signed bytes:
    //   c + x.byte0*y.byte0 + x.byte1*y.byte1 + x.byte2*y.byte2 + x.byte3*y.byte3
    // products and sum in 32 bits, so it cannot overflow here:
    // |sum| <= 512 * 128 * 128 = 8,388,608 < 2^31.
    // Passing the running sum as c makes the instruction do the accumulate too.
    int acc = 0;
    #pragma unroll
    for (int w = 0; w < WORDS; ++w) {
        const int idx = (warp * WORDS + w) * 32 + lane;  // coalesced, see above
        acc = __dp4a(a4[idx], b4[idx], acc);
    }
    partial[tid] = acc;   // kept so the host can check __dp4a on its own

    // ---- step 2: add the warp's 32 partial sums with shuffles -----------
    // __shfl_down_sync(mask, v, offset): every thread gets the v of the
    // thread `offset` lanes above it (lane + offset). A lane whose source
    // would be past lane 31 gets its own v back; those results are never
    // used. mask = 0xffffffff says all 32 lanes take part and must reach this
    // instruction together; the _sync suffix means the hardware waits for them.
    //   offset 16: lane i holds v[i] + v[i+16]       (lanes 0..15 useful)
    //   offset  8: lanes 0..7 hold sums of 4
    //   offset  4, 2, 1: lane 0 ends with the sum of all 32
    // 5 steps (log2 32) instead of 31 sequential adds, and no shared memory.
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        dot[warp] = acc;
    }
}
"""

WARP = 32              # threads per warp on every NVIDIA GPU so far
WORDS = 4              # int32 words per thread, 4 int8 each
N_DOTS = 1024          # dot products, one warp each
LENGTH = WARP * WORDS * 4          # 512 int8 values per dot product
THREADS_PER_BLOCK = 128            # 4 warps per block


def make_inputs(rng):
    """int8 data shaped (N_DOTS, WORDS, WARP, 4): [d, w, t, byte] is byte
    `byte` of the word thread t reads in step w of dot product d."""
    shape = (N_DOTS, WORDS, WARP, 4)
    a = rng.integers(-128, 128, size=shape, dtype=np.int8)
    b = rng.integers(-128, 128, size=shape, dtype=np.int8)
    # the extremes that a sign or width bug would get wrong
    a[0], b[0] = -128, -128    # largest positive dot: 512 * 16384
    a[1], b[1] = -128, 127     # most negative dot: 512 * -16256
    a[2], b[2] = 127, 127
    a[3], b[3] = -1, 1         # every byte 0xff: signed -1, not 255
    a[4], b[4] = 0, -128
    return a, b


def pack_words(x8):
    """(N_DOTS, WORDS, WARP, 4) int8 -> (N_DOTS, WORDS, WARP) int32: the four
    bytes of the last axis become one 32-bit word, byte 0 lowest (x86 and
    NVIDIA GPUs are both little-endian). Only the memory is reinterpreted."""
    return np.ascontiguousarray(x8).view(np.int32).reshape(x8.shape[:-1])


# Two planted bugs for --mutants: a test that cannot fail proves nothing.
# Each is (what it breaks, text in KERNEL_SOURCE, replacement).
MUTANTS = [
    ("unsigned __dp4a: bytes read as 0..255, so 0xff is 255, not -1",
     "acc = __dp4a(a4[idx], b4[idx], acc);",
     "acc = __dp4a((unsigned)a4[idx], (unsigned)b4[idx], (unsigned)acc);"),
    ("reduction starts at offset 8: lane 0 adds only lanes 0..15",
     "int offset = 16;", "int offset = 8;"),
]


def print_versions():
    dev = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    rt = cp.cuda.runtime.runtimeGetVersion()
    drv = cp.cuda.runtime.driverGetVersion()
    nvrtc_major, nvrtc_minor = cp.cuda.nvrtc.getVersion()
    print(f"CuPy {cp.__version__}, NumPy {np.__version__}")
    print(f"CUDA runtime (CuPy's) {rt // 1000}.{rt % 1000 // 10}, "
          f"driver supports CUDA {drv // 1000}.{drv % 1000 // 10}, "
          f"NVRTC {nvrtc_major}.{nvrtc_minor}")
    try:  # where the NVRTC DLL / .so was found (cuda-pathfinder, a CuPy dependency)
        from cuda.pathfinder import load_nvidia_dynamic_lib
        nvrtc_lib = load_nvidia_dynamic_lib("nvrtc")
        print(f"NVRTC loaded from {nvrtc_lib.abs_path} (found via {nvrtc_lib.found_via})")
    except Exception as e:  # informational only; the test does not depend on it
        print(f"NVRTC location: unknown ({type(e).__name__})")
    print(f"device 0: {name}, compute capability {dev.compute_capability}, "
          f"{props['multiProcessorCount']} SMs, warp size {props['warpSize']}")


def run_and_check(source, quiet=False):
    """Compile `source`, run it on the test data, compare with NumPy.
    Returns True if every partial sum and every warp total matches."""
    say = (lambda *a: None) if quiet else print
    rng = np.random.default_rng(0)
    a8, b8 = make_inputs(rng)
    a4, b4 = pack_words(a8), pack_words(b8)

    # NumPy reference in int64, so it cannot overflow
    prod = a8.astype(np.int64) * b8.astype(np.int64)        # (D, W, T, 4)
    want_partial = prod.sum(axis=(1, 3))                     # (D, T): per thread
    want_dot = want_partial.sum(axis=1)                      # (D,): per warp

    # compile (NVRTC, cached on disk by CuPy after the first run) and launch
    kernel = cp.RawKernel(f"#define WORDS {WORDS}\n" + source, "dot_int8_dp4a")
    a_gpu, b_gpu = cp.asarray(a4.ravel()), cp.asarray(b4.ravel())
    partial_gpu = cp.zeros(N_DOTS * WARP, dtype=cp.int32)
    dot_gpu = cp.zeros(N_DOTS, dtype=cp.int32)
    blocks = (N_DOTS * WARP + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    kernel((blocks,), (THREADS_PER_BLOCK,),
           (a_gpu, b_gpu, partial_gpu, dot_gpu, np.int32(N_DOTS)))
    cp.cuda.Device().synchronize()   # launches are asynchronous: wait for it
    say(f"kernel: {blocks} blocks x {THREADS_PER_BLOCK} threads, "
        f"{N_DOTS} dot products of {LENGTH} int8")

    got_partial = cp.asnumpy(partial_gpu).reshape(N_DOTS, WARP).astype(np.int64)
    got_dot = cp.asnumpy(dot_gpu).astype(np.int64)

    ok = True
    bad_p = np.argwhere(got_partial != want_partial)
    if len(bad_p):
        d, t = bad_p[0]
        say(f"FAIL __dp4a: {len(bad_p)} thread partial sums differ; first at dot {d}, "
            f"thread {t}: got {got_partial[d, t]}, want {want_partial[d, t]}")
        ok = False
    bad_d = np.flatnonzero(got_dot != want_dot)
    if len(bad_d):
        d = bad_d[0]
        say(f"FAIL shuffle reduction: {len(bad_d)} warp totals differ; first at dot {d}: "
            f"got {got_dot[d]}, want {want_dot[d]}")
        ok = False
    # the planted extremes, spelled out so a reader can check them by hand
    for d, want in [(0, 512 * 16384), (1, -512 * 16256), (2, 512 * 16129), (3, -512), (4, 0)]:
        if want_dot[d] != want or got_dot[d] != want:
            say(f"FAIL extreme case dot {d}: got {got_dot[d]}, NumPy {want_dot[d]}, by hand {want}")
            ok = False
    say(f"checked {N_DOTS * WARP:,} __dp4a partial sums and {N_DOTS:,} warp totals "
        f"(range {want_dot.min():,} .. {want_dot.max():,})")
    return ok


def main():
    mutants = "--mutants" in sys.argv[1:]
    print_versions()
    ok = run_and_check(KERNEL_SOURCE)
    if mutants:
        # every planted bug must make the check fail
        for what, old, new in MUTANTS:
            assert KERNEL_SOURCE.count(old) == 1, f"mutant text not found once: {old}"
            caught = not run_and_check(KERNEL_SOURCE.replace(old, new), quiet=True)
            print(f"planted bug {'caught' if caught else 'MISSED'}: {what}")
            ok = ok and caught
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
