"""
The host side of the CUDA rollout engine: loads the ROMs, compiles
cuda/rollout_kernel.cu with NVRTC (through CuPy), copies the weights into
the GPU's __constant__ memory and runs decisions.

    eng = CudaEngine()                          # rom/ and the kernel source
    r = eng.decide(theta_q, omega_q, plans)     # one decision
    r.best_idx, r.best_cost, r.cost, r.theta, r.omega
    b = eng.decide_batch(theta0, omega0, plans) # D decisions in one launch
    b.cost[d, n], b.theta[d, n], b.omega[d, n], b.best_idx[d], b.best_cost[d]

theta_q, omega_q: the int8 measured state (planner.quantize_state).
plans: (N, H) int8 actions, plans[n][t] = candidate n's action at step t,
the array model/planner.py builds; for a batch, (D, N, H). N = 32 runs the
RTL-mirror kernel (rollout_warp, one warp per decision); any other multiple
of 32 runs rollout_blocks.

eng.planner_engine() returns an engine for planner.run_episode(), so the GPU
can drive the pendulum exactly the way golden.int_engine and the RTL do.

The GPU never sees floats: the ROM files hold the integers the RTL uses,
and the only thing done on the host is packing them (4 int8 per int32 word,
the layout __dp4a reads) and copying them.
"""

import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))
import planner  # noqa: E402  (model/planner.py: quantize_state, the host side)

# CuPy keeps compiled kernels in a disk cache, by default ~/.cupy/kernel_cache.
# Keep it inside the repo instead (.cupy_cache/, in .gitignore). This must be
# set before cupy is imported.
os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT / ".cupy_cache"))
import cupy as cp  # noqa: E402

KERNEL_PATH = Path(__file__).resolve().with_name("rollout_kernel.cu")
NVRTC_OPTIONS = ("--std=c++20",)   # C++20 defines >> on a negative int as arithmetic
WARP = 32
MAX_THREADS = 256                  # threads per block for rollout_blocks (8 warps)
KEY_INIT = np.uint64(0xFFFF_FFFF_FFFF_FFFF)   # rollout_blocks: "no candidate yet"


# ------------------------------------------------------------------ ROMs --

def read_hex_signed(path, bits):
    """One two's-complement hex value per line ($readmemh format) -> ints.
    split() also drops the '\\r' of Windows line endings."""
    values = [int(tok, 16) for tok in Path(path).read_text().split()]
    return np.array([v - (1 << bits) if v >= 1 << (bits - 1) else v for v in values], np.int64)


def load_rom(rom_dir=ROOT / "rom"):
    """rom/*.hex and rom/params.vh -> the same dict layout golden.fold()
    returns (W1 (16, 3), B1 (16,), W2 (2, 16), B2 (2,), S1, S2), so the
    tests can hand it straight to golden.int_rollout."""
    rom_dir = Path(rom_dir)
    params = dict((name, int(value)) for name, value in re.findall(
        r"localparam\s+(\w+)\s*=\s*(-?\d+)", (rom_dir / "params.vh").read_text()))
    bias_w = params["BIAS_W"]
    rom = {
        "W1": read_hex_signed(rom_dir / "w1.hex", 8).reshape(16, 3),   # address 3j + i
        "B1": read_hex_signed(rom_dir / "b1.hex", bias_w),
        "W2": read_hex_signed(rom_dir / "w2.hex", 8).reshape(2, 16),   # address 16k + j
        "B2": read_hex_signed(rom_dir / "b2.hex", bias_w),
        "S1": params["S1"],
        "S2": (params["S2_0"], params["S2_1"]),
    }
    assert rom["W1"].size == 48 and rom["B1"].size == 16
    assert rom["W2"].size == 32 and rom["B2"].size == 2
    return rom


def pack_int8x4(values):
    """(..., 4k) ints in -128..127 -> (..., k) int32 words, four bytes each,
    byte 0 = the first value. NumPy and the GPU are both little-endian, so
    this only reinterprets memory."""
    v = np.asarray(values, np.int64)
    assert v.shape[-1] % 4 == 0 and v.min() >= -128 and v.max() <= 127
    return np.ascontiguousarray(v.astype(np.int8)).view(np.int32)


def choose_threads(n_cand, max_threads=MAX_THREADS):
    """Threads per block for rollout_blocks: a multiple of 32 that divides N,
    as large as possible up to max_threads."""
    return WARP * math.gcd(n_cand // WARP, max_threads // WARP)


# ------------------------------------------------------------- results --

@dataclass
class Batch:
    """Results of D decisions with N candidates each (NumPy int64)."""
    cost: np.ndarray        # (D, N) total cost of each candidate
    theta: np.ndarray       # (D, N) theta after H steps
    omega: np.ndarray       # (D, N) omega after H steps
    best_idx: np.ndarray    # (D,)
    best_cost: np.ndarray   # (D,)


@dataclass
class Decision:
    """Results of one decision."""
    cost: np.ndarray        # (N,)
    theta: np.ndarray       # (N,)
    omega: np.ndarray       # (N,)
    best_idx: int
    best_cost: int


class DeviceBatch:
    """Inputs and output buffers of one launch, already on the GPU. Kept as an
    object so bench.py can launch the same data many times without copying."""

    def __init__(self, theta0, omega0, plans):
        plans = np.asarray(plans)
        assert plans.ndim == 3, "plans must be (D, N, H)"
        self.D, self.N, self.H = plans.shape
        assert self.N % WARP == 0, "N must be a multiple of 32"
        assert plans.min() >= -128 and plans.max() <= 127, "actions are int8"
        theta0 = np.asarray(theta0, np.int64).reshape(self.D)
        omega0 = np.asarray(omega0, np.int64).reshape(self.D)
        assert theta0.min() >= -128 and theta0.max() <= 127
        assert omega0.min() >= -128 and omega0.max() <= 127

        # (D, N, H) -> (D, H, N): step-major, candidate-minor, see the kernel
        act = np.ascontiguousarray(plans.transpose(0, 2, 1).astype(np.int8))
        state = np.ascontiguousarray(np.stack([theta0, omega0], axis=1).astype(np.int8))
        self.act = cp.asarray(act)          # host -> GPU copies
        self.state0 = cp.asarray(state)
        shape = (self.D, self.N)
        self.cost = cp.empty(shape, cp.int32)
        self.theta = cp.empty(shape, cp.int32)
        self.omega = cp.empty(shape, cp.int32)
        self.best_idx = cp.empty(self.D, cp.int32)    # rollout_warp
        self.best_cost = cp.empty(self.D, cp.int32)
        self.best_key = cp.empty(self.D, cp.uint64)   # rollout_blocks
        self.kernel = None                            # set by CudaEngine.launch

    def fetch(self):
        """GPU -> host copies (they wait for the kernel to finish)."""
        if self.kernel == "warp":
            best_idx = cp.asnumpy(self.best_idx).astype(np.int64)
            best_cost = cp.asnumpy(self.best_cost).astype(np.int64)
        else:
            key = cp.asnumpy(self.best_key).astype(np.uint64)
            best_idx = (key & np.uint64(0xFFFFFFFF)).astype(np.int64)
            best_cost = (key >> np.uint64(32)).astype(np.int64)
        return Batch(cost=cp.asnumpy(self.cost).astype(np.int64),
                     theta=cp.asnumpy(self.theta).astype(np.int64),
                     omega=cp.asnumpy(self.omega).astype(np.int64),
                     best_idx=best_idx, best_cost=best_cost)


# -------------------------------------------------------------- engine --

class CudaEngine:
    """Compiles the kernels and holds the weights in constant memory.

    source: kernel source text (default: cuda/rollout_kernel.cu). The tests
    pass edited copies to check that planted bugs are caught.
    """

    def __init__(self, rom_dir=ROOT / "rom", source=None):
        self.rom = load_rom(rom_dir)
        if source is None:
            source = KERNEL_PATH.read_text(encoding="utf-8")
        self.source = source
        # NVRTC compiles the source for this GPU (sm_86 machine code on an RTX
        # 3060); CuPy caches the result on disk, keyed by source and options.
        self.module = cp.RawModule(code=source, options=NVRTC_OPTIONS)
        self.k_warp = self.module.get_function("rollout_warp")
        self.k_blocks = self.module.get_function("rollout_blocks")
        self._upload_weights()

    def _set_constant(self, name, values):
        """Copy values into the __constant__ array `name` (cudaMemcpy to the
        symbol's address, which get_global returns)."""
        host = np.ascontiguousarray(values, np.int32)
        dev = cp.ndarray(host.shape, cp.int32, self.module.get_global(name))
        dev.set(host)

    def _upload_weights(self):
        r = self.rom
        # W1 rows (16, 3) + a zero 4th input -> 16 words
        w1_padded = np.concatenate([r["W1"], np.zeros((16, 1), np.int64)], axis=1)
        self.w1_packed = pack_int8x4(w1_padded).reshape(16)
        self.w2_packed = pack_int8x4(r["W2"]).reshape(2, 4)     # 16 values -> 4 words
        self._set_constant("c_W1", self.w1_packed)
        self._set_constant("c_B1", r["B1"])
        self._set_constant("c_W2", self.w2_packed)
        self._set_constant("c_B2", r["B2"])
        self._set_constant("c_S1", [r["S1"]])
        self._set_constant("c_S2", list(r["S2"]))

    # -- launches ---------------------------------------------------------

    def launch(self, db, kernel="auto", threads=None):
        """Start the kernel on db's data. Returns immediately: the GPU runs
        it while the host goes on (call db.fetch() to wait and copy back).

        kernel: "warp" (needs N = 32), "blocks", or "auto" (warp if N = 32).
        threads: block size for "blocks" (default: choose_threads(N)).
        """
        if kernel == "auto":
            kernel = "warp" if db.N == WARP else "blocks"
        db.kernel = kernel
        h = np.int32(db.H)
        if kernel == "warp":
            assert db.N == WARP, "rollout_warp is one warp = 32 candidates"
            self.k_warp((db.D,), (WARP,),
                        (db.act, db.state0, h, db.cost, db.theta, db.omega,
                         db.best_idx, db.best_cost))
        elif kernel == "blocks":
            threads = threads or choose_threads(db.N)
            assert threads % WARP == 0 and 0 < threads <= 1024 and db.N % threads == 0
            assert db.D <= 65535, "gridDim.y limit"
            # every key starts at 2^64 - 1, above any real key. fill() runs a
            # tiny CuPy kernel on the same stream, so it is done before ours
            # starts. (A cudaMemsetAsync of 0xff bytes would do the same job;
            # results/cuda/bench.log compares the two.)
            db.best_key.fill(KEY_INIT)
            self.k_blocks((db.N // threads, db.D), (threads,),
                          (db.act, db.state0, h, np.int32(db.N),
                           db.cost, db.theta, db.omega, db.best_key))
        else:
            raise ValueError(f"unknown kernel {kernel!r}")
        return db

    def decide_batch(self, theta0, omega0, plans, kernel="auto", threads=None):
        """D decisions in one launch. theta0, omega0: (D,) int8 states;
        plans: (D, N, H) int8 actions. Returns a Batch."""
        db = DeviceBatch(theta0, omega0, plans)
        self.launch(db, kernel, threads)
        return db.fetch()

    def decide(self, theta_q, omega_q, plans, kernel="auto", threads=None):
        """One decision: plans is (N, H). Returns a Decision."""
        plans = np.asarray(plans)
        b = self.decide_batch([theta_q], [omega_q], plans[None], kernel, threads)
        return Decision(cost=b.cost[0], theta=b.theta[0], omega=b.omega[0],
                        best_idx=int(b.best_idx[0]), best_cost=int(b.best_cost[0]))

    def planner_engine(self, record=None):
        """An engine for planner.run_episode(): measured float state in,
        best plan index out, with the GPU doing the imagining. If record is
        a list, every decision's inputs and outputs are appended to it."""
        def decide(theta, omega, plans):
            theta_q, omega_q = planner.quantize_state(theta, omega)
            r = self.decide(theta_q, omega_q, plans)
            if record is not None:
                record.append({"theta0": theta_q, "omega0": omega_q, "plans": plans.copy(),
                               "best": r.best_idx, "best_cost": r.best_cost})
            return r.best_idx
        return decide


PTX_PATTERNS = [
    ("dp4a.s32.s32", "__dp4a, signed x signed"),
    ("shr.s32", "arithmetic right shifts"),
    ("shr.u32", "logical right shifts"),
    ("shfl.sync.down", "warp shuffles"),
    ("ld.const", "constant-memory loads"),
    ("bar.sync", "__syncthreads barriers"),
    ("atom.global.min.u64", "64-bit atomicMin"),
]


def ptx_text(source=None):
    """Compile the kernel source to PTX, the GPU's virtual assembly language
    (the step before machine code), with the same options as CudaEngine.
    PTX is not the final machine code (SASS): the driver's assembler still
    translates it, and with no CUDA Toolkit here there is no disassembler."""
    nv = cp.cuda.nvrtc
    if source is None:
        source = KERNEL_PATH.read_text(encoding="utf-8")
    cc = cp.cuda.Device(0).compute_capability            # "86" on an RTX 3060
    prog = nv.createProgram(source, "rollout_kernel.cu", [], [])
    try:
        nv.compileProgram(prog, list(NVRTC_OPTIONS) + [f"-arch=compute_{cc}"])
        ptx = nv.getPTX(prog)
    finally:
        nv.destroyProgram(prog)
    return ptx.decode().rstrip("\0") if isinstance(ptx, bytes) else ptx


def ptx_summary(source=None):
    """Count per kernel the PTX instructions this project is about: a check
    that __dp4a really became dp4a and that the requantization shifts are
    arithmetic (shr.s32), not logical (shr.u32), and the number of PTX
    instructions in the loop over the H imagined steps.
    Returns {kernel name: {pattern: count}}."""
    ptx = ptx_text(source)
    out = {}
    for body in ptx.split(".entry ")[1:]:
        name = body.split("(", 1)[0].strip()
        out[name] = {pat: body.count(pat) for pat, _ in PTX_PATTERNS}
        out[name]["step loop"] = step_loop_length(body)
    return out


def step_loop_length(body):
    """PTX instructions between the label after '.pragma "nounroll"' (the
    step loop of '#pragma unroll 1') and the loop's branch back to it."""
    lines = [ln.strip() for ln in body.splitlines()]
    try:
        start = lines.index('.pragma "nounroll";')
    except ValueError:
        return None
    label = lines[start - 1].rstrip(":")
    count = 0
    for ln in lines[start + 1:]:
        if ln and not ln.startswith("//"):
            count += 1
        if ln.startswith("@") and ln.endswith(label + ";"):
            return count
    return None


def versions():
    """One-line descriptions of the software and the GPU, for log headers."""
    dev = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    rt = cp.cuda.runtime.runtimeGetVersion()
    drv = cp.cuda.runtime.driverGetVersion()
    nv_major, nv_minor = cp.cuda.nvrtc.getVersion()
    return [
        f"Python {sys.version.split()[0]}, NumPy {np.__version__}, CuPy {cp.__version__}",
        f"CUDA runtime (CuPy's) {rt // 1000}.{rt % 1000 // 10}, driver supports CUDA "
        f"{drv // 1000}.{drv % 1000 // 10}, NVRTC {nv_major}.{nv_minor}, options {' '.join(NVRTC_OPTIONS)}",
        f"GPU: {name}, compute capability {dev.compute_capability}, "
        f"{props['multiProcessorCount']} SMs, warp size {props['warpSize']}",
    ]
