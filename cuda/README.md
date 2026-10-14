# CUDA port

The rollout engine of [`rtl/`](../rtl/) as CUDA kernels: the same integer math ([DESIGN.md §4](../DESIGN.md)), bit-exact with the golden model, run on an NVIDIA GeForce RTX 3060 (compute capability 8.6, 28 SMs) under Windows 11. The kernels are CUDA C++ compiled at run time by NVRTC through [CuPy](https://cupy.dev/), so the PC needs no CUDA Toolkit and no C++ compiler, only the NVIDIA driver and a Python virtualenv.

Every number below comes from a log in [`results/cuda/`](../results/cuda/), and each log starts with the command that made it. `smoke.log`, `bitexact.log` and `closed_loop.log` were made on the PC with the files of commit 8565c4a, `bench.log` with those of commit 18e97ff (the kernels' PTX is byte-identical at both); the files were copied over with scp (SHA-256 identical) and the logs copied back. Only the logs' line endings were changed, plus one edit in `bench.log` that its header describes.

**The timing rows are provisional.** `bench.log` was made with `--shared-gpu`, which overrides `bench.py`'s own refusal to run while another Python process holds a GPU context. One did (a long-running service of another program), and the rule for this PC is to skip benchmarks in that case; the override was used without the OK that rule asks for. The rows stay provisional until `bench.py` has run without `--shared-gpu`, with that process off the GPU, or the run is accepted as it is ("Not alone on the GPU", below).

## Results

| Result | Log | Command (repo root, on the PC) |
|---|---|---|
| **Smoke test**: `__dp4a` and the warp-shuffle sum agree with NumPy on 1,024 dot products of 512 int8; both planted kernel bugs caught | [`smoke.log`](../results/cuda/smoke.log) | `.venv\Scripts\python.exe cuda\smoke_test.py --mutants` |
| **Bit-exact replay**: three sets of **4,000 / 4,000** recorded decisions, each in one launch per kernel: the Mac's H = 15 set (`sim/` as golden.py writes it on the Mac, the set [`test_full.log`](../results/test_full.log) replays; its `returns.txt` equals the 20 returns in [`cosim.log`](../results/cosim.log)), the PC's own H = 15 set and its H = 5 set. Every cost, every final θ and ω, and best_idx / best_cost equal the expected values in `exp_*.hex`, for both kernels | [`bitexact.log`](../results/cuda/bitexact.log) | `.venv\Scripts\python.exe cuda\test_bitexact.py --vec sim --vec sim/mac --vec sim/h5 --save-ptx results/cuda/rollout_kernel.ptx` |
| **Random decisions, up to 4,096 candidates**, states and actions over the whole int8 range, H = 1, 15 and 30, two sets where every plan appears 2 or 8 times (so the lowest cost is always tied), against `golden.int_rollout` + `golden.argmin_first`: all equal, 150,528 candidate results (75,264 candidates, each in two runs: two block sizes of `rollout_blocks`, or both kernels at N = 32). Over all 16 set/kernel runs: **918,528** candidate results compared (459,264 distinct candidates, each checked in two runs), **0** wrong | [`bitexact.log`](../results/cuda/bitexact.log) | (same run) |
| **Planted bugs**: **11 / 11** caught: logical shift, θ saturates, ω wraps, h clamped at 255, tie goes to the higher index, tree argmin without the index comparison, and five in code only `rollout_blocks` has (level 2 reads only warp 0's winner, cost 0 for lanes with no warp winner, the key with the index in the high half, the key without the index, best_key starting at 0), each of those five required to FAIL on the N = 4,096 set with 256-thread blocks | [`bitexact.log`](../results/cuda/bitexact.log) | (same run) |
| **Closed loop, the GPU in charge**: the kernel picks every action of 20 Pendulum-v1 episodes. **20 / 20** returns identical (float `==`) to the golden integer closed loop on the same PC, **4,000 / 4,000** choices the same, every GPU answer equal to golden's on the same inputs | [`closed_loop.log`](../results/cuda/closed_loop.log) | `.venv\Scripts\python.exe cuda\closed_loop.py` |
| **One decision, N = 32, H = 15** (provisional): **4.25 µs** of GPU time per launch (100 launches replayed from a CUDA graph); 17.28 µs for a single ordinary launch on the GPU timeline; **62.95 µs** host round trip (actions in, best index out); 290.45 µs through `eng.decide()`. NumPy golden model on the same PC's CPU: 672.85 µs | [`bench.log`](../results/cuda/bench.log) | `.venv\Scripts\python.exe cuda\bench.py --shared-gpu` (the logged command; without the flag it refuses while another Python process holds the GPU) |
| **4,000 decisions in one launch** (provisional): 0.0671 ms per launch (graph), **59.6 million decisions/s** | [`bench.log`](../results/cuda/bench.log) | (same run) |
| **Candidates vs time** (provisional; one decision, H = 15, `rollout_blocks`, including the best_key fill kernel): **5.21 µs** at N = 32, **5.87 µs** at N = 4,096, 10.31 µs at 16,384, 26.98 µs at 65,536 | [`bench.log`](../results/cuda/bench.log) | (same run) |
| **Imagined steps vs time** (provisional; one decision): 1.47, 2.18, 4.25, 7.22 µs for H = 1, 5, 15, 30 at N = 32 (`rollout_warp`); least-squares slope **0.20 µs per step** (0.23 µs at N = 4,096, `rollout_blocks` with the fill kernel) | [`bench.log`](../results/cuda/bench.log) | (same run) |

Software: Python 3.14.3, NumPy 2.5.4, gymnasium 1.4.0, CuPy 14.2.0 (`cupy-cuda13x`), NVRTC 13.3, CuPy's CUDA runtime 13.2, a driver that supports CUDA 13.3, WDDM driver model. CPU: AMD Ryzen 7 3700X.

## What the benchmark shows, and what it doesn't

**Candidates are almost free until the GPU fills; imagined steps never are.** Going from 32 to 4,096 candidates (128 times as many) takes one decision from 5.21 to 5.87 µs (`rollout_blocks`, including the best_key fill kernel; `rollout_warp` alone takes 4.25 µs at N = 32). Only at 16,384 does the time clearly rise (10.31 µs), and from there it grows with N (26.98 µs at 65,536). Going from H = 1 to H = 30 at N = 32 takes it from 1.47 to 7.22 µs, about 0.20 µs per imagined step, because step t + 1 needs step t's state and no number of threads can start it early. It's the RTL's lesson ("more lanes buy more candidates, not a shorter decision") measured on a GPU. Why the curve bends where it does is question 7 of [STUDY.md](../STUDY.md#5-cuda-port); any answer to it is reasoning from these numbers, not a profiler measurement.

**At this size, the kernel is the small part.** One 32-candidate decision is 4.25 µs of GPU work, but a single launch, waited for, spans 17.28 µs on the GPU's own timeline, and a round trip from Python (copy 480 action bytes in, launch, copy the best index out) takes 62.95 µs. `eng.decide()`, the convenient path the closed loop uses, makes two copies in and five copies out (seven transfers) and allocates fresh buffers on every call: 290.45 µs.

**What these numbers don't show:**
- **Windows, WDDM.** The launch and copy overheads are what the Windows display driver model gives on this PC. A Linux machine, or a GPU in TCC mode, was not measured, and its overheads could differ either way.
- **One GPU.** An RTX 3060 (consumer Ampere, 28 SMs). Nothing here says how a data-center GPU or a Jetson would do.
- **No profiler.** No Nsight Systems or Nsight Compute; GPU time comes from CUDA events only. The only view of the compiled code is the PTX ([`rollout_kernel.ptx`](../results/cuda/rollout_kernel.ptx)). The machine code that runs (SASS) is the sm_86 cubin that NVRTC makes for CuPy, cached in `.cupy_cache/`; no disassembler was installed to read it. Where this README explains *why* a number is what it is, that is an inference.
- **NVRTC, not nvcc.** The kernels were compiled at run time by NVRTC with `--std=c++20`; an offline nvcc build was never made.
- **Clocks not locked** (that needs admin rights). nvidia-smi, asked while each timed CUDA graph replayed, read 1,957 to 1,972 MHz (P2) for the CUDA-graph series of sections (a), (c) and (d); it was not sampled for section (b).
- **GPU times and RTL times are not comparable.** The GPU numbers here are measurements on an RTX 3060; the RTL's 27.0 / 97.4 µs per decision in the main README are post-route, out-of-context estimates for an ECP5 FPGA. Neither is a speedup claim over the other.
- **Not alone on the GPU (why the timing is provisional).** Another program's Python process (a long-running service) held a GPU context while `bench.py` ran. `bench.py` refuses to run in that case, on purpose; the run overrode that with `--shared-gpu`, without the OK the rule for this PC asks for. GPU utilization read 1 % when the run started, but the script checked only then, so that process's work could have overlapped later parts of the run. The SM clock read 1,957 to 1,972 MHz in every series where it was sampled. `bench.log` records the process and the flag. `bench.py` now also looks at the GPU's Python processes again before each section and stops if a new one has appeared.
- **The NumPy figures are the golden model** (int64 arrays, a Python loop over the H steps): a correctness reference, not a tuned CPU implementation. They are not a CPU-vs-GPU comparison.
- **A 98-parameter toy.** Nothing here says anything about real robot models. H = 30 is beyond the RTL's limit of H = 16; the kernel takes any H, and its int32 cost cannot overflow for any H up to 79,607 (at most 26,976 per step, with u = −128, which the kernel accepts; the footer of `bench.log` gives 79,610, which assumes |u| ≤ 127).

## RTL to CUDA

| RTL ([`rtl/rollout_engine.v`](../rtl/rollout_engine.v)) | CUDA ([`rollout_kernel.cu`](rollout_kernel.cu)) |
|---|---|
| one lane = one candidate: its own θ, ω, h[0..15] and cost | one thread = one candidate, the same values in registers |
| 32 lanes in lockstep | one warp: 32 threads the GPU issues together, one instruction for all 32 (since Volta, a warp's threads can also diverge and be scheduled independently, which is why the shuffles are `*_sync` calls with a full-warp mask) |
| one decision | one thread block (`rollout_warp`: a block is exactly one warp; grid = number of decisions, so all 4,000 recorded decisions go in one launch) |
| weight ROM; the sequencer broadcasts one weight per clock to all lanes | `__constant__` memory; all 32 threads of a warp read the same weight in the same instruction, and the constant cache serves all 32 with one read |
| one signed 8×8 multiplier per lane, one MAC per clock | `__dp4a`: four signed 8×8 products and their sum in one instruction (what [dp4a-unit](https://github.com/Jusman0/dp4a-unit) built as hardware) |
| `>>>` on a signed accumulator | `>>` on an `int` (arithmetic since C++20; PTX `shr.s32`) |
| 8-bit θ register: overflow wraps for free | `((v + 128) & 255) - 128`, written out because an `int` has 32 bits |
| argmin: scan n = 0..31 with strict `<`, 32 clocks | 5 rounds of `__shfl_down_sync` on (cost, index) pairs; the index is compared too, because a tree does not compare in index order |
| LANES < 32: several passes over the 32 candidates | N > 32: more warps and blocks (`rollout_blocks`); argmin across warps in shared memory, across blocks with one 64-bit `atomicMin` on (cost << 32 \| index) |
| ACT memory: one 256-bit word per step | `act[(d·H + t)·N + n]`: the 32 threads of a warp read 32 neighbouring bytes, one memory transaction |
| result buffer and `dbg_*` port | `cost_out`, `theta_out`, `omega_out` arrays, copied back to the host |
| `cycles` counter | CUDA events around launches, and around CUDA-graph replays |

What maps less neatly: the RTL re-reads every weight from ROM in every imagined step. In the PTX, the compiler loads the 45 constant words once, before the step loop; what the machine code does with them was not checked, since the cubin was not disassembled (see the comment in `rollout_kernel.cu`).

## Files

| File | What it is |
|---|---|
| [`rollout_kernel.cu`](rollout_kernel.cu) | the kernels: `imagine_step` (DESIGN.md §4 for one thread), `warp_argmin`, `rollout_warp` (the RTL mirror), `rollout_blocks` (N candidates) |
| [`cuda_engine.py`](cuda_engine.py) | host side: reads `rom/`, packs the weights 4 per int32, copies them to constant memory, `decide()`, `decide_batch()`, `planner_engine()`, PTX export |
| [`test_bitexact.py`](test_bitexact.py) | replays the recorded decisions and random sets; planted bugs |
| [`closed_loop.py`](closed_loop.py) | the GPU drives 20 pendulum episodes through `model/planner.py` |
| [`bench.py`](bench.py) | CUDA-event timing; refuses if the GPU is busy or another Python process holds it, and checks again before each section |
| [`smoke_test.py`](smoke_test.py) | `__dp4a` and warp shuffles on their own, before the real kernel |

## Run it

### Windows (what the logs come from)

Needs the NVIDIA driver (one that supports CUDA 13) and Python 3.14. No CUDA Toolkit, no Visual Studio. From the repo root, with `python` being a Python 3.14 interpreter:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt cupy-cuda13x==14.2.0 nvidia-cuda-nvrtc==13.3.33 nvidia-cuda-runtime==13.3.29
.venv\Scripts\python.exe model\golden.py                          # rom\ and sim\ (H = 15)
.venv\Scripts\python.exe model\golden.py --H 5 --out sim/h5       # sim\h5\ (also rewrites rom\, same contents)
.venv\Scripts\python.exe cuda\smoke_test.py --mutants
.venv\Scripts\python.exe cuda\test_bitexact.py                    # sim and sim/h5; --vec DIR (repeatable) replaces them
.venv\Scripts\python.exe cuda\closed_loop.py
.venv\Scripts\python.exe cuda\bench.py                            # refuses (exit 2) on purpose while another Python process holds the GPU; --shared-gpu only with its owner's OK
```

This installs only NVRTC (to compile the kernels) and the CUDA runtime headers next to CuPy, not the full `cupy-cuda13x[ctk]` set (cuBLAS, cuFFT and the other CUDA libraries, much larger), so CuPy features that need those libraries, such as `cupy.random`, won't work in this venv; the kernels don't use them. CuPy's compiled-kernel cache goes to `.cupy_cache/` in the repo (in `.gitignore`).

`sim/mac/` in the logs is a copy of the Mac's `sim/` vector files. The PC's own `golden.py` run made different vectors: its closed-loop episodes come out slightly different from the Mac's, so from some decision on its recorded states differ. The integer math is not the cause: on the PC, `golden.int_rollout` reproduces every expected value in the Mac's files (`bitexact.log`, "reproduces the file: yes"). Both sets pass on the GPU.

### Linux (not tried)

The same steps with `python3 -m venv .venv` and `.venv/bin/python`. CuPy's Linux wheels need a driver that supports the CUDA version of the wheel (`cupy-cuda12x` for CUDA 12 drivers). This port has only been run on the Windows PC above.

## Limits

- Checked bit-exact on one GPU, one driver and one CuPy/NVRTC version. The planted-bug runs show the checks can fail; they don't prove the kernels are right on other hardware.
- The closed loop matches the golden model **on the same PC**. Its returns equal the PC's `sim/returns.txt` on 20 of 20 episodes and the Mac's on 2 of 20 ([`closed_loop.log`](../results/cuda/closed_loop.log)): the episodes differ between the two machines before any engine is involved (golden.py's own returns differ). The cause was not pinned down; the integer decisions are the same wherever the inputs are (the Mac's 4,000 decisions pass bit-exact on the GPU).
- `eng.decide()` is written to be read, not to be fast: two copies in, five out and fresh buffers per call. Not tried: one packed result copy, pinned host memory, prefetching the next step's action inside the kernel, several decisions per block (with one-warp blocks, `rollout_warp` is held to 16 blocks, 512 threads, per SM), Linux or TCC.
- A planted bug that changes nothing on some sets still passes there: ω wraps passes on the H = 5 set, which never saturates ω; h clamped at 255 passes on the two sets that never clamp h; the five `rollout_blocks` bugs pass on the `rollout_warp` runs, which don't have that code, and two of them also where that code has nothing to decide ("reads only warp 0's winner" with one-warp blocks, the cost-0 sentinel with 32-warp blocks, which leave no lane empty). The tie bugs pass on the sets without ties, and also on the N = 4,096 set with 256-thread blocks, where (by a scratch-script check, not in a log) the block holding the lowest tied index held only one copy of the minimum, so the cross-block `atomicMin` settled every tie correctly. Every bug fails on at least one set, the five `rollout_blocks` ones on that N = 4,096 set; `bitexact.log` lists where each still passes.
- Two gaps. The key with the index in the high half fails partly because the host then reads the two halves of the key the wrong way round, not only because of its order. And no planted bug removes the `__syncthreads()` before level 2: that bug is a race, whose effect depends on timing, so it would pass or fail by chance.

*Built with an AI pair programmer: Claude wrote the kernels and the tests, to be studied line by line ([STUDY.md](../STUDY.md#5-cuda-port)).*
