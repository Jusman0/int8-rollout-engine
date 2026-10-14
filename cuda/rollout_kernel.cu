// rollout_kernel.cu: the int8 rollout engine of rtl/ as CUDA kernels.
//
// What it computes is DESIGN.md section 4, bit for bit: for every candidate
// plan, H imagined steps of the 3 -> 16 -> 2 integer network, the integer
// cost, and the argmin over the candidates (lowest index on a tie). The
// expected values are the ones model/golden.py writes; cuda/test_bitexact.py
// compares every cost, every final state and every best index.
//
// Compiled at run time by NVRTC (NVIDIA's runtime compiler library) through
// cupy.RawModule in cuda/cuda_engine.py, with --std=c++20. No CUDA Toolkit,
// no nvcc. Source kept to plain ASCII on purpose: it is read on Windows too.
//
// How the hardware ideas of the RTL map onto a GPU:
//
//   RTL (rtl/rollout_engine.v)            CUDA (this file)
//   ------------------------------------  ------------------------------------
//   one lane = one candidate              one thread = one candidate
//   32 lanes in lockstep                  one warp = 32 threads that the GPU
//                                         issues together, one instruction
//                                         for all 32 at a time
//   one decision                          one thread block (kernel 1: a block
//                                         is exactly one warp)
//   weight ROM, one weight broadcast      __constant__ memory: every thread of
//   to all 32 lanes per clock             a warp reads the same weight in the
//                                         same instruction, and the constant
//                                         cache serves all 32 with one read
//   one signed 8x8 multiplier per lane,   __dp4a: four signed 8x8 products
//   one MAC per clock                     and their sum in one instruction
//   argmin: scan n = 0..31, strict <,     warp shuffle: 5 rounds of
//   32 clocks                             __shfl_down_sync, each thread
//                                         compares with a thread 16, 8, 4,
//                                         2, 1 lanes away
//
// Two kernels:
//   rollout_warp    the RTL mirror: one block of 32 threads per decision.
//                   All 4,000 recorded decisions go in one launch (grid =
//                   4,000 blocks), and the GPU runs as many at once as its
//                   streaming multiprocessors (SMs) can hold.
//   rollout_blocks  N candidates per decision (N a multiple of 32), spread
//                   over several blocks of up to 1,024 threads. The argmin
//                   then has three levels: inside a warp (shuffles), across
//                   the warps of a block (shared memory), across the blocks
//                   of a decision (one 64-bit atomicMin).
//
// Memory layouts (all arrays in GPU global memory, filled by the host):
//   act[(d * H + t) * N + n]  int8: candidate n's action at imagined step t
//                             of decision d. Step-major, candidate-minor,
//                             like the RTL's ACT[t] word and vec_act.hex, so
//                             the 32 threads of a warp read 32 neighbouring
//                             bytes: one memory transaction ("coalesced").
//   state0[2 * d + 0]         int8 theta0 of decision d
//   state0[2 * d + 1]         int8 omega0
//   cost_out, theta_out, omega_out [d * N + n]  int32 results per candidate

#define FULL_MASK 0xffffffffu   // all 32 lanes of the warp take part
#define N_HID 16                // hidden neurons
#define WARP_SIZE 32

// ------------------------------------------------------------ the weights --
//
// __constant__ memory is a small (64 KB) read-only region with its own cache
// on every SM. It is fast in one case: when all threads of a warp read the
// SAME address in the same instruction, one read is broadcast to all 32
// threads. If the threads asked for different addresses, the reads would be
// serialized, one per distinct address. Here every thread of a warp executes
// the same loop over j at the same time, so they always ask for the same
// weight together: exactly the RTL's weight broadcast, where the sequencer
// reads one weight from ROM and all 32 lanes multiply it by their own value.
//
// The host (cuda_engine.py) fills these once, from rom/*.hex and
// rom/params.vh, with cudaMemcpy to the symbol's address.
//
// One difference from the RTL: there the lanes store no weights; the
// sequencer re-reads every weight from the central ROM in every imagined
// step, one per clock, and broadcasts it. In the PTX the compiler emits
// (test_bitexact.py prints the counts), each of the 45 constant words
// (16 + 16 + 8 + 2 + 1 + 2) is loaded once, before the step loop. What the final
// assembler does with them is not visible here (no disassembler without the
// CUDA Toolkit): it reports 40 registers per thread (bench.py), too few to
// hold all 45 words plus the loop's own values, and NVIDIA instructions can
// take a constant-memory word directly as an operand, so it most likely
// reads them from the constant cache inside the loop. Either way, all 32
// threads of a warp read the same word at the same time: the broadcast case.

__constant__ int c_W1[N_HID];   // row j of W1 packed as 4 int8: bytes
                                // (W1[j][0], W1[j][1], W1[j][2], 0), byte 0
                                // lowest. The 4th byte is zero padding: the
                                // network has 3 inputs, __dp4a takes 4.
__constant__ int c_B1[N_HID];   // B1[j], a wide signed bias (24-bit in the RTL)
__constant__ int c_W2[2][4];    // row k of W2: word q holds W2[k][4q .. 4q+3]
__constant__ int c_B2[2];       // B2[k]
__constant__ int c_S1;          // layer-1 shift (7)
__constant__ int c_S2[2];       // layer-2 shifts (8, 7)

// ------------------------------------------------------- small helpers --
//
// __device__ functions run on the GPU and are called from kernels; with
// __forceinline__ the compiler pastes them into the caller, so they cost
// nothing.

// Four values -> one 32-bit word holding their low bytes, b0 in bits 7..0,
// b1 in 15..8, b2 in 23..16, b3 in 31..24: the layout __dp4a reads.
// "& 0xff" keeps a value's two's-complement byte (-1 -> 0xff). Built in
// unsigned arithmetic, where shifting a bit into bit 31 is always defined.
__device__ __forceinline__ int pack_int8x4(int b0, int b1, int b2, int b3)
{
    const unsigned w = ((unsigned)b0 & 0xffu)
                     | (((unsigned)b1 & 0xffu) << 8)
                     | (((unsigned)b2 & 0xffu) << 16)
                     | (((unsigned)b3 & 0xffu) << 24);
    return (int)w;
}

// The requantization shift: floor(acc / 2^s), the RTL's ">>>" on a signed
// value. For an int, C++20 defines ">>" as exactly this arithmetic shift: the
// sign bit is copied in from the left, so -2367 >> 7 is -19 (floor of
// -18.49), not -18 (rounding toward zero, what acc / 128 would give). Before
// C++20 a right shift of a negative int was "implementation-defined"; CUDA's
// compilers have always made it arithmetic, and we pass --std=c++20 anyway.
// A logical shift would fill with zeros instead and turn every negative acc
// into a huge positive number (one of the planted bugs in test_bitexact.py).
__device__ __forceinline__ int requant_shift(int acc, int s)
{
    return acc >> s;
}

__device__ __forceinline__ int clamp_int(int v, int lo, int hi)
{
    return min(max(v, lo), hi);
}

// theta is an angle: 256 LSBs are one full turn, so leaving the range
// -128..127 on one side must re-enter on the other (+pi and -pi are the same
// angle). The RTL gets this for free by keeping the low 8 bits of the sum.
// In C++ an int has 32 bits, so the wrap has to be written out:
//   v + 128      moves the range -128..127 to 0..255
//   & 255        keeps the low 8 bits: v + 128 modulo 256, also for negative
//                values, because & works on the two's-complement bits
//   - 128        moves the range back
// Example: 127 + 3 = 130 -> 258 & 255 = 2 -> -126. omega must NOT do this
// (a wrapped velocity would flip its sign); it saturates instead, below.
__device__ __forceinline__ int wrap_int8(int v)
{
    return ((v + 128) & 255) - 128;
}

// ------------------------------------------------ one imagined step --
//
// DESIGN.md section 4 for one candidate, in this thread's registers.
// theta, omega and cost are updated in place; u is this step's action.
__device__ __forceinline__ void imagine_step(int& theta, int& omega, int& cost, const int u)
{
    // x = (theta, omega, u) as four signed bytes, the 4th zero
    const int x = pack_int8x4(theta, omega, u, 0);

    // Layer 1: 16 neurons, one __dp4a each.
    // __dp4a(a, b, c) reads a and b as four signed bytes each and returns
    //   c + a.byte0*b.byte0 + a.byte1*b.byte1 + a.byte2*b.byte2 + a.byte3*b.byte3
    // in 32 bits. With a = W1 row j, b = x and c = B1[j] that is
    //   acc1_j = B1[j] + W1[j][0]*theta + W1[j][1]*omega + W1[j][2]*u + 0*0
    // the three MACs the RTL spends 3 clocks on, plus the bias, in one
    // instruction. |acc1| <= 16,957 (results/golden.log), far inside 32 bits.
    //
    // The h values go straight into four packed words h4[0..3] (h[4q+i] in
    // byte i of h4[q]), ready for layer 2's __dp4a. h is 0..127, so each byte
    // is a valid signed int8 with the same value.
    //
    // "#pragma unroll" makes the compiler write the loop out 16 times. Then j
    // is a constant in each copy, so h4[j / 4] names a fixed register; with a
    // run-time index the array would have to live in (slow) local memory.
    int h4[4] = {0, 0, 0, 0};
    #pragma unroll
    for (int j = 0; j < N_HID; ++j) {
        const int acc1 = __dp4a(c_W1[j], x, c_B1[j]);
        const int h = clamp_int(requant_shift(acc1, c_S1), 0, 127);  // ReLU + saturate
        h4[j / 4] |= h << (8 * (j % 4));
    }

    // Layer 2: 2 outputs x 16 hidden values = 4 __dp4a per output, chained
    // through the accumulator argument, starting from the bias.
    int acc2_theta = c_B2[0];
    int acc2_omega = c_B2[1];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        acc2_theta = __dp4a(c_W2[0][q], h4[q], acc2_theta);
        acc2_omega = __dp4a(c_W2[1][q], h4[q], acc2_omega);
    }
    const int d_theta = requant_shift(acc2_theta, c_S2[0]);
    const int d_omega = requant_shift(acc2_omega, c_S2[1]);

    // State update: theta wraps (an angle), omega saturates (a speed).
    theta = wrap_int8(theta + d_theta);
    omega = clamp_int(omega + d_omega, -127, 127);

    // Cost of the new state, exactly golden.int_step's expression and order:
    //   theta'^2 + ((21 * omega'^2) >> 5) + (u^2 >> 11)
    // each term floored on its own. All terms are >= 0, so these >> are plain
    // divisions by 32 and 2048. The largest step cost is 26,975, so even
    // H = 30 (the longest horizon bench.py runs) stays far inside an int.
    cost += theta * theta + ((21 * omega * omega) >> 5) + ((u * u) >> 11);
}

// ----------------------------------------------- argmin inside a warp --
//
// On entry every thread holds one (cost, idx) pair; on return lane 0 holds
// the pair with the lowest cost, and among equal costs the lowest idx. That
// is what the RTL's scan n = 0..31 with strict < returns, and what
// golden.argmin_first returns.
//
// __shfl_down_sync(mask, v, offset) gives every thread the value v of the
// thread `offset` lanes above it, straight from that thread's register, with
// no memory involved. A thread whose partner would be past lane 31 gets its
// own v back (and then compares with itself, which changes nothing). mask =
// FULL_MASK says all 32 lanes take part; the _sync means they all reach this
// instruction before any value is exchanged.
//
//   offset 16: lane i keeps the better of lanes i and i+16
//   offset  8: lanes 0..7 hold the best of 4 candidates each
//   offset 4, 2, 1: lane 0 holds the best of all 32
//
// 5 rounds (log2 of 32) instead of the RTL's 32 clocks.
//
// Why compare the index too, not just "strict <, keep mine on a tie" like
// the RTL? The RTL compares candidates in index order, so whatever it holds
// always has a lower index than the newcomer. A tree does not: after the
// offset-16 round lane 0 may hold candidate 16 while lane 8 holds candidate
// 8, and if those two tie, "keep mine" would wrongly keep 16. Comparing
// (cost, idx) as a pair makes the result the smallest pair, which does not
// depend on the order of the comparisons at all. (A planted bug in
// test_bitexact.py drops the index comparison.)
__device__ __forceinline__ void warp_argmin(int& cost, int& idx)
{
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        const int other_cost = __shfl_down_sync(FULL_MASK, cost, offset);
        const int other_idx  = __shfl_down_sync(FULL_MASK, idx, offset);
        if (other_cost < cost || (other_cost == cost && other_idx < idx)) {
            cost = other_cost;
            idx  = other_idx;
        }
    }
}

// ============================================================ kernel 1 ==
//
// rollout_warp: the RTL mirror. Launch with blockDim.x = 32 (one warp) and
// gridDim.x = number of decisions. Block d imagines decision d's 32
// candidates; thread n is candidate n, i.e. lane n of the RTL engine.
//
// extern "C" keeps the function name unmangled, so the host can find it as
// "rollout_warp". __global__ marks a kernel: called from the host, runs on
// the GPU, once per thread.
extern "C" __global__ void rollout_warp(
    const signed char* __restrict__ act,      // [d][t][32] actions
    const signed char* __restrict__ state0,   // [d][2] theta0, omega0
    const int H,                              // imagined steps
    int* __restrict__ cost_out,               // [d][32] total cost per candidate
    int* __restrict__ theta_out,              // [d][32] theta after H steps
    int* __restrict__ omega_out,              // [d][32] omega after H steps
    int* __restrict__ best_idx_out,           // [d]
    int* __restrict__ best_cost_out)          // [d]
{
    const int d = blockIdx.x;     // which decision: one block each
    const int n = threadIdx.x;    // which candidate: 0..31 = the warp lane

    // Every candidate starts from the same measured state (all 32 threads
    // read the same two bytes: one transaction, broadcast to the warp).
    int theta = state0[2 * d + 0];
    int omega = state0[2 * d + 1];
    int cost  = 0;

    // The H steps are a chain: step t + 1 needs step t's state. No amount of
    // threads can run them in parallel; this loop is the latency floor, the
    // RTL's 83 clocks per step. "unroll 1" keeps it a plain loop with one
    // copy of the step (H is only known at run time anyway), so the PTX
    // shows exactly one step's instructions.
    const signed char* my_act = act + (size_t)d * H * WARP_SIZE + n;
    #pragma unroll 1
    for (int t = 0; t < H; ++t) {
        imagine_step(theta, omega, cost, (int)my_act[t * WARP_SIZE]);
    }

    // results of this candidate (neighbouring threads write neighbouring
    // words: coalesced again)
    const int out = d * WARP_SIZE + n;
    cost_out[out]  = cost;
    theta_out[out] = theta;
    omega_out[out] = omega;

    // the argmin over the 32 candidates of this decision
    int best_cost = cost;
    int best_idx  = n;
    warp_argmin(best_cost, best_idx);
    if (n == 0) {                  // lane 0 holds the answer
        best_idx_out[d]  = best_idx;
        best_cost_out[d] = best_cost;
    }
}

// ============================================================ kernel 2 ==
//
// rollout_blocks: N candidates per decision, N a multiple of blockDim.x, and
// blockDim.x a multiple of 32 (at most 1,024, the hardware limit per block).
// Launch with gridDim = (N / blockDim.x, number of decisions). Block
// (b, d) imagines candidates b * blockDim.x ... of decision d.
//
// The answer per decision comes back as one 64-bit key in best_key[d]:
//   key = (cost << 32) | idx
// The smallest key has the lowest cost, and among equal costs the lowest
// idx: the same order as warp_argmin. The host sets every best_key to
// 0xffffffffffffffff (larger than any real key) before the launch.
extern "C" __global__ void rollout_blocks(
    const signed char* __restrict__ act,      // [d][t][N]
    const signed char* __restrict__ state0,   // [d][2]
    const int H,
    const int N,                              // candidates per decision
    int* __restrict__ cost_out,               // [d][N]
    int* __restrict__ theta_out,              // [d][N]
    int* __restrict__ omega_out,              // [d][N]
    unsigned long long* __restrict__ best_key)  // [d]
{
    const int d = blockIdx.y;                               // decision
    const int n = blockIdx.x * blockDim.x + threadIdx.x;    // candidate in it
    const int warp = threadIdx.x / WARP_SIZE;               // warp in the block
    const int lane = threadIdx.x % WARP_SIZE;               // lane in the warp

    // the same imagined rollout as kernel 1
    int theta = state0[2 * d + 0];
    int omega = state0[2 * d + 1];
    int cost  = 0;
    const signed char* my_act = act + (size_t)d * H * N + n;
    #pragma unroll 1
    for (int t = 0; t < H; ++t) {
        imagine_step(theta, omega, cost, (int)my_act[(size_t)t * N]);
    }
    const size_t out = (size_t)d * N + n;
    cost_out[out]  = cost;
    theta_out[out] = theta;
    omega_out[out] = omega;

    // level 1: inside each warp, with shuffles (lane 0 gets the warp's best)
    int best_cost = cost;
    int best_idx  = n;
    warp_argmin(best_cost, best_idx);

    // level 2: across the warps of this block. Shuffles only reach threads
    // of the same warp, so the warps meet in __shared__ memory: fast on-chip
    // memory that all threads of one block can read and write. Each warp's
    // lane 0 posts its winner; __syncthreads() is a barrier for the whole
    // block (no thread passes it until every thread has reached it), so warp
    // 0 only reads after every warp has written. Then warp 0 runs the same
    // shuffle argmin over the (at most 32) warp winners.
    __shared__ int s_cost[WARP_SIZE];
    __shared__ int s_idx[WARP_SIZE];
    if (lane == 0) {
        s_cost[warp] = best_cost;
        s_idx[warp]  = best_idx;
    }
    __syncthreads();
    if (warp == 0) {
        const int n_warps = blockDim.x / WARP_SIZE;
        // lanes with no warp to read get a pair that can never win
        best_cost = (lane < n_warps) ? s_cost[lane] : 0x7fffffff;
        best_idx  = (lane < n_warps) ? s_idx[lane]  : 0x7fffffff;
        warp_argmin(best_cost, best_idx);

        // level 3: across the blocks of this decision. Blocks run on
        // different SMs in an order nobody controls, and cannot wait for
        // each other inside a launch. An atomic read-modify-write needs no
        // waiting: atomicMin compares and stores in one indivisible step, so
        // whichever order the blocks arrive in, the smallest key survives.
        // The cost is >= 0, so (unsigned) keeps its value.
        if (lane == 0) {
            const unsigned long long key =
                ((unsigned long long)(unsigned)best_cost << 32) | (unsigned)best_idx;
            atomicMin(&best_key[d], key);
        }
    }
}
