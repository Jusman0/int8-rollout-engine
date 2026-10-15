"""
emulate_multiplier.py: was the wrong arithmetic on the board a simple
signedness bug in the lanes' 8x8 multiplier?

On 2026-10-14 the default Tang Nano 20K build (DSP blocks in use) answered
every decision with the right clock count but wrong costs and states.
chip_l8_default_5decisions.json holds that build's full answers to the first
5 recorded decisions (160 candidates), next to the golden model's. It was
captured on 2026-10-14 at about 20:42, between the default build's replay
(results/hardware/2026-10-14_replay20_default_FAIL.log, 20:41) and loading the
-nodsp build (20:44), with board/build/engine_l8.fs on the board, by calling
host.decide() on those 5 decisions; its best_idx/best_cost match that log.

This script re-implements one imagined step exactly as DESIGN.md section 4
(and rtl/rollout_lane.v) does it, with the multiplier swapped for variants:
signed (the design), or unsigned 8- or 9-bit on either or both operands, the
16-bit product read back as signed. For each variant it counts how many of
the 160 candidates it reproduces, for the golden model and for the chip.

Result on 2026-10-14: "signed" reproduces the golden model on 160/160 (so the
emulation is right) and no variant reproduces any of the chip's 160 answers,
so the fault is not one of these simple signedness errors.

Run from anywhere:  python3 board/diag/emulate_multiplier.py   (standard library only)
"""
import json, sys
import os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..") + "/"
def rd(name, bits):
    v = [int(l, 16) for l in open(R + "rom/" + name) if l.strip()]
    return [x - (1 << bits) if x >> (bits - 1) else x for x in v]
W1, B1, W2, B2 = rd("w1.hex", 8), rd("b1.hex", 24), rd("w2.hex", 8), rd("b2.hex", 24)
S1, S2 = 7, (8, 7)
def s16(x): x &= 0xFFFF; return x - 0x10000 if x & 0x8000 else x
def s24(x): x &= 0xFFFFFF; return x - 0x1000000 if x & 0x800000 else x
MULS = {
 "signed":        lambda a, b: a * b,
 "both unsigned8": lambda a, b: s16((a & 0xFF) * (b & 0xFF)),
 "both unsigned9": lambda a, b: s16((a & 0x1FF) * (b & 0x1FF)),
 "A(weight/first) unsigned8": lambda a, b: s16((a & 0xFF) * b),
 "B(operand) unsigned8": lambda a, b: s16(a * (b & 0xFF)),
 "A unsigned9": lambda a, b: s16((a & 0x1FF) * b),
 "B unsigned9": lambda a, b: s16(a * (b & 0x1FF)),
}
def rollout(th, om, plan, mul):
    cost = 0
    for u in plan:
        x = (th, om, u)
        h = []
        for j in range(16):
            acc = B1[j]
            for i in range(3): acc = s24(acc + mul(W1[3*j+i], x[i]))
            h.append(min(max(acc >> S1, 0), 127))
        d = []
        for k in range(2):
            acc = B2[k]
            for j in range(16): acc = s24(acc + mul(W2[16*k+j], h[j]))
            d.append(acc >> S2[k])
        th = ((th + d[0] + 128) & 255) - 128
        om = min(max(om + d[1], -127), 127)
        cost = (cost + mul(th, th) + ((21 * mul(om, om)) >> 5) + (mul(u, u) >> 11)) & 0xFFFFFF
    return cost, th, om
data = json.load(open(R + "board/diag/chip_l8_default_5decisions.json"))
for name, mul in MULS.items():
    match_want = match_got = total = 0
    for dec in data:
        for n in range(32):
            c, t, o = rollout(dec["theta0"], dec["omega0"], dec["plans"][n], mul)
            total += 1
            match_want += (c, t, o) == (dec["want"]["cost"][n], dec["want"]["theta"][n], dec["want"]["omega"][n])
            match_got  += (c, t, o) == (dec["got"]["cost"][n],  dec["got"]["theta"][n],  dec["got"]["omega"][n])
    print(f"{name:28s} matches golden: {match_want:3d}/{total}   matches CHIP: {match_got:3d}/{total}")
