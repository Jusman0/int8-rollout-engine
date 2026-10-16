"""
pulse_widths.py: measure the busy and done pulses in a logic-analyzer capture.

The board design copies the engine's busy signal to FPGA pin 73 and a
stretched done pulse to pin 74 (board/README.md, "Probe pins"). A capture of
those two pins, saved as VCD by sigrok-cli, shows one busy pulse per decision.
This prints every pulse's width, in microseconds and in 27 MHz clocks, and
the time between decisions.

Capture (board running the engine, analyzer D0 on pin 73, D1 on pin 74):
  sigrok-cli -d fx2lafw --config samplerate=24M --channels D0,D1 \
      --triggers D0=r --wait-trigger --samples 12M -O vcd -o busy.vcd
  (and, while it waits, in another terminal: python3 host.py replay --n 20)

Run:  python3 board/diag/pulse_widths.py board/results/hardware/2026-10-15_busy_pulse.vcd
Standard library only. At 24 MHz the analyzer samples every 41.7 ns, so each
edge is known to within one sample.
"""
import re
import sys


def read_vcd(path):
    """(time in seconds, signal id, value) for every change in the file."""
    scale = None
    names = {}
    events = []
    t = 0.0
    for line in open(path):
        line = line.strip()
        m = re.match(r"\$timescale\s+(\d+)\s*(s|ms|us|ns|ps)\s+\$end", line)
        if m:
            scale = int(m.group(1)) * {"s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}[m.group(2)]
            continue
        m = re.match(r"\$var\s+\w+\s+1\s+(\S+)\s+(\S+)\s+\$end", line)
        if m:
            names[m.group(1)] = m.group(2)
            continue
        if line.startswith("#"):
            parts = line.split()
            t = int(parts[0][1:]) * scale
            changes = parts[1:]
        else:
            changes = [line] if re.match(r"^[01]\S+$", line) else []
        for c in changes:
            events.append((t, names.get(c[1:], c[1:]), int(c[0])))
    return events


def pulses(events, name):
    """(start, width) of every complete high pulse on one signal."""
    out, start = [], None
    for t, n, v in events:
        if n != name:
            continue
        if v == 1:
            start = t
        elif start is not None:
            out.append((start, t - start))
            start = None
    return out


if __name__ == "__main__":
    ev = read_vcd(sys.argv[1])
    busy, done = pulses(ev, "D0"), pulses(ev, "D1")
    print(f"busy (D0, FPGA pin 73): {len(busy)} complete pulses")
    for s, w in busy:
        print(f"  at {s * 1e3:9.3f} ms: {w * 1e6:8.3f} us = {w * 27e6:7.1f} clocks at 27 MHz")
    print(f"done (D1, FPGA pin 74): {len(done)} pulses, widths "
          + ", ".join(f"{w * 1e9:.1f} ns" for _, w in done[:3]) + (" ..." if len(done) > 3 else ""))
    gaps = [busy[i + 1][0] - busy[i][0] for i in range(len(busy) - 1)]
    if gaps:
        print(f"time between decisions: {min(gaps) * 1e3:.2f} to {max(gaps) * 1e3:.2f} ms")
        print(f"engine busy for {busy[0][1] / (sum(gaps) / len(gaps)) * 100:.2f}% of each decision's wall time")
