"""
host.py: the PC side of the Tang Nano 20K bring-up. Talks to board/top.v
over the board's USB-UART (115,200 baud, 8N1) with the protocol in
board/README.md, and checks every answer against the golden model.

  python3 board/host.py ping                    'P' -> expect "K 8 15 1" (LANES, H, version)
  python3 board/host.py replay --n 20           send recorded decisions from sim/, compare
                                                every field bit for bit
  python3 board/host.py loop --episodes 20      the board picks the actions of Pendulum-v1
                                                episodes; returns must equal sim/returns.txt
  python3 board/host.py --fake <command> ...    the same, against a fake board (the golden
                                                model) on a pseudo-terminal: no hardware

Options before the command: --port /dev/cu.usbserial-XXXX (default: find it),
--log FILE (default board/build/logs/<command>_<time>.log), --timeout SECONDS
for each reply, --fake-lanes 8|16|32 for the fake board.

ping and replay need only the standard library. loop (and the fake board's
'D' answers) import model/golden.py and model/planner.py, so they need NumPy
and gymnasium (requirements.txt). No pyserial: the port is set up with
termios, the Unix interface for terminals, which serial ports are.

Why a fake board? A pseudo-terminal (pty) is a pair of file descriptors that
behave like the two ends of a serial cable. host.py opens one end exactly as
it opens the real port (same termios settings), and a thread running the
golden model answers on the other end, byte for byte in the board's
protocol. That tests the host's port setup, byte packing and checking on
this computer before the board exists. It does not test the FPGA: the fake
board computes with the same golden model the host checks against.
"""

import argparse
import glob
import os
import select
import sys
import termios
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BAUD = 115200
N_CAND = 32

# protocol bytes (board/README.md)
CMD_PING, CMD_DECIDE = 0x50, 0x44            # 'P', 'D'
REP_PING, REP_RESULT, REP_ERROR = 0x4B, 0x52, 0x45   # 'K', 'R', 'E'
PROTOCOL_VERSION = 1
RESULT_LEN = 7 + 5 * N_CAND                  # 'R' reply: 167 bytes
ERROR_NAMES = {1: "unknown command byte",
               2: "framing error (a byte's stop bit was 0)",
               3: "timeout (a 'D' command stopped arriving halfway)"}
# top.v waits this long after the last byte before it answers an error
# (TIMEOUT_CLKS = 2,700,000 clocks at 27 MHz)
BOARD_QUIET_S = 0.1


class ProtocolError(Exception):
    """The board answered something that is not a valid reply."""


def expected_cycles(lanes, horizon):
    """Clocks from start to done (DESIGN.md, stage 2): PASSES x (83H + 2) + 34."""
    return (N_CAND // lanes) * (83 * horizon + 2) + 34


def s8(byte):
    """A byte (0..255) read as a signed 8-bit number (-128..127)."""
    return byte - 256 if byte >= 128 else byte


# --------------------------------------------------------------- the port --

def configure_raw(fd, baud):
    """Set a terminal (serial port or pty) to raw 8N1 at `baud`.

    "Raw" matters here: by default a terminal edits what passes through it.
    It would turn 0x0D into 0x0A, treat 0x03 as Ctrl-C, eat 0x11 and 0x13 as
    XON/XOFF flow control, and hold input back until a newline. Action bytes
    take every value from 0x00 to 0xFF, so all of that must be off.
    """
    iflag, oflag, cflag, lflag, _, _, cc = termios.tcgetattr(fd)
    # input: no break handling, no parity check, no 8th-bit stripping,
    # no CR/NL translation, no XON/XOFF
    iflag &= ~(termios.IGNBRK | termios.BRKINT | termios.PARMRK | termios.INPCK |
               termios.ISTRIP | termios.INLCR | termios.IGNCR | termios.ICRNL |
               termios.IXON | termios.IXOFF | termios.IXANY)
    oflag &= ~termios.OPOST                  # output: send bytes unchanged
    # no echo, no line editing, no signals from Ctrl-C/Ctrl-Z
    lflag &= ~(termios.ECHO | termios.ECHONL | termios.ICANON | termios.ISIG |
               termios.IEXTEN)
    # 8 data bits, no parity, 1 stop bit, receiver on, ignore modem lines,
    # and do not drop DTR when the port closes
    cflag &= ~(termios.CSIZE | termios.PARENB | termios.CSTOPB | termios.HUPCL)
    cflag |= termios.CS8 | termios.CREAD | termios.CLOCAL
    if hasattr(termios, "CRTSCTS"):
        cflag &= ~termios.CRTSCTS            # no RTS/CTS flow control: there are no such wires
    # reads return whatever has arrived, even nothing; timeouts come from select()
    cc[termios.VMIN] = 0
    cc[termios.VTIME] = 0
    speed = getattr(termios, f"B{baud}")
    termios.tcsetattr(fd, termios.TCSANOW, [iflag, oflag, cflag, lflag, speed, speed, cc])


class Port:
    """One open terminal: write all bytes, read exactly n bytes with a timeout."""

    def __init__(self, fd, name):
        self.fd = fd
        self.name = name

    @classmethod
    def open_serial(cls, path, baud=BAUD):
        # O_NOCTTY: do not make the port this process's controlling terminal
        # O_NONBLOCK: open() must not wait for a modem "carrier" signal
        fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        configure_raw(fd, baud)
        # opening the port can put a stray byte or glitch on the FPGA's RX line,
        # and the board may still be answering it: give that time to arrive,
        # then drop it, so the first command starts clean
        time.sleep(0.2)
        termios.tcflush(fd, termios.TCIOFLUSH)
        return cls(fd, path)

    def write(self, data, timeout=2.0):
        """All bytes, or ProtocolError if the port stops taking them for `timeout` s."""
        view = memoryview(bytes(data))
        deadline = time.monotonic() + timeout
        while view:
            left = deadline - time.monotonic()
            if left <= 0:
                raise ProtocolError(f"write timeout: {len(view)} bytes not sent")
            select.select([], [self.fd], [], left)   # wait for room in the driver's buffer
            try:
                sent = os.write(self.fd, view)
            except BlockingIOError:
                continue
            except OSError as e:                      # e.g. the board was unplugged
                raise ProtocolError(f"port error: {e}") from e
            view = view[sent:]
            deadline = time.monotonic() + timeout     # progress: restart the clock

    def read_exact(self, n, timeout):
        """Exactly n bytes, or ProtocolError if they take longer than timeout seconds."""
        data = bytearray()
        deadline = time.monotonic() + timeout
        while len(data) < n:
            left = deadline - time.monotonic()
            if left <= 0:
                raise ProtocolError(f"timeout: got {len(data)} of {n} bytes"
                                    + (f" ({data[:8].hex(' ')} ...)" if data else ""))
            ready, _, _ = select.select([self.fd], [], [], left)
            if ready:
                try:
                    chunk = os.read(self.fd, n - len(data))
                except BlockingIOError:
                    continue
                except OSError as e:                  # e.g. the board was unplugged
                    raise ProtocolError(f"port error: {e}") from e
                if not chunk:
                    raise ProtocolError("the port closed")
                data += chunk
        return bytes(data)

    def resync(self):
        """After an error: let the board time out and answer, then drop
        everything it sent, so the next command starts clean."""
        time.sleep(3 * BOARD_QUIET_S)
        termios.tcflush(self.fd, termios.TCIFLUSH)

    def close(self):
        os.close(self.fd)


def read_reply(port, timeout):
    """One reply: (kind, the bytes after the kind byte)."""
    kind = port.read_exact(1, timeout)[0]
    if kind == REP_PING:
        return kind, port.read_exact(3, timeout)
    if kind == REP_RESULT:
        return kind, port.read_exact(RESULT_LEN - 1, timeout)
    if kind == REP_ERROR:
        code = port.read_exact(1, timeout)[0]
        raise ProtocolError(f"board replied 'E' {code}: {ERROR_NAMES.get(code, 'unknown code')}")
    raise ProtocolError(f"unexpected reply byte 0x{kind:02x}")


def ping(port, timeout):
    """'P' -> (LANES, H, version)."""
    port.write([CMD_PING])
    kind, rest = read_reply(port, timeout)
    if kind != REP_PING:
        raise ProtocolError(f"'P' got reply 0x{kind:02x}, not 'K'")
    return rest[0], rest[1], rest[2]


def first_ping(port, timeout, log=None):
    """The first ping of a session, tried twice. A stray byte while the port
    opened can leave the board answering 'E' to it; one resync clears that."""
    try:
        return ping(port, timeout)
    except ProtocolError as e:
        if log:
            log.say(f"first ping: {e}; resyncing and trying once more")
        port.resync()
        return ping(port, timeout)


def decide(port, theta_q, omega_q, plans, horizon, timeout):
    """One decision on the board. plans[n][t] is candidate n's action at step t.

    Sends 'D', theta0, omega0 and the actions in ACT[t][n] order (step outer,
    candidate inner), then decodes the 'R' reply (little-endian numbers).
    """
    payload = [CMD_DECIDE, theta_q & 0xFF, omega_q & 0xFF]
    payload += [int(plans[n][t]) & 0xFF for t in range(horizon) for n in range(N_CAND)]
    port.write(payload)
    kind, r = read_reply(port, timeout)
    if kind != REP_RESULT:
        raise ProtocolError(f"'D' got reply 0x{kind:02x}, not 'R'")
    out = {"best_idx": r[0],
           "best_cost": r[1] | r[2] << 8 | r[3] << 16,
           "cycles": r[4] | r[5] << 8,
           "cost": [], "theta": [], "omega": []}
    for n in range(N_CAND):
        b = r[6 + 5 * n: 11 + 5 * n]          # 5 bytes per candidate
        out["cost"].append(b[0] | b[1] << 8 | b[2] << 16)
        out["theta"].append(s8(b[3]))
        out["omega"].append(s8(b[4]))
    return out


def compare(got, want, cycles_want):
    """Names of the fields that differ between the board's answer and the
    golden model's; an empty list means bit-exact."""
    bad = []
    if (got["best_idx"], got["best_cost"]) != (want["best_idx"], want["best_cost"]):
        bad.append("best")
    if got["cycles"] != cycles_want:
        bad.append("cycles")
    if got["cost"] != want["cost"]:
        bad.append("cost")
    if got["theta"] != want["theta"] or got["omega"] != want["omega"]:
        bad.append("state")
    return bad


def find_port(timeout):
    """The board's UART: the only /dev/cu.usbserial-* port, or, if the BL616
    shows more than one (it can show its JTAG side too), the one that
    answers a ping."""
    ports = sorted(glob.glob("/dev/cu.usbserial-*")) + sorted(glob.glob("/dev/ttyUSB*"))
    if not ports:
        others = " ".join(sorted(glob.glob("/dev/cu.*"))) or "none"
        sys.exit(f"no /dev/cu.usbserial-* port (is the board plugged in, with a data cable?). "
                 f"Ports present: {others}. Or pass --port.")
    if len(ports) == 1:
        return ports[0]
    for path in reversed(ports):              # the UART is usually the second interface
        try:
            port = Port.open_serial(path)
            try:
                first_ping(port, min(timeout, 0.5))
                # next time, skip the search: opening the JTAG side as a serial
                # port is harmless but slow, and confusing in the logs
                print(f"board UART found on {path}; pass --port {path} next time", flush=True)
                return path
            finally:
                port.close()
        except (OSError, ProtocolError):
            continue
    sys.exit(f"none of {' '.join(ports)} answered a ping; pass --port")


# ------------------------------------------------------- recorded vectors --

def load_vectors(vec_dir, count):
    """The first `count` decisions recorded by model/golden.py (format at the
    top of golden.py): inputs and every expected output, as plain ints."""
    vh = (vec_dir / "vectors.vh").read_text()
    horizon = int(vh.split("VEC_H")[1].split("=")[1].split(";")[0])
    total = int(vh.split("VEC_COUNT")[1].split("=")[1].split(";")[0])
    if not 1 <= count <= total:
        sys.exit(f"--n {count}: {vec_dir} holds {total} decisions")

    def lines(name, n):
        with open(vec_dir / name) as f:
            return [int(f.readline(), 16) for _ in range(n)]

    state = lines("vec_state.hex", count)
    act = lines("vec_act.hex", count * horizon)
    cost = lines("exp_cost.hex", count)
    theta = lines("exp_theta.hex", count)
    omega = lines("exp_omega.hex", count)
    best = lines("exp_best.hex", count)
    decisions = []
    for d in range(count):
        # wide words hold lane 0 in the lowest bits
        decisions.append({
            "theta0": s8(state[d] & 0xFF),
            "omega0": s8(state[d] >> 8),
            "plans": [[s8(act[d * horizon + t] >> (8 * n) & 0xFF) for t in range(horizon)]
                      for n in range(N_CAND)],
            "want": {"best_idx": best[d] >> 24,
                     "best_cost": best[d] & 0xFFFFFF,
                     "cost": [cost[d] >> (24 * n) & 0xFFFFFF for n in range(N_CAND)],
                     "theta": [s8(theta[d] >> (8 * n) & 0xFF) for n in range(N_CAND)],
                     "omega": [s8(omega[d] >> (8 * n) & 0xFF) for n in range(N_CAND)]},
        })
    return horizon, decisions


# ------------------------------------------------------------ the fake board --

class FakeBoard(threading.Thread):
    """Answers on the other end of a pty the way board/top.v answers on the
    UART, computing with the golden model instead of the FPGA. Independent
    of top.v: if host.py and top.v both follow board/README.md, both agree."""

    def __init__(self, fd, lanes, horizon, fail=False):
        super().__init__(daemon=True)            # dies with the host program
        self.fd, self.lanes, self.horizon = fd, lanes, horizon
        self.fail = fail                         # answer every 'D' with E 3 (tests the host's abort path)
        self.golden = self.p = None

    def recv(self, timeout):
        """One byte, or None after `timeout` seconds of quiet."""
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return None
        data = os.read(self.fd, 1)
        return data[0] if data else None

    def send(self, data):
        os.write(self.fd, bytes(data))

    def error(self, code):
        # like top.v: wait until the host has been quiet, then answer
        while self.recv(BOARD_QUIET_S) is not None:
            pass
        self.send([REP_ERROR, code])

    def run(self):
        while True:
            cmd = self.recv(None)
            if cmd == CMD_PING:
                self.send([REP_PING, self.lanes, self.horizon, PROTOCOL_VERSION])
            elif cmd == CMD_DECIDE:
                payload = []
                while len(payload) < 2 + N_CAND * self.horizon:
                    b = self.recv(BOARD_QUIET_S)
                    if b is None:
                        break
                    payload.append(b)
                if len(payload) < 2 + N_CAND * self.horizon or self.fail:
                    self.error(3)
                else:
                    self.send(self.answer(payload))
            elif cmd is not None:
                self.error(1)

    def answer(self, payload):
        if self.golden is None:                  # NumPy is only needed from here on
            sys.path.insert(0, str(ROOT / "model"))
            import golden
            import numpy as np
            self.golden, self.np = golden, np
            self.p = golden.fold(golden.load_float_model())
        h = self.horizon
        acts = payload[2:]
        plans = self.np.array([[s8(acts[t * N_CAND + n]) for t in range(h)]
                               for n in range(N_CAND)], dtype=self.np.int64)
        cost, theta, omega = self.golden.int_rollout(s8(payload[0]), s8(payload[1]), plans, self.p)
        best = self.golden.argmin_first(cost)
        cycles = expected_cycles(self.lanes, h)
        out = [REP_RESULT, best, *int(cost[best]).to_bytes(3, "little"),
               *cycles.to_bytes(2, "little")]
        for n in range(N_CAND):
            out += [*int(cost[n]).to_bytes(3, "little"), int(theta[n]) & 0xFF, int(omega[n]) & 0xFF]
        return out


# ---------------------------------------------------------------- logging --

class Log:
    """Every line goes to the log file; say() also prints it."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = open(path, "w")

    def write(self, line):
        self.f.write(line + "\n")
        self.f.flush()

    def say(self, line):
        print(line, flush=True)
        self.write(line)


# --------------------------------------------------------------- commands --

def cmd_ping(port, args, log):
    lanes, horizon, version = first_ping(port, args.timeout, log)
    log.say(f"reply: K {lanes} {horizon} {version}   (LANES = {lanes}, H = {horizon}, "
            f"protocol version {version})")
    ok = version == PROTOCOL_VERSION and lanes in (8, 16, 32)
    log.say("PASS" if ok else "FAIL: unexpected LANES or protocol version")
    return ok


def cmd_replay(port, args, log):
    lanes, horizon, version = first_ping(port, args.timeout, log)
    vh, decisions = load_vectors(ROOT / "sim", args.n)
    if horizon != vh:
        log.say(f"FAIL: the board has H = {horizon}, the vectors in sim/ have H = {vh}")
        return False
    cyc = expected_cycles(lanes, horizon)
    log.say(f"replay: {args.n} recorded decisions from sim/ -> {port.name} "
            f"(LANES = {lanes}, H = {horizon}; expect {cyc} cycles each)")
    counts = {"best": 0, "cycles": 0, "cost": 0, "state": 0}
    errors = 0
    t0 = time.monotonic()
    for d, dec in enumerate(decisions):
        try:
            got = decide(port, dec["theta0"], dec["omega0"], dec["plans"], horizon, args.timeout)
        except ProtocolError as e:
            errors += 1
            log.say(f"decision {d}: {e}")
            port.resync()
            if errors >= 5:
                log.say("5 protocol errors: stopping")
                break
            continue
        bad = compare(got, dec["want"], cyc)
        for field in bad:
            counts[field] += 1
        log.write(f"decision {d}: state ({dec['theta0']}, {dec['omega0']}) best {got['best_idx']} "
                  f"cost {got['best_cost']} cycles {got['cycles']} "
                  + ("bit-exact" if not bad else "MISMATCH " + " ".join(bad)))
        if bad and sum(counts.values()) <= 5:
            print(f"decision {d}: MISMATCH in {', '.join(bad)} (details in {log.path})")
        if (d + 1) % 500 == 0:
            print(f"  {d + 1} decisions ...", flush=True)
    dt = time.monotonic() - t0
    # count only what really happened: after 5 protocol errors the loop stops
    # early, and decisions that were never sent must not be reported as passing
    attempted = d + 1
    answered = attempted - errors
    if attempted < args.n:
        log.say(f"stopped after {attempted} of {args.n} decisions")
    log.say(f"decisions answered: {answered}/{attempted} sent (protocol errors: {errors})")
    log.say(f"costs: {answered - counts['cost']}/{answered} answered decisions bit-exact "
            f"(all 32 candidates)")
    log.say(f"final states: {answered - counts['state']}/{answered} answered decisions bit-exact "
            f"(all 32 theta and omega)")
    log.say(f"best_idx and best_cost: {answered - counts['best']}/{answered} answered decisions match")
    log.say(f"cycles = {cyc}: {answered - counts['cycles']}/{answered} answered decisions")
    # at 115,200 baud a decision's 482 + 167 bytes take 56 ms on the wire alone
    log.say(f"wall time {dt:.1f} s ({1000 * dt / attempted:.1f} ms per decision sent)")
    ok = attempted == args.n and errors == 0 and not any(counts.values())
    log.say("PASS" if ok else "FAIL")
    log.say(f"log: {log.path}")
    return ok


def cmd_loop(port, args, log):
    sys.path.insert(0, str(ROOT / "model"))
    import gymnasium as gym
    import golden
    import planner

    lanes, horizon, version = first_ping(port, args.timeout, log)
    cyc = expected_cycles(lanes, horizon)
    ref = {}
    for line in (ROOT / "sim" / "returns.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            e, r = line.split()
            ref[int(e)] = float(r)                # repr() round-trips exactly
    if args.episodes > len(ref):
        sys.exit(f"sim/returns.txt holds {len(ref)} episodes")
    p = golden.fold(golden.load_float_model())
    seed = 0                                      # the seed sim/ was made with

    state = {"ticks": 0, "bad": 0, "errors": 0}

    def board_engine(theta, omega, plans):
        """The engine planner.run_episode() calls each tick: the board decides."""
        theta_q, omega_q = planner.quantize_state(theta, omega)
        got = decide(port, theta_q, omega_q, plans, horizon, args.timeout)
        # the golden model on exactly the same inputs, every field
        cost, th, om = golden.int_rollout(theta_q, omega_q, plans, p)
        best = golden.argmin_first(cost)
        want = {"best_idx": best, "best_cost": int(cost[best]),
                "cost": [int(c) for c in cost], "theta": [int(v) for v in th],
                "omega": [int(v) for v in om]}
        bad = compare(got, want, cyc)
        if bad:
            state["bad"] += 1
            log.say(f"tick {state['ticks']}: state ({theta_q}, {omega_q}): MISMATCH in "
                    f"{', '.join(bad)} (board best {got['best_idx']}, golden {best})")
        state["ticks"] += 1
        return got["best_idx"]                    # the board's choice moves the pendulum

    log.say(f"loop: Pendulum-v1, seed {seed}, H = {horizon}, {args.episodes} episode(s); "
            f"the board (LANES = {lanes}) on {port.name} picks every action")
    env = gym.make("Pendulum-v1")
    same = 0
    t0 = time.monotonic()
    for e in range(args.episodes):
        try:
            ret = planner.run_episode(env, board_engine, seed, e, horizon)
        except ProtocolError as err:
            state["errors"] += 1
            log.say(f"episode {e}: {err}; episode abandoned")
            port.resync()
            continue
        match = ret == ref[e]                     # float ==, no tolerance
        same += match
        log.say(f"episode {e}: return {ret!r}, golden {ref[e]!r}: "
                f"{'MATCH' if match else 'MISMATCH'}")
    dt = time.monotonic() - t0
    n = state["ticks"]
    log.say(f"decisions made by the board: {n}; every reply field equal to the golden "
            f"model on the same inputs: {n - state['bad']}/{n}")
    log.say(f"episode returns identical to sim/returns.txt (float ==): {same}/{args.episodes}")
    log.say(f"wall time {dt:.1f} s ({1000 * dt / max(n, 1):.1f} ms per decision)")
    ok = same == args.episodes and state["bad"] == 0 and state["errors"] == 0
    log.say("PASS" if ok else "FAIL")
    log.say(f"log: {log.path}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", help="serial port (default: find /dev/cu.usbserial-*)")
    ap.add_argument("--fake", action="store_true",
                    help="no hardware: talk to a golden-model fake board over a pty")
    ap.add_argument("--fake-lanes", type=int, default=8, choices=(8, 16, 32),
                    help="LANES the fake board reports (default 8)")
    ap.add_argument("--fake-fail", action="store_true",
                    help="the fake board answers every 'D' with E 3 (tests the abort path)")
    ap.add_argument("--timeout", type=float, default=2.0,
                    help="seconds to wait for a reply (default 2)")
    ap.add_argument("--log", help="log file (default board/build/logs/<command>_<time>.log)")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("ping", help="'P': print LANES, H and the protocol version")
    rp = sub.add_parser("replay", help="send recorded decisions from sim/ and compare")
    rp.add_argument("--n", type=int, default=20, help="decisions (default 20, at most 4,000)")
    lp = sub.add_parser("loop", help="the board drives Pendulum-v1 episodes")
    lp.add_argument("--episodes", type=int, default=1, help="episodes (default 1, at most 20)")
    args = ap.parse_args()

    if args.log:
        log_path = Path(args.log)
    else:
        # e.g. board/build/logs/replay_20261014_153000.log; _fake for fake-board
        # runs, and _2, _3 ... if two runs start in the same second
        base = f"{args.command}{'_fake' if args.fake else ''}_{time.strftime('%Y%m%d_%H%M%S')}"
        log_path = ROOT / "board" / "build" / "logs" / f"{base}.log"
        k = 2
        while log_path.exists():
            log_path = log_path.with_name(f"{base}_{k}.log")
            k += 1
    log = Log(log_path)

    if args.fake:
        # the two ends of a pseudo-terminal: the host gets the "slave" end,
        # which is a real terminal device and takes the same termios setup
        # as a serial port; the fake board answers on the "master" end
        master, slave = os.openpty()
        configure_raw(slave, BAUD)
        port = Port(slave, f"fake board on {os.ttyname(slave)}")
        horizon = int((ROOT / "sim" / "vectors.vh").read_text().split("VEC_H")[1]
                      .split("=")[1].split(";")[0])
        FakeBoard(master, args.fake_lanes, horizon, fail=args.fake_fail).start()
    else:
        path = args.port or find_port(args.timeout)
        port = Port.open_serial(path)
    log.write(f"# board/host.py {' '.join(sys.argv[1:])}   ({time.strftime('%Y-%m-%d %H:%M:%S')}, "
              f"port {port.name})")

    commands = {"ping": cmd_ping, "replay": cmd_replay, "loop": cmd_loop}
    try:
        ok = commands[args.command](port, args, log)
    except ProtocolError as e:
        log.say(f"FAIL: {e}")
        ok = False
    finally:
        port.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
