"""
Checks where nextpnr put every top-level port, after place and route.

Why: a wrong pin number in the .cst file (18 typed as 81, say) is a valid
constraint, so nextpnr builds without a complaint and the port lands on the
wrong pin. (A misspelt port name is caught: nextpnr stops with
"Unconstrained IO".) The bitstream loads and then does nothing useful.
This script reads the routed design (nextpnr's --write JSON) and fails
unless every port sits on the I/O site this table expects. Tried on a copy
of blinky.cst with LED 3's pin changed to 81: nextpnr built it,
this script reported "led_n[3] ... WRONG".

The table comes from the Tang Nano 20K schematic (Tang_Nano_20K_3921, the
GW2AR-18 symbol), which names each package pin's I/O site: pin 15 is
IOL47A (left edge, row 47, pad A), pin 73 is IOT40A (top edge, column 40),
and so on. nextpnr counts rows and columns from 0 where Gowin counts from 1,
so IOL47A is nextpnr's X0Y46/IOBA, IOR30A is X55Y29/IOBA (the right edge is
column 55 on this chip) and IOT40A is X39Y0/IOBA. A match therefore also
checks that nextpnr's package database agrees with the schematic.

  python3 board/check_pins.py board/build/pnr_l8_s1.json top
  python3 board/check_pins.py board/build/pnr_blinky.json blinky
Standard library only. Exit status 0 means every port is where it belongs.
"""

import json
import sys

# port -> (package pin, Gowin I/O site from the schematic, nextpnr BEL)
EXPECTED = {
    "top": {
        "clk":        (4,  "IOL7A",  "X0Y6/IOBA"),
        "btn_rst":    (88, "IOR30A", "X55Y29/IOBA"),
        "uart_tx":    (69, "IOT50A", "X49Y0/IOBA"),
        "uart_rx":    (70, "IOT44B", "X43Y0/IOBB"),
        "led_n[0]":   (15, "IOL47A", "X0Y46/IOBA"),
        "led_n[1]":   (16, "IOL47B", "X0Y46/IOBB"),
        "led_n[2]":   (17, "IOL49A", "X0Y48/IOBA"),
        "led_n[3]":   (18, "IOL49B", "X0Y48/IOBB"),
        "led_n[4]":   (19, "IOL51A", "X0Y50/IOBA"),
        "led_n[5]":   (20, "IOL51B", "X0Y50/IOBB"),
        "probe_tx":   (71, "IOT44A", "X43Y0/IOBA"),
        "probe_rx":   (72, "IOT40B", "X39Y0/IOBB"),
        "probe_busy": (73, "IOT40A", "X39Y0/IOBA"),
        "probe_done": (74, "IOT34B", "X33Y0/IOBB"),
    },
    "blinky": {
        "clk":      (4,  "IOL7A",  "X0Y6/IOBA"),
        "btn":      (88, "IOR30A", "X55Y29/IOBA"),
        "led_n[0]": (15, "IOL47A", "X0Y46/IOBA"),
        "led_n[1]": (16, "IOL47B", "X0Y46/IOBB"),
        "led_n[2]": (17, "IOL49A", "X0Y48/IOBA"),
        "led_n[3]": (18, "IOL49B", "X0Y48/IOBB"),
        "led_n[4]": (19, "IOL51A", "X0Y50/IOBA"),
        "led_n[5]": (20, "IOL51B", "X0Y50/IOBB"),
    },
}


def placed_ports(path):
    """port name -> nextpnr BEL of the I/O buffer that drives or reads it"""
    design = json.load(open(path))
    found = {}
    for module in design["modules"].values():
        # the netlist numbers every wire; map each port bit's number to its name
        bit_to_port = {}
        for name, port in module["ports"].items():
            for i, bit in enumerate(port["bits"]):
                bit_to_port[bit] = f"{name}[{i}]" if len(port["bits"]) > 1 else name
        for cell in module["cells"].values():
            if cell["type"] == "IBUF":
                pad = cell["connections"]["I"][0]       # an input buffer reads the pad
            elif cell["type"] == "OBUF":
                pad = cell["connections"]["O"][0]       # an output buffer drives it
            else:
                continue
            if pad in bit_to_port:
                found[bit_to_port[pad]] = cell["attributes"].get("NEXTPNR_BEL", "unplaced")
    return found


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in EXPECTED:
        sys.exit(f"usage: {sys.argv[0]} <routed.json> {{{'|'.join(EXPECTED)}}}")
    want = EXPECTED[sys.argv[2]]
    got = placed_ports(sys.argv[1])
    bad = 0
    for port, (pin, site, bel) in want.items():
        ok = got.get(port) == bel
        bad += not ok
        print(f"  {port:11s} pin {pin:2d} {site:7s} -> {got.get(port, 'missing'):12s} "
              f"{'ok' if ok else 'WRONG, expected ' + bel}")
    extra = sorted(set(got) - set(want))
    for port in extra:
        print(f"  {port:11s} placed at {got[port]}, but not in the table")
    bad += len(extra)
    print("pins: all ports on the expected sites" if bad == 0 else f"pins: {bad} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
