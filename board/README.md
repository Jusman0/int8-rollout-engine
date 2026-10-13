# Tang Nano 20K bring-up

**Status: nothing in this folder has run on hardware yet.** The board (a Sipeed Tang Nano 20K, GW2AR-LV18QN88C8/I7) arrives Wednesday, October 14. Everything here was built and tested on a Mac without it: the board design in simulation, the host program against a fake board, the bitstreams through the open-source toolchain. This README is the plan for the first days with the board.

The goal: run the unchanged `rtl/rollout_engine.v` on the FPGA, feed it the same 4,000 recorded decisions as `make test-full`, and check every byte it sends back against the golden model. Then let it pick the actions of 20 Pendulum-v1 episodes, with the pendulum simulated in Python on the Mac.

| File | What it is |
|---|---|
| [`top.v`](top.v) | the board design: `rollout_engine` behind a UART command interpreter, plus reset, LEDs and logic-analyzer probes |
| [`uart_rx.v`](uart_rx.v), [`uart_tx.v`](uart_tx.v) | 8N1 serial receiver (2-flop synchronizer, mid-bit sampling, framing-error flag) and transmitter |
| [`tb_top.v`](tb_top.v) | testbench: talks to `top.v` only through its UART pins, bit by bit, and checks every reply byte |
| [`host.py`](host.py) | the Mac side: `ping`, `replay`, `loop`, and `--fake` (a golden-model board on a pseudo-terminal) |
| [`tangnano20k.cst`](tangnano20k.cst) | pin constraints for `top.v` |
| [`blinky.v`](blinky.v), [`blinky.cst`](blinky.cst) | the first smoke test: the LEDs count |
| [`check_pins.py`](check_pins.py) | after place and route: every port is on the pin it should be |
| [`report.sh`](report.sh) | build summary (resources, Fmax, slack) from the nextpnr logs |
| [`Makefile`](Makefile) | `sim`, `bit`, `blinky`, `prog`, `prog-blinky`, `flash`, `host-fake`, `results`, `clean` |
| [`results/`](results/) | build reports; hardware logs go here once there are any |

## Wednesday: the first 20 minutes

Tools are in `~/oss-cad-suite/bin`; the Makefile calls them by full path. `board/build/` is not in git, so build first (about 2 minutes; the LANES = 8 bitstream is the one to start with):

```bash
cd ~/int8-rollout-engine/board
make blinky bit LANES=8
```

1. **Plug in the board** with a USB-C *data* cable, straight into the Mac (no hub). Some LEDs may light from the factory demo.
2. **Find the serial port:** `ls /dev/cu.*`. Expect one or two `/dev/cu.usbserial-…` entries (the BL616 chip on the board is the USB-JTAG and the USB-UART). None at all: see troubleshooting, it's almost always the cable.
3. **Check JTAG:** `~/oss-cad-suite/bin/openFPGALoader -b tangnano20k --detect` should name the GW2A-18 chip.
4. **`make prog-blinky`.** The six LEDs count up in binary, about 6 counts a second. Hold S1 (the button wired to pin 88): the count pauses. If it only counts *while* S1 is held, the button is the other way round from the schematic: invert `btn_rst` in `top.v` before going on.
5. **`make prog LANES=8`.** LED 0 now blinks about once a second (the heartbeat: the design is loaded and the 27 MHz clock runs). This is SRAM programming: power-cycling the board brings the factory design back.
6. **`python3 host.py ping`.** Expect `reply: K 8 15 1` (LANES 8, H 15, protocol 1) and `PASS`. LED 3 flickers when a byte arrives, LED 4 when one is sent.
7. **`python3 host.py replay --n 20`.** Expect `20/20` on every line and `PASS`. Each decision is checked in full: all 32 costs, all 32 final states, best_idx, best_cost and the clock count.

If all seven pass, copy the logs that `host.py` printed (`board/build/logs/…`) into `board/results/` and commit them: they are the first hardware evidence. `ping` and `replay` need only Python's standard library; `loop` needs NumPy and gymnasium (use the virtualenv's python).

## The board, checked pin by pin

Every number below was checked against Sipeed's schematic (`Tang_Nano_20K_3921_Schematics.pdf`, and the newest `…_3923…`, which has the same connections for every pin used here), Sipeed's own example constraint files ([TangNano-20K-example](https://github.com/sipeed/TangNano-20K-example): `uart/`, `led/`, `hdmi/`), and apicula's `examples/tangnano20k.cst`. After every build, `check_pins.py` confirms that nextpnr put each port on the I/O site the schematic gives for that pin (for example pin 73 = IOT40A = nextpnr's `X39Y0/IOBA`). nextpnr stops on a misspelt port name, but a mistyped pin *number* is a valid constraint and builds silently; tried on a copy of `blinky.cst` with LED 3 on pin 81, `check_pins.py` failed it. Every port of every build here passed, which also shows that nextpnr's package table agrees with the schematic.

| Signal | FPGA pin | Polarity, wiring (schematic) | Sources that agree |
|---|---|---|---|
| 27 MHz clock | 4 | oscillator U3 → PIN04_SYS_CLK | schematic, Sipeed examples, apicula |
| LED 0–5 | 15, 16, 17, 18, 19, 20 | **active low**: 3.3 V → 510 Ω → LED → pin, so the pin lights the LED by pulling low | schematic, Sipeed wiki ("low voltage level enable"), Sipeed `blink_leds.v` (`leds = ~value`) |
| S1 | 88 | **1 while pressed**: 330 Ω to 3.3 V through the button, 1 kΩ to ground | schematic (KEY_S1 = PIN88_MODE0_KEY1), Sipeed `uart_top.v` (`rst_n = !rst` on pin 88) |
| S2 | 87 | the same as S1 (not used) | schematic (KEY_S2 = PIN87_MODE1_KEY2) |
| UART, FPGA → Mac | 69 | FPGA output, into the BL616's RX | schematic (PIN69_SYS_TX ↔ BL616_UART_RX), Sipeed `uart/src/top.cst`, LiteX |
| UART, Mac → FPGA | 70 | FPGA input, from the BL616's TX | schematic (PIN70_SYS_RX ↔ BL616_UART_TX), same |

Pins 87 and 88 are also the chip's MODE0/MODE1 configuration pins (that's why they have pull-downs). **Don't hold S1 or S2 while plugging the board in or while programming**: the FPGA may sample them when it configures.

### Probe pins for the logic analyzer

The two 20-pin headers are J5 and J6. Four of their pins connect to nothing else on the board (schematic nets HSPI_DIN0–3, renamed HSPI_D0–3 in the 3923 revision; they go only to the headers). `top.v` drives the probes on those:

| Probe | FPGA pin | Header pin | Analyzer channel (suggested) |
|---|---|---|---|
| `probe_busy`: engine busy | 73 | J6 pin 1 | CH0 |
| `probe_done`: engine done, stretched to 4 clocks | 74 | J6 pin 2 | CH1 |
| `probe_rx`: the line from the Mac | 72 | J5 pin 17 | CH2 |
| `probe_tx`: the line to the Mac | 71 | J5 pin 18 | CH3 |
| ground | | J5 pin 2, J5 pin 15 or J6 pin 20 | GND |

**Never connect a probe to J5 pin 1 (+5 V), J5 pin 16 or J6 pin 19 (+3.3 V).** To get the orientation right, find the pins by the FPGA pin numbers on Sipeed's pin diagram (wiki page, "Pin diagram"), and check with a multimeter before connecting anything: with USB plugged in, J5 pin 1 to J5 pin 2 reads 5 V.

The full headers, from the schematic (same in 3921 and 3923):

| Header pin | J5 | J6 |
|---|---|---|
| 1 | +5 V | 73 (free: probe_busy) |
| 2 | GND | 74 (free: probe_done) |
| 3 | 76 (BL616 SPI) | 75 (BL616 SPI) |
| 4 | 80 (SD card D2) | 85 (SD card D1) |
| 5 | 42 (LCD R3) | 77 (LCD clock) |
| 6 | 41 (LCD R4) | 15 (LED 0) |
| 7 | 56 (audio BCLK) | 16 (LED 1) |
| 8 | 54 (audio DIN) | 27 (LCD B7) |
| 9 | 51 (audio amplifier enable) | 28 (LCD B6) |
| 10 | 48 (LCD DE) | 25 (LCD HS) |
| 11 | 55 (audio LRCK) | 26 (LCD VS) |
| 12 | 49 (LCD backlight enable) | 29 (LCD B5) |
| 13 | 86 (BL616 SPI) | 30 (LCD B4) |
| 14 | 79 (WS2812 RGB LED) | 31 (LCD B3) |
| 15 | GND | 17 (LED 2) |
| 16 | +3.3 V | 20 (LED 5) |
| 17 | 72 (free: probe_rx) | 19 (LED 4) |
| 18 | 71 (free: probe_tx) | 18 (LED 3) |
| 19 | 53 (HDMI DDC clock) | +3.3 V |
| 20 | 52 (HDMI DDC data) | GND |

(LiteX's platform file lists the same headers with pins 75 and 76 swapped; the schematic, which is the source, has them as above. Neither is used here.)

## The protocol, byte by byte

115,200 baud, 8 data bits, no parity, 1 stop bit, no flow control. The host sends a command and then waits for the whole reply before sending anything else. Multi-byte numbers are little-endian (lowest byte first). θ, ω and actions are signed bytes (two's complement).

| Host sends | Bytes | Board replies | Bytes |
|---|---|---|---|
| `P` (0x50) | 1 | `K` (0x4B), LANES, H, PROTOCOL_VERSION (1) | 4 |
| `D` (0x44), θ0, ω0, then ACT[t][n] for t = 0..H−1 (outer), n = 0..31 (inner) | 2 + 32·H = 482 at H = 15 | `R` (0x52), best_idx, best_cost (3 bytes), cycles (2 bytes), then for n = 0..31: cost[n] (3 bytes), θ[n], ω[n] | 7 + 32·5 = 167 |
| any other byte | 1 | `E` (0x45), error code | 2 |

- `D`: each action byte goes into the engine's `act_*` port as it arrives (one write per byte, the same as the testbenches). After the last one, `top.v` pulses `start` (one clock later, so the write rule "only while busy is low" holds), waits for `done`, and reads every candidate back through the engine's `dbg_*` port. θ[n] and ω[n] are candidate n's final imagined state. `cycles` is the engine's own clock count, which must be (32/LANES)·(83·H + 2) + 34: 5,022 at LANES = 8, 1,281 at 32.
- Error codes: **1** unknown command byte; **2** framing error (a byte's stop bit was 0: wrong baud rate, or a glitch); **3** timeout (a `D` stopped arriving halfway, so a byte was lost). After any error the board waits until the host has been silent for 100 ms, throwing away whatever arrives (the rest of a broken command must not be read as new commands), and then sends `E` and the code. The error LED stays on until the next good `P` or `D`. `host.py` recovers by waiting 300 ms and flushing its input.
- Bytes that arrive while the board is computing or replying are ignored.

Example, decision 0 of `sim/` at LANES = 8 (state θ = −7, ω = 10; candidate 0's first action is −85):

```
host  -> 44 F9 0A AB EF ...                       'D', theta0 = -7, omega0 = 10, ACT[0][0] = -85, ACT[0][1] = -17, ... (482 bytes)
board -> 52 1E 8D 01 00 9E 13 80 18 00 E2 DB ...  'R', best 30, best_cost 397, cycles 5022, cost[0] = 6272, theta[0] = -30, omega[0] = -37, ...
```

### LEDs and probes

| LED | Pin | Meaning |
|---|---|---|
| 0 | 15 | heartbeat, about 0.8 Hz: bitstream loaded, clock running |
| 1 | 16 | engine busy (each 186 µs decision is stretched to 50 ms so it's visible) |
| 2 | 17 | error, until the next good command |
| 3 | 18 | a byte arrived (stretched) |
| 4 | 19 | a byte was sent (stretched) |
| 5 | 20 | reset held (S1) |

The probes come from flip-flops one clock (37 ns) after the signals they copy, so pulse widths are exact. `done` is a single 37 ns clock, shorter than the 41.7 ns between samples of a 24 MHz analyzer, so `probe_done` holds it for 4 clocks (148 ns).

## Friday: the full runs

1. **All 4,000 decisions at LANES = 8:** `python3 host.py replay --n 4000` (about 4 minutes: the UART needs 56 ms per decision for its 649 bytes, the engine 0.19 ms). Only the full set has the decisions that saturate ω (20 of them) or clamp a hidden value (32); the first 1,000 have neither ([`results/coverage.log`](../results/coverage.log)).
2. **20 closed-loop episodes:** `<venv>/bin/python host.py loop --episodes 20`. Every action applied to the simulated pendulum is the one the FPGA picked, and each episode's return must equal `sim/returns.txt` exactly (float `==`). Once this passes, it *is* hardware-in-the-loop: a real FPGA controller, a simulated plant. It isn't real-time: a decision takes at least 56 ms over the UART, longer than Pendulum-v1's 50 ms step, and the simulated pendulum simply waits.
3. **Measure the busy pulse in PulseView** (8 channels, 24 MHz, sigrok), wired as in the probe table. Trigger on CH0 rising, run `replay`, and measure the high time. **Expected, not measured:** 5,022 clocks × 37.04 ns = **186.0 µs** at LANES = 8, and 1,281 × 37.04 ns = **47.4 µs** at LANES = 32. At 24 MHz the measurement is good to about ±42 ns. `tb_top.v` already checks in simulation that the probe is high for exactly `cycles` clocks. While at it, the UART bits: 234 clocks = 8.67 µs on `probe_tx` (the board's 234-clock bit), 1/115,200 = 8.68 µs on `probe_rx` (the BL616's).
4. **LANES = 32:** `make prog LANES=32` and `replay --n 4000` again; then the second placement seed, `make prog LANES=32 SEED=2` (both bitstreams are built by `make results`). Then `loop --episodes 20` on the one that passes.
5. Save every log in `board/results/` and commit them. Only then does anything here become a hardware result.

## The toolchain risk, and why the bit-exact check is the answer

The open-source Gowin flow (yosys → nextpnr-himbaechel → apicula's `gowin_pack`) is reverse-engineered, and two open apicula issues report designs on this exact chip and board that build without an error and then compute wrong on silicon:

- [#514](https://github.com/YosysHQ/apicula/issues/514) (June 2026): a self-checking design computes wrong when yosys uses ALU carry cells, although RTL simulation, gate-level simulation and timing all pass; the same design built with `synth_gowin -noalu` (adders made of LUTs) works.
- [#541](https://github.com/YosysHQ/apicula/issues/541) (September 2026): wrong results for some nextpnr placement seeds and right for others; the failing part is a comparison in a final selection scan, much like this engine's argmin.

This engine is carry-heavy: the LANES = 8 build has 1,550 ALU (carry-chain) cells, in the lanes' 24-bit accumulators and cost adders, the counters, and the argmin's 24-bit comparison, which is also the slowest path nextpnr reports. It also uses 272 LUT-RAM blocks (RAM16SDP4) for the action memory and the hidden-value registers.

A broken carry chain doesn't crash anything. It makes a sum or a comparison wrong now and then: a cost off by a carry, an occasional wrong best index. Simulation can't see it, because the testbenches simulate the Verilog, not the bitstream (and #514 says even gate-level simulation passed). The only defence is to check the chip's answers, and `replay` checks all of them: every one of the 167 reply bytes of every decision against the golden model, including all 32 costs, so even a wrong cost that doesn't change the chosen plan shows up. That's also why bring-up starts at LANES = 8, the smallest build that exercises everything, and why the Friday plan tries a second placement seed.

If `replay` reports mismatches:

1. **What fails?** The log names the fields per decision. Only `cycles` wrong: the FSM or its counter. Costs wrong but states right: the cost adders. States wrong: the lane arithmetic. Everything wrong from decision 0: suspect the UART or the bitstream, not arithmetic.
2. **Is it repeatable?** Run the same `replay` twice. Different decisions failing each time points at timing or the board, not a wrong netlist.
3. **Another placement:** `make prog SEED=2` (and 3, 4 …). If some seeds pass and others don't, the problem is in place and route or packing.
4. **No carry chains:** `make prog SYNTH_OPTS=-noalu` builds every adder and comparator from plain LUTs (the #514 workaround; the LANES = 8 variant is built and fits, see `results/build_l8.txt`). If that passes where the normal build fails, the ALU mapping is the culprit. `SYNTH_OPTS=-nolutram` does the same for the LUT RAM.
5. Report it upstream with the failing decision numbers, the logs and the bitstream variants.

## Troubleshooting

- **`openFPGALoader` says `CRC Error` and `ID Verify Failed`** (and programming fails). This is the BL616's own firmware, not your bitstream: version 2024122312 is known not to work with openFPGALoader ([openFPGALoader troubleshooting](https://trabucayre.github.io/openFPGALoader/guide/troubleshooting.html), [issue #511](https://github.com/trabucayre/openFPGALoader/issues/511), where updating fixed it). The same firmware has been reported to run the UART at 4× the wrong speed (a 115,200 design needed 460,800 on the host), which would show up here as no reply or `E 2` to every ping. The firmware version is the USB serial number: `system_profiler SPUSBDataType` and look for the Sipeed/converter entry. Fix: update to **2025030317**, the latest firmware Sipeed lists for the Nano 20K, with Bouffalo's BLDevCube, following [Sipeed's update page](https://wiki.sipeed.com/hardware/en/tang/common-doc/update_debugger.html): hold the UPDATE button (behind the HDMI connector) while plugging in, and the board shows up as a `/dev/tty.usbmodem…` device.
- **Close anything holding the serial port before programming.** The BL616 serves JTAG and the UART over one USB connection; quit `screen`/`minicom`/a stuck `host.py` before `make prog`. (`host.py` closes the port when each command ends.) A precaution: I couldn't find Sipeed documenting the interaction either way.
- **No `/dev/cu.usbserial-*` at all**, but the board's LEDs light: a charge-only USB-C cable (power wires, no data). Use one known to carry data, connected directly to the Mac.
- **The board doesn't power up on a USB-C-to-C cable:** try a USB-A-to-C cable, or another cable or port.
- **`ping` gets no reply:** is LED 0 blinking (design loaded)? Does LED 3 flash when you ping (bytes arrive)? If not, try the other `/dev/cu.usbserial-*` with `--port`. `E 2` replies mean framing errors: a baud-rate mismatch (see the firmware item).
- **`openFPGALoader` can't find the board:** unplug and replug, no hub, and check `openFPGALoader --scan-usb`.

## What was run here, without a board

| Check | Command (in `board/`) | Result |
|---|---|---|
| Board testbench: ping, all three error replies, ±3% host baud rate, a short glitch, the reset button, then 20 recorded decisions with every reply byte and the busy-probe width checked | `make sim` (LANES 8 and 32) | PASS at both, see `build/tb_top_l*.log` |
| `host.py` against the fake board on a pseudo-terminal | `make host-fake PYTHON=<venv python>` | ping, replay 50 (50/50 bit-exact), loop 1 (return equal to `sim/returns.txt`): PASS |
| `host.py`'s vector reader on the whole set | `<venv python> host.py --fake --fake-lanes 32 replay --n 4000` | 4,000/4,000 bit-exact, PASS (5 s: a pty has no baud rate) |
| Bitstreams, each with the pin check | `make results` (15 minutes on the M2) | `build/engine_l8.fs`, `build/engine_l8_noalu.fs`, `build/engine_l32.fs`, `build/engine_l32_s2.fs`, `build/blinky.fs`; every port on its expected pin; summaries below |

### The builds

From [`results/build_l8.txt`](results/build_l8.txt) and [`results/build_l32.txt`](results/build_l32.txt) (`make results`; Yosys 0.69+286, nextpnr-himbaechel 0.11.1-59, apicula 0.35.dev2). Every number is a tool estimate; nothing is measured.

| Build | LUT positions (LUT4 + ALU) | Flip-flops | LUT RAM | MULT9X9 | nextpnr post-route Fmax | Slack at 27 MHz |
|---|---|---|---|---|---|---|
| LANES = 8, seed 1 | 9,245 of 20,736 (45%) | 3,913 (25%) | 272 of 648 | 8 of 96 | 117.03 MHz | +28.50 ns |
| LANES = 8, `-noalu`, seed 1 | 8,862 (43%) | 3,913 | 272 | 8 | 91.63 MHz | +26.13 ns |
| LANES = 32, seed 1 | 20,519 (99%) | 5,927 (38%) | 320 of 648 | 32 of 96 | 112.13 MHz | +28.12 ns |
| LANES = 32, seed 2 | the same cells | | | | 106.30 MHz | +27.63 ns |

- **LUT positions:** an ALU cell (one bit of a carry chain) takes the place of a LUT; in every placed design here no slice uses ALU *k* and LUT *k* together. So LANES = 32 fills 99% of the chip's logic. It placed and routed, but there is no room left for anything else.
- **Fmax:** nextpnr's Gowin estimates look optimistic. The same engine on an ECP5 reached 47 to 54 MHz ([`results/synth/summary.txt`](../results/synth/summary.txt)), with the lane multiplier on the critical path; here the slowest path is in the argmin scan in every build: from the ARG counter `n` or a `cost_mem` entry, through the `cost_mem[n]` read and the 24-bit compare, to the enable of `best_idx`/`best_cost`. The board doesn't depend on these numbers: it runs at 27 MHz from the crystal, and even three times nextpnr's slowest path (3 × 9.41 ns = 28 ns) would fit in the 37.04 ns clock period. That is arithmetic on an estimate, not a measurement.
- `blinky` is 38 LUT4s and 30 ALUs.
