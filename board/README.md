# Tang Nano 20K bring-up

**Status: running on a real FPGA (October 14, 2026).** On a Sipeed Tang Nano 20K (GW2AR-LV18QN88C8/I7) at 27 MHz, the LANES = 8 engine is bit-exact on all 4,000 recorded decisions and chose every action of 20 closed-loop Pendulum-v1 episodes with returns identical to the golden model's: hardware-in-the-loop, a real FPGA controller with a simulated plant. It needed one workaround: builds in which this open-source flow maps the lanes' multipliers onto the chip's DSP blocks computed wrong results on the chip, so the board build makes the multipliers from LUTs (`-nodsp`) (see [Results on hardware](#results-on-hardware)). The rest of this README is the bring-up plan as written before the board arrived, with notes where reality differed.

The goal: run the unchanged `rtl/rollout_engine.v` on the FPGA, feed it the same 4,000 recorded decisions as `make test-full`, and check every byte it sends back against the golden model. Then let it pick the actions of 20 Pendulum-v1 episodes, with the pendulum simulated in Python on the Mac.

| File | What it is |
|---|---|
| [`top.v`](top.v) | the board design: `rollout_engine` behind a UART command interpreter, plus reset, LEDs and logic-analyzer probes |
| [`uart_rx.v`](uart_rx.v), [`uart_tx.v`](uart_tx.v) | 8N1 serial receiver (2-flop synchronizer, mid-bit sampling, framing-error flag) and transmitter |
| [`tb_top.v`](tb_top.v) | testbench: talks to `top.v` only through its UART pins, bit by bit, and checks every reply byte |
| [`host.py`](host.py) | the Mac side: `ping`, `replay`, `loop`, and `--fake` (a golden-model board on a pseudo-terminal) |
| [`tangnano20k.cst`](tangnano20k.cst) | pin constraints for `top.v` |
| [`blinky.v`](blinky.v), [`blinky.cst`](blinky.cst) | the first smoke test: the LEDs count, and holding S1 lights one LED (checks LED and button polarity) |
| [`check_pins.py`](check_pins.py) | after place and route: every port is on the pin it should be |
| [`report.sh`](report.sh) | build summary (resources, Fmax, slack) from the nextpnr logs |
| [`Makefile`](Makefile) | `sim`, `bit`, `blinky`, `prog`, `prog-blinky`, `flash`, `host-fake`, `results`, `clean` |
| [`results/`](results/) | build reports, and the hardware logs in [`results/hardware/`](results/hardware/) |

## Results on hardware

October 14, 2026: Tang Nano 20K, 27 MHz oscillator, no PLL; BL616 debugger firmware 2025030317; host a MacBook over USB at 115,200 baud. Logs in [`results/hardware/`](results/hardware/) (each one's first line names the bitstream that was on the board).

| Test | Bitstream | Result | Log |
|---|---|---|---|
| `blinky`: LEDs count; holding S1 lights exactly one LED | `blinky.fs` | ✅ toolchain, clock, LED polarity (active low) and S1 polarity (1 when pressed) as designed | observed by eye, [`2026-10-14_blinky.md`](results/hardware/2026-10-14_blinky.md) |
| `ping` | default (DSP blocks) | ✅ `K 8 15 1` | [`2026-10-14_ping.log`](results/hardware/2026-10-14_ping.log) |
| `replay --n 20` | default: DSP multipliers ON, carry chains ON | ❌ every decision answered and `cycles` = 5,022 on all 20, but 0/20 costs, final states or best index right | [`…replay20_default_FAIL.log`](results/hardware/2026-10-14_replay20_default_FAIL.log) |
| `replay --n 20` | `-nodsp`: DSP multipliers OFF (built from LUTs), carry chains ON | ✅ 20/20 bit-exact | [`…replay20_nodsp_PASS.log`](results/hardware/2026-10-14_replay20_nodsp_PASS.log) |
| `replay --n 20` | `-noalu`: DSP multipliers ON, carry chains OFF | ❌ the same failure as the default build | [`…replay20_noalu_FAIL.log`](results/hardware/2026-10-14_replay20_noalu_FAIL.log) |
| `replay --n 4000` | `-nodsp` | ✅ 4,000/4,000 decisions: every cost, final θ and ω, best index and best cost; `cycles` = 5,022 on all | [`…replay4000_nodsp_PASS.log`](results/hardware/2026-10-14_replay4000_nodsp_PASS.log) |
| `loop --episodes 20` | `-nodsp` | ✅ the FPGA chose all 4,000 actions; 20/20 episode returns identical (float `==`) to `sim/returns.txt` | [`…loop20_nodsp_PASS.log`](results/hardware/2026-10-14_loop20_nodsp_PASS.log) |

**What the A/B builds show.** With the DSP blocks in use, the chip ran the right sequence (the clock count was exact every time) but computed wrong numbers; turning the carry chains off did not help, and turning the DSP blocks off did. The default and `-noalu` builds gave byte-identical wrong answers, which points to a deterministic logic fault rather than timing. So with this flow (Yosys 0.69+286 `synth_gowin`, nextpnr-himbaechel 0.11.1-59, apicula's `gowin_pack`) and this design, the builds that put the lanes' signed 8x8 multipliers on the GW2A's DSP blocks compute wrong, and the same RTL built without them is bit-exact; the carry chains (apicula issue #514) are not the cause here. Which stage of the flow is at fault is not isolated yet: the Yosys netlist has one MULT9X9 per lane with both inputs marked signed, which hints the fault is later in the flow, but that is not proof. RTL simulation and the place-and-route timing report did not show this; checking the chip's answers did. The next step is a post-synthesis simulation of the Yosys netlist (Yosys's own cell models), which would separate a Yosys mapping error from a nextpnr or `gowin_pack` one. The wrong results are not explained by a simple signedness error: a Python emulation of the engine with the multiplier treated as unsigned (8- or 9-bit, either or both operands) matches none of the chip's 160 wrong candidate results from the first 5 decisions, while the same emulation with a signed multiplier matches the golden model on all 160 ([`diag/emulate_multiplier.py`](diag/emulate_multiplier.py), with the chip's answers in [`diag/chip_l8_default_5decisions.json`](diag/chip_l8_default_5decisions.json)). Not tested yet: a product that arrives one clock late (a DSP output register set wrongly), which would also leave the clock count right. Narrowing it to a minimal design for an upstream bug report is open.

**Cost of the workaround.** Without DSP blocks, each lane's multiplier is built from LUTs: LANES = 8 still fits easily, and a LANES = 16 `-nodsp` build places and routes at 91% of the LUT4s (not run on the board yet). LANES = 32 would not fit without the DSP blocks (inferred from the 16-lane build's 91%; not built).

**Where reality differed from the plan below.**
- The very first `host.py` port search, right after programming, found no port (its log is empty). A `ping` with `--port` passed 30 seconds later, and a port search half a minute after that found `/dev/cu.usbserial-…1` at once. The second of the two ports is the UART.
- One more failing run, a `replay --n 20` at 21:19, is left out of `results/hardware/`: nothing records which bitstream was loaded for it (its answers are byte-identical to the default build's).
- With this debugger firmware on macOS, programming over JTAG after a `host.py` run worked without replugging (`openFPGALoader --detect` and `make prog` both succeeded straight after a UART session). Replug if programming fails.
- The logic analyzer had not arrived, so the busy-pulse timing below is still expected, not measured.

## Wednesday: the first 20 minutes

Tools are in `~/oss-cad-suite/bin`; the Makefile calls them by full path. `board/build/` is not in git, so build first (about 2 minutes; the LANES = 8 bitstream is the one to start with):

```bash
cd ~/int8-rollout-engine/board
make blinky bit LANES=8
```

1. **Plug in the board** with a USB-C *data* cable, straight into the Mac (no hub). Some LEDs may light from the factory demo.
2. **Find the serial port:** `ls /dev/cu.*`. Expect one or two `/dev/cu.usbserial-…` entries (the BL616 chip on the board is the USB-JTAG and the USB-UART). None at all: see troubleshooting, it's almost always the cable.
3. **Check JTAG:** `~/oss-cad-suite/bin/openFPGALoader -b tangnano20k --detect` should name the GW2A-18 chip.
4. **`make prog-blinky`.** The six LEDs count up in binary, about 6 counts a second. Now hold S1 (the button wired to pin 88): **exactly one LED should light (LED 0)**. That one pattern checks both polarities `top.v` relies on. One LED *dark* and five lit instead: the LED polarity is the other way round. A single LED lit *without* pressing, and counting only while S1 is held: the button polarity is the other way round (invert `btn_rst` in `top.v`). Either way, stop and fix that before going on.
5. **`make prog LANES=8`.** LED 0 now blinks about once a second (the heartbeat: the design is loaded and the 27 MHz clock runs). This is SRAM programming: power-cycling the board brings the factory design back.
6. **`python3 host.py ping`.** Expect `reply: K 8 15 1` (LANES 8, H 15, protocol 1) and `PASS`. LED 3 flickers when a byte arrives, LED 4 when one is sent.
7. **`python3 host.py replay --n 20`.** Expect `20/20` on every line and `PASS`. Each decision is checked in full: all 32 costs, all 32 final states, best_idx, best_cost and the clock count.

**If a later `make prog`/`make flash` fails, unplug and replug the USB cable.** Sipeed's FAQ (below) says the debugger disables JTAG while its UART is used. On October 14, with firmware 2025030317 on macOS, programming after a closed `host.py` session worked without replugging. Replugging loses nothing: an SRAM design is gone at power-off anyway.

If all seven pass, copy the logs that `host.py` printed (`board/build/logs/…`) into `board/results/` and commit them: they are the first hardware evidence. Copy only logs whose first line has no `--fake`: the fake-board runs are in the same folder (named `…_fake_…`) and are not hardware results. `ping` and `replay` need only Python's standard library; `loop` needs NumPy and gymnasium (use the virtualenv's python).

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

Pin 88 (S1) is also the chip's MODE0 configuration pin and pin 87 (S2) its MODE1 (that's why they have pull-downs). **Don't hold S1 or S2 while plugging the board in or while programming**: the FPGA may sample them when it configures.

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

(LiteX's platform file lists the same headers with 75/76 and 25/26 swapped; the schematic, which is the source, has them as above, and so does Sipeed's pin diagram. None of those four is used here.)

Two things that could make you second-guess the probe pins, neither of which is a problem:
- The 3921 schematic's DISPLAY sheet has labels `PIN73_SYS_LED3` … `PIN76_SYS_LED0` near the LCD connector. Those nets actually go to LED pins 17–20 through the `LCD_INIT0..3` hierarchical pins, not to FPGA pins 73–76; the 3923 revision renames them `PIN15..18_SYS_LED*`. Pins 71–74 connect only to the headers.
- Sipeed's pin diagram marks J5 pins 17 and 18 (FPGA 72 and 71) as the DS2 joystick adapter's ATTN/DATA. Don't fit the joystick adapter while probing.

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
4. **LANES = 32:** (as planned; not done, see Results on hardware: 32 lanes need the DSP blocks, `SYNTH_OPTS=`, which computed wrong on this board.) Replug if programming fails, `make prog LANES=32 SYNTH_OPTS=` and `replay --n 4000` again; then the second placement seed, `make prog LANES=32 SEED=2` (both bitstreams are built by `make results`). Then `loop --episodes 20` on the one that passes.
5. Save every log in `board/results/` and commit them. Only then does anything here become a hardware result.

## The toolchain risk, and why the bit-exact check is the answer

The open-source Gowin flow (yosys → nextpnr-himbaechel → apicula's `gowin_pack`) is reverse-engineered, and two open apicula issues report designs on this exact chip and board that build without an error and then compute wrong on silicon:

- [#514](https://github.com/YosysHQ/apicula/issues/514) (June 2026): a self-checking design computes wrong when yosys uses ALU carry cells, although RTL simulation, gate-level simulation and timing all pass; the same design built with `synth_gowin -noalu` (adders made of LUTs) works. The thread reports that a nextpnr fix (PR #1736) cured the original reproducers while a larger variant still fails; I haven't checked whether the nextpnr build used here includes it.
- [#541](https://github.com/YosysHQ/apicula/issues/541) (September 2026): a *signed* comparison evaluated as if it were unsigned, for some nextpnr placement seeds and not others. The engine's argmin compare is unsigned, so the closer match here is the lanes' signed clamp and saturation logic, and those paths are only exercised by the full set: the first 1,000 recorded decisions never saturate ω or clamp a hidden value. So Wednesday's `replay --n 20` cannot catch a #541-type bug; Friday's `replay --n 4000` can.

This engine is carry-heavy: the LANES = 8 build has 1,550 ALU (carry-chain) cells, in the lanes' 24-bit accumulators and cost adders, the counters, and the argmin's 24-bit comparison, which is also the slowest path nextpnr reports. It also uses 272 LUT-RAM blocks (RAM16SDP4) for the action memory and the hidden-value registers.

A broken carry chain doesn't crash anything. It makes a sum or a comparison wrong now and then: a cost off by a carry, an occasional wrong best index. Simulation can't see it, because the testbenches simulate the Verilog, not the bitstream (and #514 says even gate-level simulation passed). The only defence is to check the chip's answers, and `replay` checks all of them: every one of the 167 reply bytes of every decision against the golden model, including all 32 costs, so even a wrong cost that doesn't change the chosen plan shows up. That's also why bring-up starts at LANES = 8, the smallest build that exercises everything, and why the Friday plan tries a second placement seed.

If `replay` reports mismatches:

1. **What fails?** The log names the fields per decision. Only `cycles` wrong: the FSM or its counter. Costs wrong but states right: the cost adders. States wrong: the lane arithmetic. Everything wrong from decision 0: suspect the UART or the bitstream, not arithmetic.
2. **Is it repeatable?** Run the same `replay` twice. Different decisions failing each time points at timing or the board, not a wrong netlist.
3. **Another placement:** replug the board if programming fails, then `make prog SEED=2` (and 3, 4 …). If some seeds pass and others don't, the problem is in place and route or packing.
4. **No carry chains:** replug, then `make prog SYNTH_OPTS=-noalu` builds every adder and comparator from plain LUTs (the #514 workaround; the LANES = 8 variant is built and fits, see `results/build_l8.txt`). If that passes where the normal build fails, the ALU mapping is the culprit. `SYNTH_OPTS=-nolutram` does the same for the LUT RAM.
5. Report it upstream with the failing decision numbers, the logs and the bitstream variants.

## Troubleshooting

- **`openFPGALoader` says `CRC Error` and `ID Verify Failed`** (and programming fails). This is the BL616's own firmware, not your bitstream: openFPGALoader's [troubleshooting page](https://trabucayre.github.io/openFPGALoader/guide/troubleshooting.html) says firmware 2024122312 cannot be flashed on Linux-based systems, and [issue #511](https://github.com/trabucayre/openFPGALoader/issues/511) (on Debian) was fixed by a firmware update. Nothing has been reported for macOS either way. Separately, Sipeed's FAQ says to update the debugger firmware if the UART's actual baud rate is always 4× the set one, or its output is garbled; here that would show up as no reply or `E 2` to every ping. The firmware version is the USB serial number: `system_profiler SPUSBDataType` and look for the Sipeed/converter entry. Fix: update to **2025030317**, the latest firmware Sipeed lists for the Nano 20K, with Bouffalo's BLDevCube, following [Sipeed's update page](https://wiki.sipeed.com/hardware/en/tang/common-doc/update_debugger.html): hold the UPDATE button (behind the HDMI connector) while plugging in, and the board shows up as a `/dev/tty.usbmodem…` device.
- **Programming fails after using the UART: unplug and replug first.** The BL616 serves JTAG and the UART over one USB connection, and [Sipeed's Tang FAQ](https://wiki.sipeed.com/hardware/en/tang/common-doc/questions.html) (section "Programmer") says: "when using its uart function, Jtag is disabled". Its fix is to replug the Type-C cable; closing the terminal is not enough. So after any `host.py` run (or `screen`/`minicom`), replug before `make prog`/`make flash`. Once `host.py` has found the UART port it prints it: pass it with `--port` from then on, so `host.py` never opens the JTAG-side port while searching.
- **No `/dev/cu.usbserial-*` at all**, but the board's LEDs light: a charge-only USB-C cable (power wires, no data). Use one known to carry data, connected directly to the Mac.
- **The board doesn't power up on a USB-C-to-C cable:** try a USB-A-to-C cable, or another cable or port.
- **`ping` gets no reply:** is LED 0 blinking (design loaded)? Does LED 3 flash when you ping (bytes arrive)? If not, try the other `/dev/cu.usbserial-*` with `--port`. `E 2` replies mean framing errors: a baud-rate mismatch (see the firmware item).
- **LED 1 (busy) stuck on, and nothing answers, not even `ping`:** the engine started a decision and never raised `done` (top.v waits for it without a time limit), which on this toolchain most likely means a miscompiled bitstream. Press S1 to reset, then replug and try another placement (`SEED=2`) or `SYNTH_OPTS=-noalu` as in the mismatch checklist above.
- **`openFPGALoader` can't find the board:** unplug and replug, no hub, and check `openFPGALoader --scan-usb`.

## What was run here, without a board

| Check | Command (in `board/`) | Result |
|---|---|---|
| Board testbench: ping, all three error replies, a bad stop bit inside a `D` followed by 60 command-like bytes (exactly one `E 2`, only after the quiet timeout), ±3% host baud rate, a short glitch, the reset button, then 20 recorded decisions with every reply byte, the busy-probe width and the 4-clock done-probe width checked; throughout, no action byte may be written while the engine is busy | `make sim` (LANES 8 and 32) | PASS at both, see `build/tb_top_l*.log`. Planted bugs (an immediate error reply, an unstretched done probe, a forced write while busy) each make it FAIL |
| `host.py` against the fake board on a pseudo-terminal | `make host-fake PYTHON=<venv python>` | ping, replay 50 (50/50 bit-exact), loop 1 (return equal to `sim/returns.txt`), and the abort path (a fake board that answers every `D` with an error: replay stops after 5, counts nothing as passing, FAIL): PASS |
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
- **Fmax:** nextpnr's Gowin estimates look optimistic. The same engine on an ECP5 reached 47 to 54 MHz ([`results/synth/summary.txt`](../results/synth/summary.txt)), with the lane multiplier on the critical path; here the slowest path is in the argmin scan in every build: from the ARG counter `n` or a `cost_mem` entry, through the `cost_mem[n]` read and the 24-bit compare, to the enable of `best_idx`/`best_cost`. The board doesn't depend on these numbers: it runs at 27 MHz from the oscillator, and even three times the slowest path of any build here (the `-noalu` LANES = 8 build: 3 × 10.91 ns = 32.7 ns) would fit in the 37.04 ns clock period. That is arithmetic on an estimate, not a measurement.
- `blinky` is 38 LUT4s and 30 ALUs.
