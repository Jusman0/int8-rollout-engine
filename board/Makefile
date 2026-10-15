# Tang Nano 20K bring-up kit. Run from this directory (cd board), or with
# make -C board <target> from the repo root.
#
#   make sim               board testbench (UART bit level), LANES 8 and 32, N decisions each
#   make bit               engine bitstream: build/engine_l$(LANES)$(VAR).fs, by default
#                          build/engine_l8_nodsp.fs (LANES ?= 8, SYNTH_OPTS ?= -nodsp, see below)
#   make bit SYNTH_OPTS=   the DSP-block build, build/engine_l8.fs (computed wrong on the board)
#   make bit LANES=32 SEED=2 SYNTH_OPTS=   32 lanes need the DSP blocks: build/engine_l32_s2.fs
#   make bit SYNTH_OPTS=-noalu  adders without carry chains: build/engine_l8_noalu.fs
#   make blinky            the LED counter: build/blinky.fs
#   make prog-blinky       load blinky into the FPGA's SRAM (lost at power-off)
#   make prog              load the same bitstream as make bit into SRAM
#   make flash             write it to the board's flash (kept at power-off)
#   make host-fake         host.py against a fake board on a pseudo-terminal: ping, replay 50, loop 1
#   make results           build l8 (seed 1, and -noalu) and l32 (seeds 1, 2); write results/build_l*.txt
#   make clean             delete build/
#
# Every tool command runs from the repo root: the engine's $readmemh opens
# rom/*.hex relative to the current directory, and `include "rom/params.vh"
# is found through -I. (the same as the top-level Makefile and synth/ecp5.ys).
#
# Tool paths default to the OSS CAD Suite in ~/oss-cad-suite/bin, called by
# full path (no "source environment"). Override any of them, for example:
#   make bit OSS=/opt/oss-cad-suite/bin
#   make host-fake PYTHON=../.venv/bin/python

OSS      ?= $(HOME)/oss-cad-suite/bin
YOSYS    ?= $(OSS)/yosys
NEXTPNR  ?= $(OSS)/nextpnr-himbaechel
PACK     ?= $(OSS)/gowin_pack
LOADER   ?= $(OSS)/openFPGALoader
IVERILOG ?= iverilog
VVP      ?= vvp
PYTHON   ?= python3
# every recipe runs from the repo root (cd $(ROOT)), so a relative path such as
# ../.venv/bin/python is made absolute here; a bare name like python3 stays as is
PY       := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))

# the chip on the Tang Nano 20K, its family name in nextpnr and gowin_pack,
# the clock in MHz (the board's 27 MHz oscillator U3, no PLL) and openFPGALoader's
# name for the board. (Comments sit on their own lines: make would keep the
# spaces before a trailing comment as part of the value.)
DEVICE   = GW2AR-LV18QN88C8/I7
FAMILY   = GW2A-18C
FREQ     = 27
BOARD    = tangnano20k

# engine lanes, placement seed, decisions per testbench run
LANES    ?= 8
SEED     ?= 1
N        ?= 20

# Extra synth_gowin options (see board/README.md): -nodsp builds the
# multipliers from LUTs instead of the chip's DSP blocks, -noalu builds adders
# from LUTs instead of carry chains, -nolutram builds memories from flip-flops
# instead of LUT RAM. They become part of the file names (-nodsp -> _nodsp),
# so variants never mix.
#
# The default is -nodsp because that is the build that computes correctly on
# the board: on 2026-10-14 the plain build (DSP blocks) gave the right clock
# count but wrong costs and states on every decision, -noalu (DSP blocks still
# on) failed the same way, and -nodsp was bit-exact on all 4,000 recorded
# decisions (board/results/hardware/). Pass SYNTH_OPTS= to build with the DSP
# blocks again.
SYNTH_OPTS ?= -nodsp

ROOT  := $(abspath $(CURDIR)/..)
BUILD := $(ROOT)/board/build
empty :=
space := $(empty) $(empty)
VAR   := $(subst $(space),,$(foreach o,$(SYNTH_OPTS),_$(subst -,,$(o))))
NAME  := l$(LANES)$(VAR)
# seed 1 gives the plain name; other seeds get _sN, so a seed sweep keeps every bitstream
SUF   := $(if $(filter 1,$(SEED)),,_s$(SEED))
FS    := $(BUILD)/engine_$(NAME)$(SUF).fs

RTL   = $(ROOT)/rtl/rollout_lane.v $(ROOT)/rtl/rollout_engine.v
# the weights and shifts the engine loads: new weights from golden.py must
# rebuild the simulator and the netlist, or replay compares against stale ROMs
ROM   = $(wildcard $(ROOT)/rom/*.hex) $(ROOT)/rom/params.vh
BRD   = $(ROOT)/board/uart_rx.v $(ROOT)/board/uart_tx.v $(ROOT)/board/top.v
# the same files, as the tools see them from the repo root
SRCS  = rtl/rollout_lane.v rtl/rollout_engine.v board/uart_rx.v board/uart_tx.v board/top.v

.PHONY: sim bit blinky prog prog-blinky flash host-fake results clean

# ------------------------------------------------------------ simulation
sim: $(BUILD)/tb_top_l8 $(BUILD)/tb_top_l32
	cd $(ROOT) && for l in 8 32; do \
		$(VVP) -n board/build/tb_top_l$$l +n=$(N) | tee board/build/tb_top_l$$l.log; \
		grep -q '^PASS' board/build/tb_top_l$$l.log || exit 1; \
	done
	@echo "board testbench PASSED at LANES 8 and 32"

$(BUILD)/tb_top_l%: tb_top.v $(BRD) $(RTL) $(ROM) $(ROOT)/sim/vectors.vh | $(BUILD)
	cd $(ROOT) && $(IVERILOG) -g2005 -Wall -I. -Isim -Ptb_top.LANES=$* \
		-o board/build/tb_top_l$* $(SRCS) board/tb_top.v

# the recorded decisions come from the golden model (top-level "make golden")
$(ROOT)/sim/vectors.vh:
	$(MAKE) -C $(ROOT) golden PYTHON=$(PY)

# ------------------------------------------------------------ bitstreams
# 1. yosys: Verilog -> Gowin cells (LUT4, ALU, DFF, MULT9X9, ...)
# 2. nextpnr: place and route on the chip, pins from the .cst file, timing
#    checked against FREQ
# 3. check_pins.py: every port landed on the pin the .cst asks for
# 4. gowin_pack: routed design -> .fs bitstream
bit: $(FS)

$(BUILD)/engine_$(NAME).json: $(BRD) $(RTL) $(ROM) | $(BUILD)
	cd $(ROOT) && $(YOSYS) -q -l board/build/yosys_$(NAME).log \
		-p "read_verilog -I. $(SRCS); chparam -set LANES $(LANES) top; synth_gowin -family gw2a $(SYNTH_OPTS) -top top -json board/build/engine_$(NAME).json"

$(FS): $(BUILD)/engine_$(NAME).json tangnano20k.cst
	cd $(ROOT) && $(NEXTPNR) --json board/build/engine_$(NAME).json \
		--write board/build/pnr_$(NAME)_s$(SEED).json \
		--device $(DEVICE) --vopt family=$(FAMILY) --vopt cst=board/tangnano20k.cst \
		--freq $(FREQ) --seed $(SEED) --quiet --log board/build/pnr_$(NAME)_s$(SEED).log
	cd $(ROOT) && $(PY) board/check_pins.py board/build/pnr_$(NAME)_s$(SEED).json top
	cd $(ROOT) && $(PACK) -c -d $(FAMILY) -o $@ board/build/pnr_$(NAME)_s$(SEED).json
	@grep "Max frequency" $(BUILD)/pnr_$(NAME)_s$(SEED).log | tail -n 1

blinky: $(BUILD)/blinky.fs

$(BUILD)/blinky.fs: blinky.v blinky.cst | $(BUILD)
	cd $(ROOT) && $(YOSYS) -q -l board/build/yosys_blinky.log \
		-p "read_verilog board/blinky.v; synth_gowin -family gw2a -top blinky -json board/build/blinky.json"
	cd $(ROOT) && $(NEXTPNR) --json board/build/blinky.json --write board/build/pnr_blinky.json \
		--device $(DEVICE) --vopt family=$(FAMILY) --vopt cst=board/blinky.cst \
		--freq $(FREQ) --quiet --log board/build/pnr_blinky.log
	cd $(ROOT) && $(PY) board/check_pins.py board/build/pnr_blinky.json blinky
	cd $(ROOT) && $(PACK) -c -d $(FAMILY) -o $@ board/build/pnr_blinky.json

# ------------------------------------------------------------ programming
# SRAM: fast, and gone at power-off, so a bad bitstream cannot stick.
# Close host.py (or any serial terminal) first: see board/README.md.
prog-blinky: $(BUILD)/blinky.fs
	$(LOADER) -b $(BOARD) $<

prog: $(FS)
	$(LOADER) -b $(BOARD) $<

# Flash: the board loads this bitstream by itself at every power-up.
flash: $(FS)
	$(LOADER) -b $(BOARD) -f $<

# ------------------------------------------------------------ host tests without a board
host-fake: | $(BUILD)
	cd $(ROOT) && $(PY) board/host.py --fake ping
	cd $(ROOT) && $(PY) board/host.py --fake replay --n 50
	cd $(ROOT) && $(PY) board/host.py --fake loop --episodes 1
	@# the abort path: a board that answers every 'D' with an error must make
	@# replay stop after 5 errors, say so, count nothing as passing, and FAIL
	cd $(ROOT) && ! $(PY) board/host.py --fake --fake-fail \
		--log board/build/logs/abort_fake.log replay --n 4000
	grep -q "stopped after 5 of 4000 decisions" $(BUILD)/logs/abort_fake.log
	grep -q "decisions answered: 0/5 sent" $(BUILD)/logs/abort_fake.log
	grep -q "^FAIL" $(BUILD)/logs/abort_fake.log
	@echo "host.py fake-board tests PASSED (including the abort path)"

# ------------------------------------------------------------ build reports
# LANES 8 (seed 1, and the -noalu variant: the fallback if carry chains
# misbehave on the chip), LANES 32 (seeds 1 and 2), and blinky.
results:
	$(MAKE) bit LANES=8 SEED=1 SYNTH_OPTS=
	$(MAKE) bit LANES=8 SEED=1 SYNTH_OPTS=-noalu
	$(MAKE) bit LANES=32 SEED=1 SYNTH_OPTS=
	$(MAKE) bit LANES=32 SEED=2 SYNTH_OPTS=
	$(MAKE) blinky
	mkdir -p results
	{ OSS=$(OSS) bash report.sh 8 1; echo; echo "--------"; echo; \
	  OSS=$(OSS) bash report.sh 8 1 -noalu; } > results/build_l8.txt
	OSS=$(OSS) bash report.sh 32 "1 2" > results/build_l32.txt

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
