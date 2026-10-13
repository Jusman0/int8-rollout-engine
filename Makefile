# make test       replay 1,000 recorded planner decisions through the RTL, bit-exact
# make test-full  all 4,000 decisions (20 closed-loop episodes)
# make golden     regenerate rom/ and sim/ from the Python golden model
# make cosim      closed loop: the RTL picks every action of 20 pendulum episodes
# make mutants    planted bugs (DESIGN.md §7.4): every one must FAIL
# make sweep      clocks per decision, LANES 8/16/32 x H 5/15 -> results/sweep.log, results/cycles.log
# make synth      yosys resource report for the 32-lane engine on an ECP5 -> results/synth/
# make timing     place and route on an ECP5-45F, 3 seeds, LANES 32 and 8 -> results/synth/
# make clean      delete sim/ (vectors, compiled testbenches, logs)
#
# Other configurations (each gets its own compiled testbench):
#   make test LANES=8                          8 lanes, so 4 passes per decision
#   python3 model/golden.py --H 5 --out sim/h5 H = 5 vectors (also rewrites rom/,
#   make test VEC=sim/h5                       with identical contents)
#
# synth and timing need yosys and nextpnr-ecp5. Other tool names, for example
# the YoWASP pip packages:  make timing YOSYS=yowasp-yosys NEXTPNR=yowasp-nextpnr-ecp5

PYTHON ?= python3
YOSYS ?= yosys
NEXTPNR ?= nextpnr-ecp5
IVERILOG = iverilog -g2005 -Wall -I.
RTL = rtl/rollout_lane.v rtl/rollout_engine.v
N ?= 1000
LANES ?= 32
VEC ?= sim
TB = $(VEC)/tb_rollout_engine_l$(LANES)
LOG = $(VEC)/tb_l$(LANES).log

.PHONY: test test-full golden clean cosim mutants sweep synth timing

test: $(TB) $(VEC)/vec_state.hex
	vvp -n $(TB) +n=$(N) | tee $(LOG)
	@grep -q '^PASS' $(LOG) && echo "ALL TESTS PASSED"

test-full:
	$(MAKE) test N=4000

# golden.py writes rom/params.vh as well as the vectors. Listing it here makes
# the testbench, which includes it, rebuild whenever golden runs.
golden rom/params.vh sim/vec_state.hex sim/vectors.vh: model/golden.py model/planner.py model/dynamics_fp32.npz | sim
	$(PYTHON) model/golden.py

# vectors.vh gives the testbench VEC_COUNT and VEC_H; -I$(VEC) is how it finds it
$(TB): tb/tb_rollout_engine.v $(RTL) rom/params.vh $(VEC)/vectors.vh
	$(IVERILOG) -I$(VEC) -DVEC_DIR='"$(VEC)"' -Ptb_rollout_engine.LANES=$(LANES) \
		-o $@ $(RTL) tb/tb_rollout_engine.v

# Closed-loop co-simulation (DESIGN.md §7.3). cosim/host.py folds the weights
# itself from model/dynamics_fp32.npz, so rom/ must be up to date: hence the
# rom/params.vh prerequisite. Exit status 0 means PASS; the report is in
# sim/cosim/cosim.log. EPISODES=1 on the command line runs a quick check.
cosim: rom/params.vh
	PYTHON=$(PYTHON) LANES=$(LANES) bash cosim/run.sh

# Planted bugs. N is not passed on purpose: N = 1000 above is for make test.
# run_mutants.sh uses N = 300 unless you type N=... on the make command line
# (make hands command-line variables to the script's environment). The pipe
# into tee hides the script's exit status, so grep checks the verdict line.
mutants: rom/params.vh sim/vec_state.hex sim/vectors.vh
	PYTHON=$(PYTHON) bash mutants/run_mutants.sh | tee sim/mutants.log
	@grep -q '^ALL MUTANTS CAUGHT' sim/mutants.log

# Cycle sweep (DESIGN.md §7.5). perf/sweep.sh makes its own 200-decision
# vectors in sim/perf/ and runs "make test" six times.
sweep: rom/params.vh
	PYTHON=$(PYTHON) bash perf/sweep.sh

# Resources only (no place and route). Leaves the ~17 MB netlist in sim/.
synth: | sim
	mkdir -p results/synth
	$(YOSYS) -q -s synth/ecp5.ys

# Synthesis, then place and route with 3 seeds; see synth/timing_ecp5.sh
timing:
	YOSYS=$(YOSYS) NEXTPNR=$(NEXTPNR) bash synth/timing_ecp5.sh

sim:
	mkdir -p sim

clean:
	rm -rf sim
