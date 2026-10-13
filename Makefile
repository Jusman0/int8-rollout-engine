# make test       replay 1,000 recorded planner decisions through the RTL, bit-exact
# make test-full  all 4,000 decisions (20 closed-loop episodes)
# make golden     regenerate rom/ and sim/ from the Python golden model
# make clean      delete sim/ (vectors, compiled testbenches, logs)
#
# Other configurations (each gets its own compiled testbench):
#   make test LANES=8                          8 lanes, so 4 passes per decision
#   python3 model/golden.py --H 5 --out sim/h5 H = 5 vectors (also rewrites rom/,
#   make test VEC=sim/h5                       with identical contents)

PYTHON ?= python3
IVERILOG = iverilog -g2005 -Wall -I.
RTL = rtl/rollout_lane.v rtl/rollout_engine.v
N ?= 1000
LANES ?= 32
VEC ?= sim
TB = $(VEC)/tb_rollout_engine_l$(LANES)
LOG = $(VEC)/tb_l$(LANES).log

.PHONY: test test-full golden clean

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

sim:
	mkdir -p sim

clean:
	rm -rf sim
