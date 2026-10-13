# file: rtl/rollout_engine.v
# models: lanes read the live theta0/omega0 ports, not the copies made at start
#
# The engine copies the measured state into theta0_r/omega0_r when it sees
# start, and the lanes load it from there at INIT. Wiring the lanes to the
# input ports instead looks the same as long as the host holds theta0 and
# omega0 steady for the whole decision. The testbench changes them while
# busy is high, so this bug imagines every candidate from the wrong state.
s|.init(ex_init), .theta0(theta0_r), .omega0(omega0_r),|.init(ex_init), .theta0(theta0), .omega0(omega0),|
