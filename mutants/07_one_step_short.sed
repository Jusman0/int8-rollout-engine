# file: rtl/rollout_engine.v
# models: off-by-one in the step loop: imagines H-1 steps, not H
#
# A fencepost error: the FSM leaves the step loop one step early, so every
# candidate is imagined 14 steps ahead instead of 15. Every decision also
# takes 83 fewer clocks, but the testbench's clock check compares the
# engine's cycles output with its own count of the same clocks, so that
# check cannot see it. The costs and final states do.
s|if (t == H - 1) state <= S_STORE;|if (t == H - 2) state <= S_STORE;|
