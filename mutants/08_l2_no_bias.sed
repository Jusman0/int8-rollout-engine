# file: rtl/rollout_engine.v
# models: layer-2 dot products never restart from the bias
#
# "Forgot to clear the accumulator": first is never raised in layer 2, so
# d_theta's sum keeps adding onto neuron 15's layer-1 sum, and d_omega's
# onto d_theta's, and B2 is never added. Layer 1 is unchanged.
s|ex_first <= iss_l1 ? (i == 2'd0) : (j == 4'd0);|ex_first <= iss_l1 ? (i == 2'd0) : 1'b0;|
