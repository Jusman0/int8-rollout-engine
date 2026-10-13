# file: rtl/rollout_engine.v
# models: omega' written back one clock late: its cost uses the OLD omega
#
# A read-after-write hazard. omega' is normally written back during the
# theta'^2 clock, one clock before the omega'^2 clock reads it. Here the
# write-back is triggered off the first COST beat instead, so it lands one
# clock later, in the same clock that squares omega: the square reads the
# omega from BEFORE this step. The value written is still right (acc does
# not change during COST), so every final state matches the golden model;
# only the costs are wrong. That is why the testbench checks costs and
# states separately.
s|ex_rq_omega <= iss_l2 \&\& (i == 2'd1) \&\& (j == N_HID - 1); // d_omega's 16th MAC|ex_rq_omega <= (state == S_COST) \&\& (i == 2'd0);        // 1st COST beat|
