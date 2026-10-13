# file: rtl/rollout_lane.v
# models: omega cost term is omega'^2, the 21/32 weight dropped
#
# DESIGN.md §4.2: c_step = theta'^2 + ((21 * omega'^2) >> 5) + (u^2 >> 11).
# Without the 21/32 the cost over-weights speed. The final states are
# still right (the cost never feeds back into the dynamics), so only the
# cost check and the best index can catch it.
s|((sq20 << 4) + (sq20 << 2) + sq20) >> 5|sq20[15:0]|
