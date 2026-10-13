# file: rtl/rollout_lane.v
# models: logical >> instead of arithmetic >>> in requantization
#
# >> always shifts in zeros, even on a signed value. A negative
# accumulator then turns into a large positive number instead of rounding
# toward minus infinity (DESIGN.md §4: "arithmetic shift right = floor").
# Applied to all three requantization shifts (h, d_theta, d_omega). Only
# the "wire signed" lines are touched, so the comments that mention
# "acc >>> S1" stay as they are.
/^ *wire signed/s|acc >>> S|acc >> S|
