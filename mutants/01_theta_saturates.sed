# file: rtl/rollout_lane.v
# models: theta saturates at +-127 instead of wrapping
#
# The omega clamp copied onto theta: an easy copy-paste slip, because the
# two write-backs sit next to each other. DESIGN.md §2 says theta WRAPS:
# +127 + 1 is -128, which for an angle is the same as going from +pi to -pi.
# Saturating would pin a pendulum that swings over the top at +pi.
# Caught only by decisions where an imagined theta crosses +-pi.
s|theta <= theta + d_theta\[7:0\];|theta <= (theta + d_theta > 127) ? 8'sd127 : (theta + d_theta < -127) ? -8'sd127 : theta + d_theta[7:0];|
