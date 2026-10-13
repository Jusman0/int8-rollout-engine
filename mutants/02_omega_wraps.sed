# file: rtl/rollout_lane.v
# models: omega wraps (keeps the low 8 bits) instead of saturating
#
# The opposite mistake: treat omega like theta and let the 8-bit add wrap.
# For a speed that is wrong: +127 + 5 would become -124, a pendulum
# spinning fast one way suddenly spinning fast the other way.
# Two lines, one mutation: the clamp statement spans two lines, so line 1
# replaces its first line and line 2 deletes its second.
# Caught only by decisions where an imagined omega goes past +-127.
s|omega <= (omega_sum >  127) ?  8'sd127 :|omega <= omega_sum[7:0];|
/(omega_sum < -127) ? -8'sd127 : omega_sum\[7:0\];/d
