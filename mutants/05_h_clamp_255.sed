# file: rtl/rollout_lane.v
# models: hidden values clamp at 255 instead of 127
#
# "h is unsigned 8-bit, so it can go up to 255", forgetting that h must
# fit the signed 8x8 multiplier as a positive number. An h of 128..255
# also reads as negative there (opnd is signed). DESIGN.md §2: h is
# clamped to [0, 127]. Caught only by decisions where some acc1 >>> S1 is
# above 127.
s|(acc_s1 > 127)  ? 8'd127|(acc_s1 > 255)  ? 8'd255|
