# file: rtl/rollout_engine.v
# models: argmin compares with <=, so a tie picks the LAST index
#
# DESIGN.md §5: on a tie the lowest index wins (strict < while scanning
# n = 0..31). With <= a later candidate with the same cost replaces the
# best. best_cost is still right; only best_idx changes, and only in a
# decision where two candidates tie for the lowest cost. This is the bug
# tiny-tensor-core could not catch: none of its 10,000 images had a tie.
s|cost_mem\[n\] < best_cost|cost_mem[n] <= best_cost|
