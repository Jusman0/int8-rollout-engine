`timescale 1ns/1ps
// rollout_lane: imagines one candidate plan, one multiply per clock
// (DESIGN.md §4 and §6).
//
// The lane holds one candidate's imagined state (theta, omega), its 16 hidden
// values h[0..15], a multiply-accumulate register and the running cost. It
// has no FSM of its own: every clock rollout_engine broadcasts the same
// control bits, weight and bias to all lanes, and each lane applies them to
// its *own* operands. That is the GPU-warp idea: one instruction, many lanes.
//
// Two kinds of work happen in the same clock:
//
//   execute    mac: acc  <= (first ? bias : acc) + w * opnd
//              sq:  cost <= cost + scaled(opnd * opnd)
//              opnd is theta, omega, u or h[j], picked by src
//   writeback  one clock after a dot product's last MAC, acc holds the
//              finished sum; turn it into h[j], theta' or omega'. The
//              multiplier has already moved on to the next beat, so this
//              costs no extra clocks.
//
// Only one multiplier per lane: the same 8x8 multiplier does the network's
// MACs (weight x operand) and the cost squares (operand x operand).

module rollout_lane #(
    parameter ACC_W  = 24,       // worst case |acc| = 85,570 < 2^23 (rom/params.vh)
    parameter COST_W = 24,       // worst case cost 16 x 26,975 < 2^24
    parameter S1     = 7,        // shifts from rom/params.vh, passed in by
    parameter S2_0   = 8,        // rollout_engine
    parameter S2_1   = 7
) (
    input  wire                    clk,
    // start of a pass
    input  wire                    init,      // theta, omega <= start state; cost <= 0
    input  wire signed [7:0]       theta0,
    input  wire signed [7:0]       omega0,
    // execute stage (broadcast, the same for every lane)
    input  wire                    mac,       // acc <= (first ? bias : acc) + w * opnd
    input  wire                    first,     // first MAC of a dot product: start from the bias
    input  wire                    sq,        // cost <= cost + scaled(opnd * opnd)
    input  wire [1:0]              src,       // which operand, see SRC_* below
    input  wire [3:0]              j,         // h entry read when src = SRC_H
    input  wire signed [7:0]       w,         // broadcast weight
    input  wire signed [ACC_W-1:0] bias,      // broadcast bias, used when first = 1
    input  wire signed [7:0]       u,         // this lane's own action for this step
    // writeback stage (broadcast, one clock behind)
    input  wire                    wb_h,      // h[wb_j] <= clamp(acc >>> S1, 0, 127)
    input  wire [3:0]              wb_j,
    input  wire                    wb_theta,  // theta <= low 8 bits of theta + (acc >>> S2_0)
    input  wire                    wb_omega,  // omega <= clamp(omega + (acc >>> S2_1), -127, 127)
    // results
    output reg  signed [7:0]       theta,
    output reg  signed [7:0]       omega,
    output reg  [COST_W-1:0]       cost
);
    // operand codes; rollout_engine uses the same numbers
    localparam SRC_THETA = 2'd0, SRC_OMEGA = 2'd1, SRC_U = 2'd2, SRC_H = 2'd3;

    reg        [7:0]       h [0:15];     // hidden values, always 0..127
    reg signed [ACC_W-1:0] acc;

    // ------------------------------------------------------- operand mux
    // A continuous assign for the h read: an always @* that reads h[j]
    // makes Icarus warn that it is sensitive to all 16 words.
    wire [7:0] h_rd = h[j];

    reg signed [7:0] opnd;
    always @* begin
        case (src)
            SRC_THETA: opnd = theta;
            SRC_OMEGA: opnd = omega;
            SRC_U:     opnd = u;
            default:   opnd = h_rd;      // SRC_H. 0..127, so read as signed it stays positive
        endcase
    end

    // ------------------------------------------------------- the multiplier
    // MAC: weight x operand. Cost: operand x operand (a square).
    // Signed 8x8 fits in 16 bits: -128 x 127 = -16,256 up to -128 x -128 = 16,384.
    wire signed [7:0]  mul_a = sq ? opnd : w;
    wire signed [15:0] prod  = mul_a * opnd;

    // base must be signed so prod is sign-extended to ACC_W bits in the sum
    // (one unsigned operand would make the whole sum unsigned).
    wire signed [ACC_W-1:0] base = first ? bias : acc;

    always @(posedge clk)
        if (mac) acc <= base + prod;

    // ------------------------------------------------------- cost term
    // c_step = theta'^2 + ((21 * omega'^2) >> 5) + (u^2 >> 11), one term per
    // clock. Adding the three floored terms one at a time gives the same
    // total as adding them all at once. 21 = 16 + 4 + 1, so 21x is two
    // shifts and two adds: no multiplier for the constants (DESIGN.md §4.2).
    // 20 bits: 21 x 127^2 = 338,709 < 2^19.
    wire [19:0] sq20 = {4'd0, prod};     // a square is never negative

    reg [15:0] term;                     // at most 128^2 = 16,384
    always @* begin
        case (src)
            SRC_THETA: term = sq20[15:0];
            SRC_OMEGA: term = ((sq20 << 4) + (sq20 << 2) + sq20) >> 5;
            default:   term = sq20 >> 11;            // SRC_U (sq never uses SRC_H)
        endcase
    end

    // ------------------------------------------------------- requantize
    // acc is a signed register, so >>> is an arithmetic shift: it rounds
    // toward minus infinity, the same as Python's >> in model/golden.py.
    // The rounding offset 2^(S-1) is already inside the bias.
    wire signed [ACC_W-1:0] acc_s1  = acc >>> S1;
    wire signed [ACC_W-1:0] d_theta = acc >>> S2_0;
    wire signed [ACC_W-1:0] d_omega = acc >>> S2_1;

    // ReLU and saturate: negative -> 0 (sign bit set), above 127 -> 127
    wire [7:0] h_new = acc_s1[ACC_W-1] ? 8'd0   :
                       (acc_s1 > 127)  ? 8'd127 : acc_s1[7:0];

    // omega is sign-extended to ACC_W bits so the sum cannot overflow
    // before it is clamped
    wire signed [ACC_W-1:0] omega_sum = omega + d_omega;

    // ------------------------------------------------------- state and cost
    always @(posedge clk) begin
        if (init) begin
            theta <= theta0;
            omega <= omega0;
            cost  <= {COST_W{1'b0}};
        end
        if (wb_h)
            h[wb_j] <= h_new;
        // theta is an angle: an 8-bit add that drops its carry wraps +pi
        // around to -pi, which is exactly right (DESIGN.md §2)
        if (wb_theta)
            theta <= theta + d_theta[7:0];
        // omega is a speed: wrapping would flip its sign, so saturate
        if (wb_omega)
            omega <= (omega_sum >  127) ?  8'sd127 :
                     (omega_sum < -127) ? -8'sd127 : omega_sum[7:0];
        if (sq)
            cost <= cost + term;
    end
endmodule
