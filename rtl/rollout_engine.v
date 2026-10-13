`timescale 1ns/1ps
// rollout_engine: sampling-based model-predictive control with the model
// rollouts in hardware (DESIGN.md §5-§6).
//
// The host writes N_CAND = 32 candidate plans of H int8 actions, sets the
// measured state and pulses start. The engine imagines every candidate H
// steps ahead through the int8 dynamics network, adds up each one's cost,
// and returns the cheapest: best_idx, best_cost.
//
// LANES rollout_lane instances imagine LANES candidates at once. One
// decision, LANES = 32:
//
//   INIT     1 clock    every lane loads (theta0, omega0) and clears its cost
//   per imagined step t = 0..H-1, 83 clocks:
//     L1    48 clocks   16 hidden neurons x 3 inputs (theta, omega, u)
//     L2    32 clocks   2 outputs (d_theta, d_omega) x 16 hidden values
//     COST   3 clocks   theta'^2, omega'^2, u^2
//   STORE    1 clock    copy every lane's cost and final state to the result buffer
//   DRAIN    1 clock    wait for that copy to land (see "pipeline" below)
//   ARG     32 clocks   scan the 32 costs for the smallest
//   DONE     1 clock    raise done
//
// A step has no overhead clocks: each lane writes h[j], theta' and omega'
// back while its multiplier already works on the next MAC (rollout_lane).
//
// With LANES = 8 or 16, INIT..STORE repeats once per group of LANES
// candidates (a "pass"). cycles counts every clock outside IDLE, so
//
//   cycles = (N_CAND / LANES) * (1 + 83 * H + 1) + 1 + 32 + 1
//          = (N_CAND / LANES) * (83 * H + 2) + 34
//
// More lanes buy more candidates per pass, not a shorter step: step t + 1
// needs step t's state, so the H steps of one candidate are a chain.
//
// Pipeline. Each clock the FSM *issues* one beat: it picks the ROM address
// and the control bits. Those are registered (ex_*) and broadcast to every
// lane, which *executes* the beat on the next clock. Registering the
// broadcast lets the ROMs and the action memory be synchronous (address this
// clock, data next clock, like block RAM), and every lane's multiplier
// starts from a register instead of from the ROM read and the FSM's decode
// logic. The write-back controls (wb_*) are the same bits one clock later
// still. The pipeline adds no clocks per step, only the one DRAIN clock.

module rollout_engine #(
    parameter LANES   = 32,          // candidates imagined at once: 8, 16 or 32
    parameter N_CAND  = 32,          // candidates per decision (the 5-bit ports assume 32)
    parameter H       = 15,          // imagined steps per plan, 1..16 (act_t is 4 bits)
    parameter ACC_W   = 24,          // lane accumulator; must be >= BIAS_W
    parameter W1_FILE = "rom/w1.hex",
    parameter B1_FILE = "rom/b1.hex",
    parameter W2_FILE = "rom/w2.hex",
    parameter B2_FILE = "rom/b2.hex"
) (
    input  wire               clk,
    input  wire               rst,
    // host writes ACT[t][n], candidate n's action at imagined step t
    // (only while busy is low)
    input  wire               act_we,
    input  wire [3:0]         act_t,
    input  wire [4:0]         act_n,
    input  wire [7:0]         act_data,
    // measured state, copied when start is seen
    input  wire signed [7:0]  theta0,
    input  wire signed [7:0]  omega0,
    input  wire               start,
    output reg                busy,
    output reg                done,       // one-clock pulse
    output reg  [4:0]         best_idx,   // valid from done until the next start
    output reg  [23:0]        best_cost,
    output reg  [15:0]        cycles,     // clocks from start to done
    // read back candidate dbg_n's cost and final state (for the testbench)
    input  wire [4:0]         dbg_n,
    output wire [23:0]        dbg_cost,
    output wire [7:0]         dbg_theta,
    output wire [7:0]         dbg_omega
);
    `include "rom/params.vh"    // S1, S2_0, S2_1, BIAS_W: written by model/golden.py

    localparam PASSES = N_CAND / LANES;
    localparam N_HID  = 16;
    localparam COST_W = 24;

    // operand codes, the same numbers as in rollout_lane
    localparam SRC_THETA = 2'd0, SRC_OMEGA = 2'd1, SRC_U = 2'd2, SRC_H = 2'd3;

    // synthesis translate_off
    initial
        if (N_CAND % LANES != 0 || H < 1 || H > 16 || ACC_W < BIAS_W) begin
            $display("rollout_engine: unsupported parameters LANES=%0d H=%0d ACC_W=%0d",
                     LANES, H, ACC_W);
            $finish;
        end
    // synthesis translate_on

    // ---------------------------------------------------------- memories
    reg [7:0]        w1_rom [0:47];          // W1[j][i] at 3*j + i
    reg [BIAS_W-1:0] b1_rom [0:15];          // B1[j]
    reg [7:0]        w2_rom [0:31];          // W2[k][j] at 16*k + j
    reg [BIAS_W-1:0] b2_rom [0:1];           // B2[k]

    initial begin
        $readmemh(W1_FILE, w1_rom);
        $readmemh(B1_FILE, b1_rom);
        $readmemh(W2_FILE, w2_rom);
        $readmemh(B2_FILE, b2_rom);
    end

    // Action memory. One word = one imagined step for one pass, LANES bytes
    // wide, so every lane gets its own action in a single read. Word
    // 16*p + t holds candidates p*LANES .. p*LANES + LANES-1 at step t; with
    // LANES = 32 that is exactly DESIGN.md's 256-bit ACT[t].
    reg [8*LANES-1:0] act_mem [0:16*PASSES-1];

    always @(posedge clk)
        if (act_we)
            act_mem[16 * (act_n / LANES) + act_t][8 * (act_n % LANES) +: 8] <= act_data;

    // Result buffer: every candidate's cost and final state, filled one pass
    // at a time. The argmin and the debug port read only this buffer, so they
    // work the same for any LANES.
    reg [COST_W-1:0] cost_mem  [0:N_CAND-1];
    reg [7:0]        theta_mem [0:N_CAND-1];
    reg [7:0]        omega_mem [0:N_CAND-1];

    assign dbg_cost  = cost_mem[dbg_n];
    assign dbg_theta = theta_mem[dbg_n];
    assign dbg_omega = omega_mem[dbg_n];

    // ---------------------------------------------------------- FSM state
    localparam S_IDLE  = 4'd0, S_INIT = 4'd1, S_L1  = 4'd2, S_L2   = 4'd3,
               S_COST  = 4'd4, S_STORE = 4'd5, S_DRAIN = 4'd6, S_ARG = 4'd7,
               S_DONE  = 4'd8;

    reg [3:0]        state;
    reg [4:0]        pass;       // which group of LANES candidates
    reg [3:0]        t;          // imagined step
    reg [3:0]        j;          // L1: hidden neuron; L2: hidden value
    reg [1:0]        i;          // L1: input (0 theta, 1 omega, 2 u); L2: output k; COST: term
    reg [4:0]        n;          // ARG: candidate being compared
    reg [15:0]       cnt;
    reg signed [7:0] theta0_r, omega0_r;

    wire iss_l1 = (state == S_L1);
    wire iss_l2 = (state == S_L2);

    // ---------------------------------------------------------- broadcast
    // execute stage: what every lane does this clock
    reg                    ex_init, ex_mac, ex_sq, ex_store, ex_first;
    reg [1:0]              ex_src;
    reg [3:0]              ex_j;
    reg signed [7:0]       ex_w;
    reg signed [ACC_W-1:0] ex_bias;
    reg [8*LANES-1:0]      ex_u;      // lane l's action in byte l
    reg [4:0]              ex_pass;
    // a dot product's last MAC is executing; write it back next clock
    reg                    ex_rq_h, ex_rq_theta, ex_rq_omega;
    // write-back stage
    reg                    wb_h, wb_theta, wb_omega;
    reg [3:0]              wb_j;

    always @(posedge clk) begin
        // control bits are reset, so the lanes do nothing until the first start
        if (rst) begin
            ex_init  <= 1'b0;  ex_mac      <= 1'b0;  ex_sq       <= 1'b0;
            ex_store <= 1'b0;  ex_rq_h     <= 1'b0;  ex_rq_theta <= 1'b0;
            ex_rq_omega <= 1'b0;
            wb_h     <= 1'b0;  wb_theta    <= 1'b0;  wb_omega    <= 1'b0;
        end else begin
            ex_init     <= (state == S_INIT);
            ex_mac      <= iss_l1 || iss_l2;
            ex_sq       <= (state == S_COST);
            ex_store    <= (state == S_STORE);
            ex_rq_h     <= iss_l1 && (i == 2'd2);                   // neuron j's 3rd MAC
            ex_rq_theta <= iss_l2 && (i == 2'd0) && (j == N_HID - 1); // d_theta's 16th MAC
            ex_rq_omega <= iss_l2 && (i == 2'd1) && (j == N_HID - 1); // d_omega's 16th MAC
            wb_h        <= ex_rq_h;
            wb_theta    <= ex_rq_theta;
            wb_omega    <= ex_rq_omega;
        end
        // data that rides along with the control bits; it is ignored unless
        // a control bit uses it, so it needs no reset
        ex_first <= iss_l1 ? (i == 2'd0) : (j == 4'd0);
        // L1 and COST both use i as the operand code (theta, omega, u)
        ex_src   <= iss_l2 ? SRC_H : i;
        ex_j     <= j;
        ex_w     <= iss_l1 ? w1_rom[3 * j + i] : w2_rom[{i[0], j}];
        // $signed so a narrower bias would be sign-extended to ACC_W bits
        ex_bias  <= iss_l1 ? $signed(b1_rom[j]) : $signed(b2_rom[i[0]]);
        ex_u     <= act_mem[16 * pass + t];
        ex_pass  <= pass;
        wb_j     <= ex_j;
    end

    // ---------------------------------------------------------- lanes
    wire [COST_W*LANES-1:0] lane_cost;
    wire [8*LANES-1:0]      lane_theta, lane_omega;

    genvar g;
    generate
        for (g = 0; g < LANES; g = g + 1) begin : gen_lane
            rollout_lane #(
                .ACC_W(ACC_W), .COST_W(COST_W), .S1(S1), .S2_0(S2_0), .S2_1(S2_1)
            ) lane (
                .clk(clk),
                .init(ex_init), .theta0(theta0_r), .omega0(omega0_r),
                .mac(ex_mac), .first(ex_first), .sq(ex_sq), .src(ex_src), .j(ex_j),
                .w(ex_w), .bias(ex_bias), .u(ex_u[8*g +: 8]),
                .wb_h(wb_h), .wb_j(wb_j), .wb_theta(wb_theta), .wb_omega(wb_omega),
                .theta(lane_theta[8*g +: 8]),
                .omega(lane_omega[8*g +: 8]),
                .cost(lane_cost[COST_W*g +: COST_W])
            );
        end
    endgenerate

    // STORE executes here, one clock after the FSM issues it, when the
    // pass's last cost term has landed. l is a loop constant, so each lane
    // writes its own fixed slot of this pass: no wide multiplexer.
    integer l;
    always @(posedge clk)
        if (ex_store)
            for (l = 0; l < LANES; l = l + 1) begin
                cost_mem [ex_pass * LANES + l] <= lane_cost [COST_W*l +: COST_W];
                theta_mem[ex_pass * LANES + l] <= lane_theta[8*l +: 8];
                omega_mem[ex_pass * LANES + l] <= lane_omega[8*l +: 8];
            end

    // ---------------------------------------------------------- FSM
    always @(posedge clk) begin
        done <= 1'b0;
        if (rst) begin
            state <= S_IDLE;
            busy  <= 1'b0;
        end else begin
            if (state != S_IDLE) cnt <= cnt + 16'd1;
            case (state)
                S_IDLE: if (start) begin
                    theta0_r <= theta0;
                    omega0_r <= omega0;
                    pass     <= 5'd0;
                    cnt      <= 16'd1;
                    busy     <= 1'b1;
                    state    <= S_INIT;
                end
                S_INIT: begin                       // lanes load the start state
                    t <= 4'd0;  j <= 4'd0;  i <= 2'd0;
                    state <= S_L1;
                end
                S_L1: begin                         // MAC W1[j][i] * x[i]
                    if (i == 2'd2) begin
                        i <= 2'd0;
                        if (j == N_HID - 1) begin j <= 4'd0; state <= S_L2; end
                        else j <= j + 4'd1;
                    end else i <= i + 2'd1;
                end
                S_L2: begin                         // MAC W2[k][j] * h[j], k = i
                    if (j == N_HID - 1) begin
                        j <= 4'd0;
                        if (i == 2'd1) begin i <= 2'd0; state <= S_COST; end
                        else i <= i + 2'd1;
                    end else j <= j + 4'd1;
                end
                S_COST: begin                       // cost term i: theta'^2, omega'^2, u^2
                    if (i == 2'd2) begin
                        i <= 2'd0;
                        if (t == H - 1) state <= S_STORE;
                        else begin t <= t + 4'd1; state <= S_L1; end
                    end else i <= i + 2'd1;
                end
                S_STORE: begin                      // save this pass's results
                    if (pass == PASSES - 1) state <= S_DRAIN;
                    else begin pass <= pass + 5'd1; state <= S_INIT; end
                end
                S_DRAIN: begin                      // the last STORE lands this clock
                    n     <= 5'd0;
                    state <= S_ARG;
                end
                S_ARG: begin
                    // strict <: on a tie the earlier (lower) index stays best
                    if (n == 5'd0 || cost_mem[n] < best_cost) begin
                        best_cost <= cost_mem[n];
                        best_idx  <= n;
                    end
                    if (n == N_CAND - 1) state <= S_DONE;
                    else n <= n + 5'd1;
                end
                S_DONE: begin
                    cycles <= cnt;
                    done   <= 1'b1;
                    busy   <= 1'b0;
                    state  <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
