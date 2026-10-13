// tb_rollout_engine: replays the decisions recorded by model/golden.py
// through rollout_engine and checks, for every decision, that
//   - all 32 candidate costs,
//   - all 32 final imagined states (theta, omega), read through the dbg port,
//   - best_idx and best_cost
// match the Python golden model bit for bit. Also reports the clocks per
// decision, cross-checked against the testbench's own count, and for
// decision 0 how many clocks the FSM spent in each state.
//
// Each decision is driven the way a host would: write the 32 x H actions
// one byte per clock through the act_* port, set the state, pulse start,
// wait for done. While the engine is busy the testbench changes theta0 and
// omega0, which the engine must have copied at start. A watchdog turns an
// FSM that never finishes into a FAIL.
//
// Built by the Makefile with
//   -I$(VEC)                       finds vectors.vh (VEC_COUNT, VEC_H)
//   -DVEC_DIR='"$(VEC)"'           where the vector files are
//   -Ptb_rollout_engine.LANES=8    lanes in the engine (default 32)
// Runs all VEC_COUNT decisions by default; +n=1000 for fewer. Asking for
// 0 decisions, or for more than VEC_COUNT, prints FAIL.
`timescale 1ns/1ps

`ifndef VEC_DIR
`define VEC_DIR "sim"
`endif

module tb_rollout_engine;
    parameter LANES = 32;
    `include "vectors.vh"       // VEC_COUNT, VEC_H: written by model/golden.py

    localparam N_CAND   = 32;
    // No decision needs more than 4 x (83 x 16 + 2) + 34 = 5,354 clocks
    // (LANES = 8, H = 16), so 10,000 clocks without done means a hung FSM.
    localparam WATCHDOG = 10000;

    reg [15:0]          vec_state [0:VEC_COUNT-1];          // {omega0, theta0}
    reg [8*N_CAND-1:0]  vec_act   [0:VEC_COUNT*VEC_H-1];    // line d*H + t = ACT[t]
    reg [24*N_CAND-1:0] exp_cost  [0:VEC_COUNT-1];
    reg [8*N_CAND-1:0]  exp_theta [0:VEC_COUNT-1];
    reg [8*N_CAND-1:0]  exp_omega [0:VEC_COUNT-1];
    reg [31:0]          exp_best  [0:VEC_COUNT-1];          // {best_idx[7:0], best_cost[23:0]}

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg         rst = 1'b1, act_we = 1'b0, start = 1'b0;
    reg  [3:0]  act_t = 4'd0;
    reg  [4:0]  act_n = 5'd0, dbg_n = 5'd0;
    reg  [7:0]  act_data = 8'd0, theta0 = 8'd0, omega0 = 8'd0;
    wire        busy, done;
    wire [4:0]  best_idx;
    wire [23:0] best_cost, dbg_cost;
    wire [7:0]  dbg_theta, dbg_omega;
    wire [15:0] cycles;

    rollout_engine #(.LANES(LANES), .H(VEC_H)) dut (
        .clk(clk), .rst(rst),
        .act_we(act_we), .act_t(act_t), .act_n(act_n), .act_data(act_data),
        .theta0(theta0), .omega0(omega0),
        .start(start), .busy(busy), .done(done),
        .best_idx(best_idx), .best_cost(best_cost), .cycles(cycles),
        .dbg_n(dbg_n), .dbg_cost(dbg_cost), .dbg_theta(dbg_theta), .dbg_omega(dbg_omega)
    );

    integer n_dec, d, t, c, waited;
    integer bad_cost = 0, bad_state = 0, bad_best = 0, bad_clk = 0, bad_proto = 0;
    integer min_cyc = 65535, max_cyc = 0;
    reg     cost_ok, state_ok;
    integer state_clocks [0:15];     // decision 0: clocks spent in each FSM state

    initial begin
        if (!$value$plusargs("n=%d", n_dec)) n_dec = VEC_COUNT;
        // Running 0 decisions, or fewer than asked for, must not print PASS:
        // "PASS" has to mean that every decision asked for was checked.
        if (n_dec < 1 || n_dec > VEC_COUNT) begin
            $display("FAIL: +n=%0d, but %0s holds %0d decisions (allowed: 1..%0d)",
                     n_dec, `VEC_DIR, VEC_COUNT, VEC_COUNT);
            $finish;
        end
        $readmemh({`VEC_DIR, "/vec_state.hex"}, vec_state);
        $readmemh({`VEC_DIR, "/vec_act.hex"},   vec_act);
        $readmemh({`VEC_DIR, "/exp_cost.hex"},  exp_cost);
        $readmemh({`VEC_DIR, "/exp_theta.hex"}, exp_theta);
        $readmemh({`VEC_DIR, "/exp_omega.hex"}, exp_omega);
        $readmemh({`VEC_DIR, "/exp_best.hex"},  exp_best);
        for (c = 0; c < 16; c = c + 1) state_clocks[c] = 0;

        repeat (2) @(negedge clk);
        rst = 1'b0;

        for (d = 0; d < n_dec; d = d + 1) begin
            // 1. write the 32 x H actions, one byte per clock
            for (t = 0; t < VEC_H; t = t + 1)
                for (c = 0; c < N_CAND; c = c + 1) begin
                    @(negedge clk);
                    act_we = 1'b1;  act_t = t;  act_n = c;
                    act_data = vec_act[d*VEC_H + t][8*c +: 8];
                end
            // 2. set the measured state and pulse start for one clock
            @(negedge clk);
            act_we = 1'b0;
            theta0 = vec_state[d][7:0];
            omega0 = vec_state[d][15:8];
            start  = 1'b1;
            @(negedge clk);
            start  = 1'b0;
            if (!busy) bad_proto = bad_proto + 1;     // busy rises with start
            // The engine must copy theta0/omega0 when it sees start
            // (DESIGN.md Changelog, stage 2). Change them now, to the bitwise
            // inverse of the real state, so a design that keeps reading the
            // inputs during the decision imagines from the wrong state and
            // fails the cost and state checks.
            theta0 = ~vec_state[d][7:0];
            omega0 = ~vec_state[d][15:8];

            // 3. wait for done; waited ends up equal to the clocks from the
            //    edge that saw start to the edge that raised done
            waited = 0;
            while (!done && waited < WATCHDOG) begin
                if (d == 0) state_clocks[dut.state] = state_clocks[dut.state] + 1;
                @(negedge clk);
                waited = waited + 1;
            end
            if (!done) begin
                $display("decision %0d: no done within %0d clocks (state %0d)",
                         d, WATCHDOG, dut.state);
                $display("FAIL: watchdog");
                $finish;
            end
            if (busy) bad_proto = bad_proto + 1;      // busy falls with done
            @(negedge clk);
            if (done) bad_proto = bad_proto + 1;      // done lasts one clock

            // 4. compare every candidate through the dbg port
            cost_ok  = 1'b1;
            state_ok = 1'b1;
            for (c = 0; c < N_CAND; c = c + 1) begin
                dbg_n = c;
                #1;
                if (dbg_cost !== exp_cost[d][24*c +: 24]) begin
                    if (bad_cost < 3 && cost_ok)
                        $display("decision %0d lane %0d: cost %0d, golden %0d",
                                 d, c, dbg_cost, exp_cost[d][24*c +: 24]);
                    cost_ok = 1'b0;
                end
                if (dbg_theta !== exp_theta[d][8*c +: 8] ||
                    dbg_omega !== exp_omega[d][8*c +: 8]) begin
                    if (bad_state < 3 && state_ok)
                        $display("decision %0d lane %0d: (theta, omega) = (%0d, %0d), golden (%0d, %0d)",
                                 d, c, $signed(dbg_theta), $signed(dbg_omega),
                                 $signed(exp_theta[d][8*c +: 8]), $signed(exp_omega[d][8*c +: 8]));
                    state_ok = 1'b0;
                end
            end
            if (!cost_ok)  bad_cost  = bad_cost + 1;
            if (!state_ok) bad_state = bad_state + 1;

            if (best_idx !== exp_best[d][28:24] || best_cost !== exp_best[d][23:0]) begin
                if (bad_best < 3)
                    $display("decision %0d: best %0d (cost %0d), golden %0d (cost %0d)",
                             d, best_idx, best_cost, exp_best[d][31:24], exp_best[d][23:0]);
                bad_best = bad_best + 1;
            end

            // 5. clocks: the engine's own counter must agree with ours
            if (cycles != waited) begin
                if (bad_clk < 3)
                    $display("decision %0d: cycles output %0d, testbench counted %0d",
                             d, cycles, waited);
                bad_clk = bad_clk + 1;
            end
            if (cycles < min_cyc) min_cyc = cycles;
            if (cycles > max_cyc) max_cyc = cycles;

            // the measured state from the vectors (theta0/omega0 were changed above)
            if (d < 3)
                $display("decision %0d: state (%0d, %0d), best %0d, cost %0d, %0d clocks",
                         d, $signed(vec_state[d][7:0]), $signed(vec_state[d][15:8]),
                         best_idx, best_cost, cycles);
        end

        $display("clocks in each FSM state, decision 0: INIT %0d, L1 %0d, L2 %0d, COST %0d, STORE %0d, DRAIN %0d, ARG %0d, DONE %0d",
                 state_clocks[dut.S_INIT], state_clocks[dut.S_L1], state_clocks[dut.S_L2],
                 state_clocks[dut.S_COST], state_clocks[dut.S_STORE], state_clocks[dut.S_DRAIN],
                 state_clocks[dut.S_ARG], state_clocks[dut.S_DONE]);
        $display("config: LANES = %0d, H = %0d, vectors from %0s", LANES, VEC_H, `VEC_DIR);
        $display("decisions run: %0d", n_dec);
        $display("costs: %0d/%0d decisions bit-exact (all 32 candidates)", n_dec - bad_cost, n_dec);
        $display("final states: %0d/%0d decisions bit-exact (all 32 theta and omega)",
                 n_dec - bad_state, n_dec);
        $display("best_idx: %0d/%0d decisions match (best_cost too)", n_dec - bad_best, n_dec);
        $display("clocks per decision: %0d..%0d (cycles output matches the testbench count on %0d/%0d)",
                 min_cyc, max_cyc, n_dec - bad_clk, n_dec);
        $display("busy/done protocol errors: %0d", bad_proto);
        if (bad_cost == 0 && bad_state == 0 && bad_best == 0 && bad_clk == 0 && bad_proto == 0)
            $display("PASS");
        else
            $display("FAIL");
        $finish;
    end
endmodule
