// tb_cosim: the hardware half of the closed-loop co-simulation (DESIGN.md
// §7.3). cosim/host.py runs the pendulum and the host side of the planner;
// this testbench runs rollout_engine and answers one question per control
// tick: "which of these 32 plans is cheapest?" The answer it sends back
// (best_idx) is the plan whose first action the host applies to the
// pendulum, so the RTL is in the control loop.
//
// The two programs talk through two named pipes (FIFOs) that cosim/run.sh
// makes with mkfifo. A named pipe looks like a file, but what one program
// writes into it the other reads out, in order, and a read waits until
// there is something to read. That waiting keeps the two programs in step:
// the simulation stops inside $fscanf until the host sends the next tick.
//
//   from_rtl, once at the start:  LANES H
//       so the host can check it builds plans of the right length
//   to_rtl, one line per tick:    1 theta_q omega_q u u u ... (32 x H of them)
//       command 1, the measured state, then the actions in ACT[t][n] order:
//       step t = 0 for candidates n = 0..31, then step t = 1, and so on.
//       All numbers are signed decimal.
//   from_rtl, one line per tick:  best_idx best_cost cycles
//   to_rtl, at the end:           0   ("no more ticks")
//
// No $fscanf format string here ends in whitespace ("%d", never "%d\n").
// In a scanf format, whitespace means "skip all whitespace up to the next
// non-space character", so a trailing "\n" would wait for the first
// character of the *next* tick, which the host only sends after it gets
// this tick's answer: both programs would wait for each other forever.
// Leaving the newline unread is harmless, because "%d" skips leading
// whitespace anyway.
//
// Built by cosim/run.sh with
//   -DCOSIM_DIR='"sim/cosim"'     where the two pipes are
//   -Ptb_cosim.LANES=8            lanes in the engine (default 32)
`timescale 1ns/1ps

`ifndef COSIM_DIR
`define COSIM_DIR "sim/cosim"
`endif

module tb_cosim;
    parameter LANES = 32;
    parameter H     = 15;

    localparam N_CAND   = 32;
    // the same watchdog as tb_rollout_engine: no decision needs more than
    // 5,354 clocks, so 10,000 clocks without done means a hung FSM
    localparam WATCHDOG = 10000;

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

    rollout_engine #(.LANES(LANES), .H(H)) dut (
        .clk(clk), .rst(rst),
        .act_we(act_we), .act_t(act_t), .act_n(act_n), .act_data(act_data),
        .theta0(theta0), .omega0(omega0),
        .start(start), .busy(busy), .done(done),
        .best_idx(best_idx), .best_cost(best_cost), .cycles(cycles),
        .dbg_n(dbg_n), .dbg_cost(dbg_cost), .dbg_theta(dbg_theta), .dbg_omega(dbg_omega)
    );

    integer to_rtl, from_rtl;      // file handles of the two pipes
    integer got;                   // how many numbers $fscanf managed to read
    integer cmd, th, om, u;        // numbers from the host
    integer t, c, waited, ticks;
    reg     running;

    // Stop the simulation if the host's input is not what we expect, instead
    // of carrying on with garbage. host.py notices that from_rtl closed.
    task check_got;
        input integer want;
        begin
            if (got != want) begin
                // got = -1 means end of file: the host closed its end of the pipe
                $display("FAIL: tick %0d: expected %0d number(s) from the host, got %0d",
                         ticks, want, got);
                $finish;
            end
        end
    endtask

    // One control tick: read the state and the 32 x H actions, run the
    // engine, send back its answer.
    task do_tick;
        begin
            got = $fscanf(to_rtl, "%d %d", th, om);
            check_got(2);

            // write the actions one byte per clock through the host port,
            // exactly as tb_rollout_engine does. u[7:0] is the low byte of
            // the integer, which is the int8 two's complement value
            // (-5 -> 8'hfb).
            for (t = 0; t < H; t = t + 1)
                for (c = 0; c < N_CAND; c = c + 1) begin
                    got = $fscanf(to_rtl, "%d", u);
                    check_got(1);
                    @(negedge clk);
                    act_we   = 1'b1;
                    act_t    = t;
                    act_n    = c;
                    act_data = u[7:0];
                end

            // set the measured state and pulse start for one clock
            @(negedge clk);
            act_we = 1'b0;
            theta0 = th[7:0];
            omega0 = om[7:0];
            start  = 1'b1;
            @(negedge clk);
            start  = 1'b0;

            // wait for done (a one-clock pulse; we sample at negedges)
            waited = 0;
            while (!done && waited < WATCHDOG) begin
                @(negedge clk);
                waited = waited + 1;
            end
            if (!done) begin
                $display("tick %0d: no done within %0d clocks (state %0d)",
                         ticks, WATCHDOG, dut.state);
                $display("FAIL: watchdog");
                $finish;
            end

            // the answer; $fflush pushes it into the pipe now instead of
            // leaving it in a buffer while the host waits for it
            $fwrite(from_rtl, "%0d %0d %0d\n", best_idx, best_cost, cycles);
            $fflush(from_rtl);
            ticks = ticks + 1;
        end
    endtask

    initial begin
        // Opening a named pipe waits until the other program opens the other
        // end. host.py opens to_rtl first and from_rtl second, the same order
        // as here, so neither program waits on a pipe the other is not
        // opening yet.
        to_rtl   = $fopen({`COSIM_DIR, "/to_rtl"}, "r");
        from_rtl = $fopen({`COSIM_DIR, "/from_rtl"}, "w");
        if (to_rtl == 0 || from_rtl == 0) begin
            $display("FAIL: cannot open the pipes in %0s (start this through cosim/run.sh)",
                     `COSIM_DIR);
            $finish;
        end
        $fwrite(from_rtl, "%0d %0d\n", LANES, H);
        $fflush(from_rtl);

        repeat (2) @(negedge clk);
        rst = 1'b0;

        ticks   = 0;
        running = 1'b1;
        while (running) begin
            got = $fscanf(to_rtl, "%d", cmd);
            check_got(1);                 // the host stopped without sending 0
            if (cmd == 1)
                do_tick;
            else if (cmd == 0)
                running = 1'b0;
            else begin
                $display("FAIL: tick %0d: unknown command %0d from the host", ticks, cmd);
                $finish;
            end
        end

        $display("tb_cosim: answered %0d ticks (LANES = %0d, H = %0d); the host sent 0, done",
                 ticks, LANES, H);
        $fclose(to_rtl);
        $fclose(from_rtl);
        $finish;
    end
endmodule
