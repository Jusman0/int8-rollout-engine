// tb_top: tests the board design (board/top.v) the way the host will use
// it: only through the UART pins, one bit at a time.
//
// A model of the host's UART drives top's uart_rx pin, and a model of the
// host's receiver decodes top's uart_tx pin into bytes. The test:
//   1. 'P' must answer 'K' LANES H 1
//   2. an unknown byte must answer 'E' 1 and light the error LED
//   3. a byte with a bad stop bit must answer 'E' 2
//   4. a 'D' command that stops halfway must answer 'E' 3 (after the timeout);
//      a bad stop bit in the middle of a 'D', followed by 60 more bytes
//      ('P', 'D', 'Z'), must answer exactly 'E' 2, and only after the host
//      has been quiet for the timeout (the rest of a broken command must not
//      be read as new commands)
//   5. 'P' again: 'K', and the error LED goes off
//   6. 'P' sent with the host's baud rate 3% slow, then 3% fast: still 'K';
//      a glitch on the rx line shorter than half a bit is ignored
//   7. holding the reset button (S1) lights its LED; the board works after
//   8. the first N recorded decisions from model/golden.py (sim/ vectors):
//      every one of the 167 reply bytes must equal the golden model's, and
//      the busy probe must be high for exactly the expected clocks
//   9. no byte arrives that was not expected
// Running checks the whole time: the host never writes an action byte into
// the engine while it is busy, and the done probe is exactly 4 clocks wide
// (stretched so a 24 MHz logic analyzer cannot miss it), once per decision.
// A watchdog turns any reply that never comes into a FAIL.
//
// Built by board/Makefile from the repo root (rom/ and sim/ are found from there):
//   iverilog -g2005 -Wall -I. -Isim -Ptb_top.LANES=8 ...
// Run: vvp tb_top_l8 +n=20     (default 20 decisions)
`timescale 1ns/1ps

module tb_top;
    parameter LANES = 8;
    `include "vectors.vh"            // VEC_COUNT, VEC_H: written by model/golden.py

    localparam N_CAND    = 32;
    localparam CPB       = 8;        // clocks per UART bit: small, so the simulation is quick
    localparam TIMEOUT   = 2000;     // top's quiet timeout, in clocks
    localparam REPLY_LEN = 7 + 5 * N_CAND;                          // 167 bytes
    localparam EXP_CYC   = (N_CAND / LANES) * (83 * VEC_H + 2) + 34; // DESIGN.md, stage 2
    // The longest wait for one reply byte: the error replies come TIMEOUT
    // clocks after the host goes quiet, a decision takes EXP_CYC clocks, and
    // a byte takes 10 x CPB. Anything much longer means a hung design.
    localparam WATCHDOG  = TIMEOUT + EXP_CYC + 100 * CPB;

    // the vectors, as in tb/tb_rollout_engine.v
    reg [15:0]          vec_state [0:VEC_COUNT-1];          // {omega0, theta0}
    reg [8*N_CAND-1:0]  vec_act   [0:VEC_COUNT*VEC_H-1];    // line d*H + t = ACT[t]
    reg [24*N_CAND-1:0] exp_cost  [0:VEC_COUNT-1];
    reg [8*N_CAND-1:0]  exp_theta [0:VEC_COUNT-1];
    reg [8*N_CAND-1:0]  exp_omega [0:VEC_COUNT-1];
    reg [31:0]          exp_best  [0:VEC_COUNT-1];          // {best_idx[7:0], best_cost[23:0]}

    // The simulation clock period does not matter (the board runs at 27 MHz);
    // only clocks per bit and clocks per decision do.
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg        host_tx = 1'b1;       // host -> board line, idles high
    reg        btn     = 1'b0;
    wire       board_tx;             // board -> host line
    wire [5:0] led_n;
    wire       probe_busy, probe_done, probe_rx, probe_tx;

    top #(
        .LANES(LANES), .H(VEC_H), .CLKS_PER_BIT(CPB), .TIMEOUT_CLKS(TIMEOUT),
        .LED_CLKS(64), .HEART_BIT(6)
    ) dut (
        .clk(clk), .btn_rst(btn), .uart_rx(host_tx), .uart_tx(board_tx), .led_n(led_n),
        .probe_busy(probe_busy), .probe_done(probe_done),
        .probe_rx(probe_rx), .probe_tx(probe_tx)
    );

    // ------------------------------------------------------------ host UART: send
    // Drives one 8N1 byte onto the board's rx pin. The line changes on the
    // falling clock edge, half a clock away from the rising edge that the
    // board samples on, so there is no race between the two.
    // stop = 0 sends a bad stop bit (a framing error).
    task send_byte;
        input [7:0] b;
        input       stop;
        integer     k;
        begin
            @(negedge clk);
            host_tx = 1'b0;                          // start bit
            repeat (CPB) @(negedge clk);
            for (k = 0; k < 8; k = k + 1) begin
                host_tx = b[k];                      // data, LSB first
                repeat (CPB) @(negedge clk);
            end
            host_tx = stop;                          // stop bit
            repeat (CPB) @(negedge clk);
            host_tx = 1'b1;                          // idle
        end
    endtask

    // The same byte with a bit time in nanoseconds instead of clocks, so the
    // host's baud rate can be a few percent off the board's, and its edges
    // land anywhere between the board's clock edges, as on real hardware.
    // A receiver that samples near the middle of each bit reads it
    // correctly at +-3%; one that samples near a bit's edge does not.
    task send_byte_at;
        input [7:0] b;
        input real  bit_ns;
        integer     k;
        begin
            #(bit_ns / 3.0);
            host_tx = 1'b0;                          // start bit
            #(bit_ns);
            for (k = 0; k < 8; k = k + 1) begin
                host_tx = b[k];
                #(bit_ns);
            end
            host_tx = 1'b1;                          // stop bit, then idle
            #(bit_ns);
        end
    endtask

    // ------------------------------------------------------------ host UART: receive
    // Runs on its own, all the time: waits for a start bit on the board's tx
    // pin, samples each bit in its middle (on falling clock edges, when the
    // line is stable) and stores the byte. Running separately means a reply
    // that starts while the host is still sending is not missed.
    reg [7:0] rx_buf [0:1023];       // received bytes, a ring
    integer   n_rx       = 0;        // bytes received so far
    integer   rx_ferrs   = 0;        // bytes from the board with a bad stop bit
    integer   rx_glitch  = 0;        // falling edges that were not start bits
    integer   rd         = 0;        // bytes checked so far
    integer   clk_count  = 0;        // clocks since time 0, for timing checks
    integer   rx_t [0:1023];         // clock at which each received byte's start bit began
    always @(posedge clk) clk_count = clk_count + 1;

    always begin : host_receiver
        integer   k, t_start;
        reg [7:0] b;
        @(negedge board_tx);
        t_start = clk_count;
        repeat (CPB / 2) @(negedge clk);             // middle of the start bit
        if (board_tx !== 1'b0) begin
            rx_glitch = rx_glitch + 1;
        end else begin
            for (k = 0; k < 8; k = k + 1) begin
                repeat (CPB) @(negedge clk);
                b[k] = board_tx;
            end
            repeat (CPB) @(negedge clk);             // middle of the stop bit
            if (board_tx !== 1'b1) rx_ferrs = rx_ferrs + 1;
            rx_buf[n_rx % 1024] = b;
            rx_t[n_rx % 1024]   = t_start;
            n_rx = n_rx + 1;
        end
    end

    // ------------------------------------------------------------ busy probe
    // Measures every high pulse on the busy probe pin, in clocks: this is the
    // pulse a logic analyzer on the board will show.
    integer busy_len = 0, last_busy_len = 0, busy_pulses = 0;
    always @(posedge clk) begin
        if (probe_busy) busy_len = busy_len + 1;
        else if (busy_len != 0) begin
            last_busy_len = busy_len;
            busy_pulses   = busy_pulses + 1;
            busy_len      = 0;
        end
    end

    // ------------------------------------------------------------ done probe
    // probe_done is stretched to 4 clocks so the logic analyzer (24 MHz, one
    // sample per 41.7 ns) cannot miss it: every pulse must be exactly 4 long.
    integer done_len = 0, done_pulses = 0, bad_done = 0;
    always @(posedge clk) begin
        if (probe_done) done_len = done_len + 1;
        else if (done_len != 0) begin
            done_pulses = done_pulses + 1;
            if (done_len != 4) bad_done = bad_done + 1;
            done_len = 0;
        end
    end

    // top.v writes ACT only while the engine is idle (its comments rely on
    // it). The engine prints an ERROR for a write while busy; count it here so
    // it fails the test instead of scrolling past.
    integer act_while_busy = 0;
    always @(posedge clk)
        if (dut.u_eng.act_we && dut.u_eng.busy) act_while_busy = act_while_busy + 1;

    // ------------------------------------------------------------ checking
    integer bad_bytes = 0;           // reply bytes that differ from the expected
    integer shown     = 0;           // mismatches printed so far

    // Waits for the next received byte (watchdog: FAIL if none comes) and
    // compares it with want. what/idx say where it is, for the message.
    task expect_byte;
        input [7:0]    want;
        input [8*20:1] what;
        input integer  idx;
        integer        waited;
        begin
            waited = 0;
            while (n_rx <= rd && waited < WATCHDOG) begin
                @(negedge clk);
                waited = waited + 1;
            end
            if (n_rx <= rd) begin
                $display("no reply byte within %0d clocks (waiting for %0s byte %0d)",
                         WATCHDOG, what, idx);
                $display("FAIL: watchdog");
                $finish;
            end
            if (rx_buf[rd % 1024] !== want) begin
                bad_bytes = bad_bytes + 1;
                if (shown < 5)
                    $display("%0s byte %0d: got 0x%02h, expected 0x%02h",
                             what, idx, rx_buf[rd % 1024], want);
                shown = shown + 1;
            end
            rd = rd + 1;
        end
    endtask

    integer n_dec, d, t, c, k, errors, bad_dec, bad_busy, bad_mark, quiet_from, e_at;
    reg [7:0] reply [0:REPLY_LEN-1];

    initial begin
        if (!$value$plusargs("n=%d", n_dec)) n_dec = 20;
        if (n_dec < 1 || n_dec > VEC_COUNT) begin
            $display("FAIL: +n=%0d, but sim/ holds %0d decisions (allowed: 1..%0d)",
                     n_dec, VEC_COUNT, VEC_COUNT);
            $finish;
        end
        $readmemh("sim/vec_state.hex", vec_state);
        $readmemh("sim/vec_act.hex",   vec_act);
        $readmemh("sim/exp_cost.hex",  exp_cost);
        $readmemh("sim/exp_theta.hex", exp_theta);
        $readmemh("sim/exp_omega.hex", exp_omega);
        $readmemh("sim/exp_best.hex",  exp_best);
        errors   = 0;
        bad_dec  = 0;
        bad_busy = 0;

        repeat (40) @(negedge clk);                  // power-on reset is 16 clocks

        // 1. ping
        send_byte(8'h50, 1'b1);
        expect_byte(8'h4B, "ping", 0);
        expect_byte(LANES, "ping", 1);
        expect_byte(VEC_H, "ping", 2);
        expect_byte(8'd1,  "ping", 3);
        $display("ping: K %0d %0d 1 expected, %0s", LANES, VEC_H,
                 bad_bytes == 0 ? "matches" : "MISMATCH");
        bad_mark = bad_bytes;

        // 2. unknown command byte
        send_byte(8'h5A, 1'b1);                      // 'Z'
        expect_byte(8'h45, "unknown", 0);
        expect_byte(8'd1,  "unknown", 1);
        @(negedge clk);
        if (led_n[2] !== 1'b0) begin
            $display("error LED is not on after an error");
            errors = errors + 1;
        end

        // 3. framing error: a 'P' whose stop bit is 0 must not be taken as 'P'
        send_byte(8'h50, 1'b0);
        expect_byte(8'h45, "frame", 0);
        expect_byte(8'd2,  "frame", 1);

        // 4. a 'D' that stops after the state and 5 of the 32 x H actions
        send_byte(8'h44, 1'b1);
        send_byte(8'h01, 1'b1);
        send_byte(8'h02, 1'b1);
        for (k = 0; k < 5; k = k + 1) send_byte(k, 1'b1);
        expect_byte(8'h45, "timeout", 0);
        expect_byte(8'd3,  "timeout", 1);
        $display("errors: unknown byte -> E 1, bad stop bit -> E 2, half a 'D' -> E 3: %0s",
                 bad_bytes == bad_mark ? "all as expected" : "MISMATCH");

        //    a bad stop bit in the middle of a 'D', then 60 more bytes that
        //    look like commands: the board must treat all of it as one broken
        //    command, wait until the host has been quiet for TIMEOUT clocks,
        //    and answer exactly 'E' 2 (anything extra fails step 9)
        bad_mark = bad_bytes;
        send_byte(8'h44, 1'b1);
        send_byte(8'h05, 1'b1);
        send_byte(8'h06, 1'b1);
        for (k = 0; k < 3; k = k + 1) send_byte(k, 1'b1);
        send_byte(8'h33, 1'b0);                      // the bad stop bit
        for (k = 0; k < 60; k = k + 1)
            send_byte(k % 3 == 0 ? 8'h50 : k % 3 == 1 ? 8'h44 : 8'h5A, 1'b1);
        quiet_from = clk_count;                      // the host goes quiet here
        e_at = rd;                                   // index of the 'E' about to arrive
        expect_byte(8'h45, "drain", 0);
        expect_byte(8'd2,  "drain", 1);
        if (rx_t[e_at % 1024] - quiet_from < TIMEOUT) begin
            $display("error reply came %0d clocks after the host went quiet, before the %0d-clock timeout",
                     rx_t[e_at % 1024] - quiet_from, TIMEOUT);
            errors = errors + 1;
        end
        $display("bad stop bit inside a 'D', then 60 command-like bytes -> one E 2 after %0d quiet clocks: %0s",
                 rx_t[e_at % 1024] - quiet_from,
                 bad_bytes == bad_mark ? "as expected" : "MISMATCH");

        // 5. a good command clears the error LED
        send_byte(8'h50, 1'b1);
        expect_byte(8'h4B, "ping", 0);
        expect_byte(LANES, "ping", 1);
        expect_byte(VEC_H, "ping", 2);
        expect_byte(8'd1,  "ping", 3);
        repeat (2) @(negedge clk);
        if (led_n[2] !== 1'b1) begin
            $display("error LED is still on after a good command");
            errors = errors + 1;
        end
        if (bad_bytes != 0) errors = errors + 1;

        // 6. the host's baud rate 3% slow, then 3% fast (bit times of
        //    1.03 and 0.97 x CPB clocks of 10 ns): 'P' must still be read
        bad_mark = bad_bytes;
        send_byte_at(8'h50, 1.03 * CPB * 10.0);
        expect_byte(8'h4B, "slow ping", 0);
        expect_byte(LANES, "slow ping", 1);
        expect_byte(VEC_H, "slow ping", 2);
        expect_byte(8'd1,  "slow ping", 3);
        send_byte_at(8'h50, 0.97 * CPB * 10.0);
        expect_byte(8'h4B, "fast ping", 0);
        expect_byte(LANES, "fast ping", 1);
        expect_byte(VEC_H, "fast ping", 2);
        expect_byte(8'd1,  "fast ping", 3);
        $display("host baud rate 3%% slow and 3%% fast: %0s",
                 bad_bytes == bad_mark ? "both read correctly" : "MISMATCH");

        //    a 2-clock low glitch (shorter than half a bit) is not a start
        //    bit: the next 'P' must get exactly one 'K' and nothing else
        bad_mark = bad_bytes;
        @(negedge clk);
        host_tx = 1'b0;
        repeat (2) @(negedge clk);
        host_tx = 1'b1;
        repeat (20 * CPB) @(negedge clk);
        send_byte(8'h50, 1'b1);
        expect_byte(8'h4B, "ping after glitch", 0);
        expect_byte(LANES, "ping after glitch", 1);
        expect_byte(VEC_H, "ping after glitch", 2);
        expect_byte(8'd1,  "ping after glitch", 3);
        $display("a 2-clock glitch on the rx line: %0s",
                 bad_bytes == bad_mark ? "ignored" : "MISMATCH");

        // 7. the reset button (S1): led_n[5] lights while it is held, and
        //    the board answers normally afterwards
        @(negedge clk);
        btn = 1'b1;
        repeat (10) @(negedge clk);
        if (led_n[5] !== 1'b0) begin
            $display("reset LED is not on while S1 is held");
            errors = errors + 1;
        end
        btn = 1'b0;
        repeat (10) @(negedge clk);
        if (led_n[5] !== 1'b1) begin
            $display("reset LED is still on after S1 is released");
            errors = errors + 1;
        end

        // 8. recorded decisions
        for (d = 0; d < n_dec; d = d + 1) begin
            send_byte(8'h44, 1'b1);
            send_byte(vec_state[d][7:0], 1'b1);      // theta0
            send_byte(vec_state[d][15:8], 1'b1);     // omega0
            for (t = 0; t < VEC_H; t = t + 1)
                for (c = 0; c < N_CAND; c = c + 1)
                    send_byte(vec_act[d*VEC_H + t][8*c +: 8], 1'b1);

            // the reply the golden model predicts
            reply[0] = 8'h52;
            reply[1] = exp_best[d][31:24];
            reply[2] = exp_best[d][7:0];
            reply[3] = exp_best[d][15:8];
            reply[4] = exp_best[d][23:16];
            reply[5] = EXP_CYC % 256;
            reply[6] = EXP_CYC / 256;
            for (c = 0; c < N_CAND; c = c + 1) begin
                reply[7 + 5*c]     = exp_cost[d][24*c      +: 8];
                reply[7 + 5*c + 1] = exp_cost[d][24*c + 8  +: 8];
                reply[7 + 5*c + 2] = exp_cost[d][24*c + 16 +: 8];
                reply[7 + 5*c + 3] = exp_theta[d][8*c +: 8];
                reply[7 + 5*c + 4] = exp_omega[d][8*c +: 8];
            end
            bad_mark = bad_bytes;
            for (k = 0; k < REPLY_LEN; k = k + 1)
                expect_byte(reply[k], "decision", k);
            if (bad_bytes != bad_mark) begin
                bad_dec = bad_dec + 1;
                if (bad_dec <= 3) $display("decision %0d: %0d reply bytes differ",
                                           d, bad_bytes - bad_mark);
            end
            if (last_busy_len != EXP_CYC) begin
                bad_busy = bad_busy + 1;
                if (bad_busy <= 3) $display("decision %0d: busy probe high %0d clocks, expected %0d",
                                            d, last_busy_len, EXP_CYC);
            end
            if (d < 3)
                $display("decision %0d: state (%0d, %0d), best %0d, cost %0d, busy probe %0d clocks",
                         d, $signed(vec_state[d][7:0]), $signed(vec_state[d][15:8]),
                         exp_best[d][31:24], exp_best[d][23:0], last_busy_len);
        end

        // 9. nothing extra may arrive
        repeat (20 * CPB) @(negedge clk);

        $display("config: LANES = %0d, H = %0d, %0d clocks per UART bit, vectors from sim",
                 LANES, VEC_H, CPB);
        $display("decisions sent over the UART: %0d", n_dec);
        $display("decisions with all %0d reply bytes equal to the golden model: %0d/%0d",
                 REPLY_LEN, n_dec - bad_dec, n_dec);
        $display("busy probe high for exactly %0d clocks: %0d/%0d decisions",
                 EXP_CYC, n_dec - bad_busy, n_dec);
        $display("done probe pulses: %0d (expected %0d), not exactly 4 clocks wide: %0d",
                 done_pulses, n_dec, bad_done);
        $display("action bytes written while the engine was busy: %0d", act_while_busy);
        $display("bytes received: %0d, checked: %0d; bad stop bits from the board: %0d; glitches: %0d",
                 n_rx, rd, rx_ferrs, rx_glitch);
        if (n_rx != rd) errors = errors + 1;
        if (errors == 0 && bad_dec == 0 && bad_busy == 0 && bad_bytes == 0 &&
            rx_ferrs == 0 && rx_glitch == 0 && busy_pulses == n_dec &&
            done_pulses == n_dec && bad_done == 0 && act_while_busy == 0)
            $display("PASS");
        else
            $display("FAIL");
        $finish;
    end
endmodule
