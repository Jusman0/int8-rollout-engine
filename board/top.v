`timescale 1ns/1ps
// top: the Tang Nano 20K board design. It wraps rollout_engine (unchanged)
// in a small command interpreter that the host talks to over the board's
// USB-UART at 115,200 baud, 8N1. The protocol is specified byte by byte in
// board/README.md; in short:
//
//   host -> board                          board -> host
//   'P'                                    'K' LANES H VERSION
//   'D' theta0 omega0 ACT[0][0..31] ...    'R' best_idx best_cost(3) cycles(2)
//       ... ACT[H-1][0..31]                    then for n = 0..31: cost[n](3) theta[n] omega[n]
//   anything else                          'E' code
//
// Multi-byte numbers are little-endian (lowest byte first). Each action byte
// is written into the engine through its act_* port as soon as it arrives,
// the same order and the same one-byte-per-write as tb/tb_rollout_engine.v.
//
// Error codes (the 'E' reply):
//   1  unknown command byte
//   2  framing error: a byte's stop bit was 0 (wrong baud rate or a glitch)
//   3  timeout: a 'D' command stopped arriving halfway (a byte was lost)
// After an error the board waits until the host has been quiet for
// TIMEOUT_CLKS (so the rest of a broken command is thrown away, not read as
// new commands), then answers 'E' code. The error LED stays on until the
// next good 'P' or 'D'.
//
// Everything runs on the 27 MHz oscillator clock (U3, pin 4), no PLL.
//
// LEDs (active low: the LED's anode is on 3.3 V through 510 ohms, so the pin
// lights it by pulling low; see board/README.md):
//   led_n[0] heartbeat, about 0.8 Hz: the bitstream is loaded and the clock runs
//   led_n[1] engine busy, stretched to LED_CLKS so a 186 us decision is visible
//   led_n[2] error (sticky until the next good command)
//   led_n[3] a byte was received (stretched)
//   led_n[4] a byte was sent (stretched)
//   led_n[5] reset is held (the S1 button)
// Probe pins for a logic analyzer (free header pins, see board/README.md):
//   probe_busy  engine busy: high for exactly cycles clocks per decision
//   probe_done  engine done, stretched from 1 to 4 clocks (148 ns) so a
//               24 MHz analyzer (41.7 ns per sample) cannot miss it
//   probe_rx    the UART line from the host
//   probe_tx    the UART line to the host
// Each probe comes from a flip-flop, one clock (37 ns) after the signal it
// copies; both edges move together, so pulse widths are unchanged.

module top #(
    parameter LANES        = 8,          // 8, 16 or 32 (rollout_engine checks this)
    parameter H            = 15,         // imagined steps per plan
    parameter CLKS_PER_BIT = 234,        // 27 MHz / 115,200 baud = 234.4
    parameter TIMEOUT_CLKS = 2700000,    // 100 ms at 27 MHz
    parameter LED_CLKS     = 1350000,    // 50 ms: how long a stretched LED stays on
    parameter HEART_BIT    = 24          // heartbeat = this bit of a free-running counter
) (
    input  wire       clk,               // 27 MHz oscillator, pin 4
    input  wire       btn_rst,           // S1, pin 88: 1 while pressed
    input  wire       uart_rx,           // pin 70: from the BL616 USB-UART
    output wire       uart_tx,           // pin 69: to the BL616 USB-UART
    output reg  [5:0] led_n,             // pins 15-20, 0 = LED on
    output reg        probe_busy,
    output reg        probe_done,
    output reg        probe_rx,
    output reg        probe_tx
);
    localparam N_CAND = 32;

    // protocol bytes
    localparam [7:0] CMD_PING   = 8'h50,  // 'P'
                     CMD_DECIDE = 8'h44,  // 'D'
                     REP_PING   = 8'h4B,  // 'K'
                     REP_RESULT = 8'h52,  // 'R'
                     REP_ERROR  = 8'h45;  // 'E'
    localparam [7:0] ERR_UNKNOWN = 8'd1, ERR_FRAME = 8'd2, ERR_TIMEOUT = 8'd3;
    localparam [7:0] PROTOCOL_VERSION = 8'd1;
    localparam [7:0] LANES_BYTE = LANES, H_BYTE = H;

    // ------------------------------------------------------------ reset
    // Power-on reset: after configuration every flip-flop starts at its
    // initial value, so por_cnt starts at 0 and the design is held in reset
    // until it reaches 15 (16 clocks). The S1 button also resets everything;
    // it is asynchronous to clk, so it goes through two flip-flops first,
    // like the UART input.
    reg [3:0] por_cnt  = 4'd0;
    reg       btn_meta = 1'b0, btn_sync = 1'b0;
    reg       rst      = 1'b1;
    always @(posedge clk) begin
        if (por_cnt != 4'hf) por_cnt <= por_cnt + 4'd1;
        btn_meta <= btn_rst;
        btn_sync <= btn_meta;
        rst      <= (por_cnt != 4'hf) || btn_sync;
    end

    // ------------------------------------------------------------ UART
    wire [7:0] rx_data;
    wire       rx_valid, rx_ferr;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk(clk), .rst(rst), .rx(uart_rx),
        .data(rx_data), .valid(rx_valid), .frame_err(rx_ferr)
    );

    reg  [7:0] tx_byte;                  // the reply byte to send next (mux below)
    wire       tx_send;
    wire       tx_busy;
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk(clk), .rst(rst), .data(tx_byte), .send(tx_send),
        .tx(uart_tx), .busy(tx_busy)
    );

    // ------------------------------------------------------------ engine
    reg               act_we = 1'b0;
    reg  [3:0]        act_t;
    reg  [4:0]        act_n;
    reg  [7:0]        act_data;
    reg  [7:0]        theta0, omega0;
    reg               start = 1'b0;
    wire              busy, done;
    wire [4:0]        best_idx;
    wire [23:0]       best_cost, dbg_cost;
    wire [15:0]       cycles;
    wire [7:0]        dbg_theta, dbg_omega;
    reg  [4:0]        cn;                // candidate being sent back; drives dbg_n

    rollout_engine #(.LANES(LANES), .H(H)) u_eng (
        .clk(clk), .rst(rst),
        .act_we(act_we), .act_t(act_t), .act_n(act_n), .act_data(act_data),
        .theta0(theta0), .omega0(omega0),
        .start(start), .busy(busy), .done(done),
        .best_idx(best_idx), .best_cost(best_cost), .cycles(cycles),
        .dbg_n(cn), .dbg_cost(dbg_cost), .dbg_theta(dbg_theta), .dbg_omega(dbg_omega)
    );

    // ------------------------------------------------------------ quiet timer
    // Clocks since the last byte (good or bad) arrived, stopping at
    // TIMEOUT_CLKS. A 'D' command whose next byte takes this long has lost a
    // byte; after an error, this much quiet means the host has stopped
    // sending the rest of the broken command.
    localparam QW = $clog2(TIMEOUT_CLKS + 1);
    reg [QW-1:0] quiet = {QW{1'b0}};
    wire         timed_out = (quiet == TIMEOUT_CLKS);
    always @(posedge clk)
        if (rst || rx_valid || rx_ferr) quiet <= {QW{1'b0}};
        else if (!timed_out)            quiet <= quiet + 1'b1;

    // ------------------------------------------------------------ command FSM
    localparam S_IDLE  = 3'd0,   // wait for a command byte
               S_THETA = 3'd1,   // 'D': next byte is theta0
               S_OMEGA = 3'd2,   //      then omega0
               S_ACT   = 3'd3,   //      then 32 x H action bytes
               S_START = 3'd4,   // pulse the engine's start
               S_RUN   = 3'd5,   // wait for done
               S_REPLY = 3'd6,   // send the reply, one byte per tx_busy low
               S_DRAIN = 3'd7;   // error: wait for quiet, then reply 'E'

    // which reply S_REPLY is sending
    localparam K_PING = 2'd0, K_RESULT = 2'd1, K_ERROR = 2'd2;

    reg [2:0] state = S_IDLE;
    reg [3:0] t;                 // S_ACT: step of the next action byte
    reg [4:0] n;                 // S_ACT: candidate of the next action byte
    reg [1:0] kind;              // reply kind
    reg [2:0] hidx;              // reply header byte index
    reg       cand;              // 0: sending the header, 1: the 32 candidates
    reg [2:0] cf;                // candidate field: 0..2 cost bytes, 3 theta, 4 omega
    reg [7:0] err_code;
    reg       err_flag = 1'b0;   // drives the error LED

    // Send whenever the transmitter is free. tx_byte is chosen from
    // (kind, hidx, cand, cn, cf), so the same clock edge that hands a byte
    // to uart_tx also moves those counters on to the next byte.
    assign tx_send = (state == S_REPLY) && !tx_busy;

    // the last header byte of each reply kind: 'K' + 3, 'E' + 1, 'R' + 6
    wire last_hdr = (kind == K_PING   && hidx == 3'd3) ||
                    (kind == K_ERROR  && hidx == 3'd1) ||
                    (kind == K_RESULT && hidx == 3'd6);

    always @* begin
        tx_byte = 8'h00;
        if (!cand) begin
            case (kind)
                K_PING:
                    case (hidx)
                        3'd0:    tx_byte = REP_PING;
                        3'd1:    tx_byte = LANES_BYTE;
                        3'd2:    tx_byte = H_BYTE;
                        default: tx_byte = PROTOCOL_VERSION;
                    endcase
                K_ERROR:
                    tx_byte = (hidx == 3'd0) ? REP_ERROR : err_code;
                default:                                   // K_RESULT
                    case (hidx)
                        3'd0:    tx_byte = REP_RESULT;
                        3'd1:    tx_byte = {3'b000, best_idx};
                        3'd2:    tx_byte = best_cost[7:0];
                        3'd3:    tx_byte = best_cost[15:8];
                        3'd4:    tx_byte = best_cost[23:16];
                        3'd5:    tx_byte = cycles[7:0];
                        default: tx_byte = cycles[15:8];
                    endcase
            endcase
        end else begin
            // candidate cn, read through the engine's debug port (dbg_n = cn)
            case (cf)
                3'd0:    tx_byte = dbg_cost[7:0];
                3'd1:    tx_byte = dbg_cost[15:8];
                3'd2:    tx_byte = dbg_cost[23:16];
                3'd3:    tx_byte = dbg_theta;
                default: tx_byte = dbg_omega;
            endcase
        end
    end

    always @(posedge clk) begin
        act_we <= 1'b0;                  // one-clock pulses by default
        start  <= 1'b0;
        if (rst) begin
            state    <= S_IDLE;
            err_flag <= 1'b0;
        end else begin
            case (state)
                S_IDLE:
                    if (rx_ferr) begin
                        err_code <= ERR_FRAME;
                        state    <= S_DRAIN;
                    end else if (rx_valid) begin
                        if (rx_data == CMD_PING) begin
                            kind  <= K_PING;
                            hidx  <= 3'd0;
                            cand  <= 1'b0;
                            state <= S_REPLY;
                        end else if (rx_data == CMD_DECIDE) begin
                            state <= S_THETA;
                        end else begin
                            err_code <= ERR_UNKNOWN;
                            state    <= S_DRAIN;
                        end
                    end
                S_THETA, S_OMEGA, S_ACT:
                    if (rx_ferr) begin
                        err_code <= ERR_FRAME;
                        state    <= S_DRAIN;
                    end else if (rx_valid) begin
                        if (state == S_THETA) begin
                            theta0 <= rx_data;
                            state  <= S_OMEGA;
                        end else if (state == S_OMEGA) begin
                            omega0 <= rx_data;
                            t      <= 4'd0;
                            n      <= 5'd0;
                            state  <= S_ACT;
                        end else begin
                            // ACT[t][n]: step outer, candidate inner, like the
                            // testbench and the co-simulation
                            act_we   <= 1'b1;
                            act_t    <= t;
                            act_n    <= n;
                            act_data <= rx_data;
                            n        <= n + 5'd1;          // wraps 31 -> 0
                            if (n == N_CAND - 1) begin
                                if (t == H - 1) state <= S_START;
                                else            t     <= t + 4'd1;
                            end
                        end
                    end else if (timed_out) begin
                        err_code <= ERR_TIMEOUT;
                        state    <= S_DRAIN;
                    end
                // The last act write lands at this clock edge, so start is
                // seen one clock after it, while busy is still low: the
                // engine's rule "write ACT only while busy is low" holds.
                S_START: begin
                    start <= 1'b1;
                    state <= S_RUN;
                end
                S_RUN:
                    if (done) begin              // results valid from done until the next start
                        kind  <= K_RESULT;
                        hidx  <= 3'd0;
                        cand  <= 1'b0;
                        state <= S_REPLY;
                    end
                S_REPLY:
                    // Bytes from the host are ignored here and in S_START /
                    // S_RUN: the host waits for the whole reply first.
                    if (tx_send) begin
                        if (!cand) begin
                            if (!last_hdr)
                                hidx <= hidx + 3'd1;
                            else if (kind == K_RESULT) begin
                                cand <= 1'b1;    // header done, now the candidates
                                cn   <= 5'd0;
                                cf   <= 3'd0;
                            end else begin
                                if (kind == K_PING) err_flag <= 1'b0;
                                state <= S_IDLE;
                            end
                        end else if (cf != 3'd4) begin
                            cf <= cf + 3'd1;
                        end else begin
                            cf <= 3'd0;
                            if (cn != N_CAND - 1)
                                cn <= cn + 5'd1;
                            else begin
                                err_flag <= 1'b0;    // a good 'D' clears the error LED
                                state    <= S_IDLE;
                            end
                        end
                    end
                S_DRAIN: begin
                    err_flag <= 1'b1;
                    // a framing error while draining is just more of the
                    // broken command; any byte restarts the quiet timer
                    if (timed_out) begin
                        kind  <= K_ERROR;
                        hidx  <= 3'd0;
                        cand  <= 1'b0;
                        state <= S_REPLY;
                    end
                end
                default:
                    state <= S_IDLE;
            endcase
        end
    end

    // ------------------------------------------------------------ LEDs
    reg [HEART_BIT:0] heart = {(HEART_BIT + 1){1'b0}};
    always @(posedge clk) heart <= heart + 1'b1;

    wire busy_led, rx_led, tx_led;
    led_stretch #(.CLKS(LED_CLKS)) u_busy_led (.clk(clk), .trigger(busy),     .on(busy_led));
    led_stretch #(.CLKS(LED_CLKS)) u_rx_led   (.clk(clk), .trigger(rx_valid), .on(rx_led));
    led_stretch #(.CLKS(LED_CLKS)) u_tx_led   (.clk(clk), .trigger(tx_send),  .on(tx_led));

    // registered, and inverted because 0 lights an LED
    always @(posedge clk)
        led_n <= ~{rst, tx_led, rx_led, err_flag, busy_led, heart[HEART_BIT]};

    // ------------------------------------------------------------ probes
    reg [1:0] done_hold = 2'd0;          // keeps probe_done up 3 clocks after done
    always @(posedge clk) begin
        if (done)                done_hold <= 2'd3;
        else if (done_hold != 0) done_hold <= done_hold - 2'd1;
        probe_busy <= busy;
        probe_done <= done || (done_hold != 0);
        probe_rx   <= uart_rx;
        probe_tx   <= uart_tx;
    end
endmodule

// led_stretch: an LED that lights for CLKS clocks after every trigger, so
// that events lasting microseconds are visible to a human. A new trigger
// restarts the count.
module led_stretch #(
    parameter CLKS = 1350000
) (
    input  wire clk,
    input  wire trigger,
    output wire on
);
    localparam W = $clog2(CLKS);
    reg [W-1:0] cnt = {W{1'b0}};
    always @(posedge clk)
        if (trigger)       cnt <= CLKS - 1;
        else if (cnt != 0) cnt <= cnt - 1'b1;
    assign on = (cnt != 0);
endmodule
