`timescale 1ns/1ps
// uart_rx: receives serial bytes in the 8N1 format the USB-UART chip sends:
//
//   line idles high | start bit 0 | 8 data bits, LSB first | stop bit 1 | idle ...
//
// Every bit lasts CLKS_PER_BIT clocks. There is no clock wire, so the
// receiver finds each byte by its falling edge (idle 1 -> start bit 0) and
// then samples every bit in its middle, where the line is furthest from the
// edges on either side and so has the most margin for the two ends' clocks
// not quite agreeing.
//
//   rx       ^^^^^\_____/ d0  X d1  X ... X d7  /^^^^^^^^^^
//   samples          *       *     *         *      *
//                 start     bit 0  bit 1 ... bit 7  stop
//
// Outputs, each a one-clock pulse:
//   valid      data holds a byte whose stop bit was 1
//   frame_err  the stop bit was 0: the byte is garbage (wrong baud rate, a
//              glitch, or the line held low as a "break"); data is not changed
//
// On the board CLKS_PER_BIT = 27,000,000 / 115,200 = 234.4, rounded to 234:
// the receiver runs 0.16% fast, far inside the few-percent margin that
// mid-bit sampling leaves. Simulation uses a small value so it runs quickly.

module uart_rx #(
    parameter CLKS_PER_BIT = 234
) (
    input  wire       clk,
    input  wire       rst,
    input  wire       rx,          // the pin, asynchronous to clk
    output reg  [7:0] data,
    output reg        valid,
    output reg        frame_err
);
    localparam HALF = CLKS_PER_BIT / 2;
    // the counter holds CLKS_PER_BIT - 1 at most
    localparam CW = $clog2(CLKS_PER_BIT);

    // synthesis translate_off
    // Below 4 clocks per bit, "the middle of a bit" is not well defined and
    // the 2-clock synchronizer delay is a big part of a bit.
    initial
        if (CLKS_PER_BIT < 4) begin
            $display("uart_rx: CLKS_PER_BIT = %0d is too small (need >= 4)", CLKS_PER_BIT);
            $finish;
        end
    // synthesis translate_on

    // ------------------------------------------------ 2-flop synchronizer
    // rx changes whenever the host's clock says so, not when ours does. A
    // flip-flop that samples it just as it changes can go metastable (stuck
    // between 0 and 1 for a while). The second flip-flop gives that a whole
    // clock to settle, so the logic below only ever sees a clean 0 or 1.
    // Both start at 1, the idle level, so power-up does not look like a
    // start bit.
    reg rx_meta = 1'b1;
    reg rx_sync = 1'b1;
    always @(posedge clk) begin
        rx_meta <= rx;
        rx_sync <= rx_meta;
    end

    // ------------------------------------------------ receiver FSM
    localparam S_IDLE  = 3'd0,   // waiting for a falling edge
               S_START = 3'd1,   // half a bit to the middle of the start bit
               S_DATA  = 3'd2,   // one bit time to the middle of each data bit
               S_STOP  = 3'd3,   // one bit time to the middle of the stop bit
               S_BREAK = 3'd4;   // bad stop bit: wait until the line is idle again

    reg [2:0]    state = S_IDLE;
    reg [CW-1:0] cnt;            // clocks left until the next sample
    reg [2:0]    bitn;           // which data bit is next
    reg [7:0]    shift;          // the data bits received so far

    always @(posedge clk) begin
        valid     <= 1'b0;       // pulses: high for one clock only
        frame_err <= 1'b0;
        if (rst) begin
            state <= S_IDLE;
        end else begin
            case (state)
                S_IDLE:
                    if (!rx_sync) begin              // falling edge: a start bit begins
                        cnt   <= HALF - 1;
                        state <= S_START;
                    end
                // The synchronizer delays the falling edge by 2 clocks, but it
                // delays every later sample by the same 2 clocks, so the two
                // cancel: the check below sees the pin as it was HALF (to
                // HALF + 1) clocks after it fell, the middle of the start bit.
                S_START:
                    if (cnt != 0)
                        cnt <= cnt - 1'b1;
                    else if (!rx_sync) begin         // still low: a real start bit
                        cnt   <= CLKS_PER_BIT - 1;
                        bitn  <= 3'd0;
                        state <= S_DATA;
                    end else
                        state <= S_IDLE;             // a glitch shorter than half a bit
                S_DATA:
                    if (cnt != 0)
                        cnt <= cnt - 1'b1;
                    else begin
                        // LSB first: each new bit enters at the top and moves
                        // down, so after 8 bits d0 has reached bit 0
                        shift <= {rx_sync, shift[7:1]};
                        cnt   <= CLKS_PER_BIT - 1;
                        if (bitn == 3'd7) state <= S_STOP;
                        else              bitn  <= bitn + 1'b1;
                    end
                S_STOP:
                    if (cnt != 0)
                        cnt <= cnt - 1'b1;
                    else if (rx_sync) begin          // stop bit is 1: a good byte
                        data  <= shift;
                        valid <= 1'b1;
                        state <= S_IDLE;             // the rest of the stop bit is idle (1)
                    end else begin
                        frame_err <= 1'b1;
                        state     <= S_BREAK;
                    end
                // Going straight back to IDLE with the line still low would
                // read the same low level as another start bit, and another
                // garbage byte, and so on. Wait for the line to be 1 first.
                S_BREAK:
                    if (rx_sync) state <= S_IDLE;
                default:
                    state <= S_IDLE;
            endcase
        end
    end
endmodule
