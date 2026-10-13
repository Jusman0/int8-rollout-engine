`timescale 1ns/1ps
// uart_tx: sends one byte in the 8N1 format (start bit 0, 8 data bits LSB
// first, stop bit 1), each bit CLKS_PER_BIT clocks long.
//
// Handshake: when busy is low, holding send high for a clock edge starts
// a byte (data is copied at that edge). busy rises at the same edge and
// falls after the stop bit has lasted a full bit time, so the caller can
// simply send the next byte whenever busy is low.
//
// tx comes straight from a flip-flop, so the pin never glitches: a glitch on
// a serial line looks like a start bit to the receiver.

module uart_tx #(
    parameter CLKS_PER_BIT = 234
) (
    input  wire       clk,
    input  wire       rst,
    input  wire [7:0] data,
    input  wire       send,
    output wire       tx,
    output reg        busy
);
    localparam CW = $clog2(CLKS_PER_BIT);

    // Starts at 1 (idle). Without the "= 1" the flip-flop would power up at
    // 0 and the line would sit low, a start bit, until the reset arrives.
    reg tx_r = 1'b1;
    assign tx = tx_r;

    reg [8:0]    frame;          // bits still to send, next one in bit 0: {stop, d7..d0}
    reg [3:0]    left;           // how many of them are left
    reg [CW-1:0] cnt;            // clocks left in the bit now on the line

    always @(posedge clk) begin
        if (rst) begin
            tx_r <= 1'b1;
            busy <= 1'b0;
        end else if (!busy) begin
            if (send) begin
                tx_r  <= 1'b0;               // the start bit goes out now
                frame <= {1'b1, data};       // then d0..d7, then the stop bit
                left  <= 4'd9;
                cnt   <= CLKS_PER_BIT - 1;
                busy  <= 1'b1;
            end
        end else if (cnt != 0) begin
            cnt <= cnt - 1'b1;               // hold the current bit
        end else if (left != 0) begin
            tx_r  <= frame[0];               // next bit
            frame <= frame >> 1;
            left  <= left - 1'b1;
            cnt   <= CLKS_PER_BIT - 1;
        end else begin
            busy <= 1'b0;                    // the stop bit has had its full bit time
        end
    end
endmodule
