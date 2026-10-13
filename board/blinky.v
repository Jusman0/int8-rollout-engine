`timescale 1ns/1ps
// blinky: the first thing to load onto a new Tang Nano 20K. It proves the
// toolchain (yosys, nextpnr, gowin_pack), the USB cable and openFPGALoader
// all work before anything more complicated is tried.
//
// The six LEDs show a binary count that goes up about 6 times a second, so
// the whole pattern repeats every 64 counts (about 10 s). Holding S1 (the
// button next to pin 88) pauses the count.
//
// What each outcome means:
//   LEDs count                 bitstream loaded, clock runs, LED polarity right
//   count pauses while S1 held S1 reads 1 when pressed, as board/README.md says
//   counts only while S1 held  the button polarity is the other way round
//   all LEDs dark or all lit   not loaded, or the LED polarity is wrong

module blinky (
    input  wire       clk,         // 27 MHz, pin 4
    input  wire       btn,         // S1, pin 88: 1 while pressed
    output wire [5:0] led_n        // pins 15-20, 0 = LED on
);
    // 27,000,000 / 2^22 = 6.4 counts per second on the lowest LED
    reg [27:0] count = 28'd0;
    always @(posedge clk)
        if (!btn) count <= count + 28'd1;

    // the LEDs light on 0, so invert: a 1 in the count is a lit LED
    assign led_n = ~count[27:22];
endmodule
