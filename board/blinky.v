`timescale 1ns/1ps
// blinky: the first thing to load onto a new Tang Nano 20K. It proves the
// toolchain (yosys, nextpnr, gowin_pack), the USB cable and openFPGALoader
// all work before anything more complicated is tried, and it checks the LED
// and button polarity that board/top.v relies on.
//
// Not pressed: the six LEDs show a binary count that goes up about 6 times a
// second (the whole pattern repeats about every 10 s).
// While S1 (the button on pin 88) is held: exactly ONE LED is lit, LED 0.
//
// Why the S1 pattern: a counter looks like "counting" with either LED
// polarity (inverted, it is just a different pattern), so it cannot show
// polarity. A single lit LED can: with the polarity wrong, five are lit and
// one is dark instead.
//
// What each outcome means:
//   LEDs count                      bitstream loaded, 27 MHz clock running
//   hold S1: one LED lit (LED 0)    LED polarity right (0 = on) and S1 reads 1
//                                   when pressed, as board/README.md says
//   hold S1: one LED dark, 5 lit    LED polarity is the other way round
//   one LED lit WITHOUT pressing,   S1 reads 0 when pressed (button polarity
//     counting while S1 is held     is the other way round)
//   all LEDs dark, nothing changes  not loaded (check the make prog-blinky output)

module blinky (
    input  wire       clk,         // 27 MHz oscillator, pin 4
    input  wire       btn,         // S1, pin 88: 1 while pressed
    output wire [5:0] led_n        // pins 15-20, 0 = LED on
);
    // 27,000,000 / 2^22 = 6.4 counts per second on the lowest LED
    reg [27:0] count = 28'd0;
    always @(posedge clk)
        count <= count + 28'd1;

    // the LEDs light on 0, so invert: a 1 in the pattern is a lit LED
    wire [5:0] pattern = btn ? 6'b000001 : count[27:22];
    assign led_n = ~pattern;
endmodule
