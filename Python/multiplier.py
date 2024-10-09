import random
import cocotb
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
from cocotb.binary import BinaryValue
from cocotb.coverage import (
    CoverageGroup,
    Cross,
    tuple_list,
    BitCoverage,
)

from coveragepy.report import Reporter, SummaryReporter
from pathlib import Path

@cocotb.test()
async def multiplier_tb(dut):
    """Testbench for a Multiplier DUT."""

    # Configuration
    DATA_WIDTH = 32  
    CLK_PERIOD = 20  # ns
    DATA_RANGE = {"to": 2**31 - 1, "from": -2**31}

    # Setup Clocks and Signals
    clock = Clock(dut.clk, period=CLK_PERIOD, units="ns")
    cocotb.fork(clock.start())
    await RisingEdge(dut.clk)

    # Reset Functionality
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    assert dut.product.value == BinaryValue(0, width=2*DATA_WIDTH) 

    # Coverage Setup
    product_cover = CoverageGroup("product_coverage")
    product_bit = BitCoverage("bit_coverage", dut.product)
    product_cover.add_coverage_item(product_bit)

    input_ranges = [(0, 0), (0, DATA_RANGE["to"]), (DATA_RANGE["to"], 0), 
                    (DATA_RANGE["to"], DATA_RANGE["to"]), 
                    (DATA_RANGE["from"], 0), (0, DATA_RANGE["from"]),
                    (DATA_RANGE["from"], DATA_RANGE["from"]), 
                    (DATA_RANGE["from"], DATA_RANGE["to"])]

    input_cover = CoverageGroup("input_coverage")
    input_cross = Cross("input_cross", [
        tuple_list("low", input_ranges[0]),
        tuple_list("high_a", input_ranges[1]), 
        tuple_list("high_b", input_ranges[2]),
        tuple_list("high_both", input_ranges[3]), 
        tuple_list("low_a", input_ranges[4]),
        tuple_list("low_b", input_ranges[5]),
        tuple_list("low_ab", input_ranges[6]), 
        tuple_list("mixed", input_ranges[7])
    ])
    input_cover.add_coverage_item(input_cross)

    # Test Scenarios
    async def apply_input(a, b):
        dut.a.value = BinaryValue(a, width=DATA_WIDTH)
        dut.b.value = BinaryValue(b, width=DATA_WIDTH)
        await RisingEdge(dut.clk)

        if (a == 0) and (b == 0):
            product_cover.low.sample()
        elif a == 0:
            product_cover.high_b.sample()
        elif b == 0:
            product_cover.high_a.sample()
        else:
            if a > 0:
                if b > 0:
                    product_cover.high_both.sample()
                else:
                    product_cover.mixed.sample()
            else: 
                if b > 0:
                    product_cover.mixed.sample()
                else:
                    product_cover.low_ab.sample()
        input_cross.sample()

        assert dut.product.value == BinaryValue(a*b, width=2*DATA_WIDTH)

    # 1. Edge-Triggered Inputs (with multiple clocks)
    await apply_input(1, 2)

    await apply_input(3, 4)
    await apply_input(10, 20)
    await apply_input(65532, 97563)
    await apply_input(-5, 3)
    await apply_input(65532, -97563)
    await apply_input(-5, -3)

    await RisingEdge(dut.clk)
    await apply_input(1, -2)
    await RisingEdge(dut.clk)
    await apply_input(1, 2)

    # 2. Random Inputs (sparse random)
    for _ in range(100):
        a = random.randint(DATA_RANGE["from"], DATA_RANGE["to"])
        b = random.randint(DATA_RANGE["from"], DATA_RANGE["to"])
        await apply_input(a, b)

    for _ in range(10):
        a = random.choice([0, DATA_RANGE["to"], DATA_RANGE["from"]])
        b = random.choice([0, DATA_RANGE["to"], DATA_RANGE["from"]])
        await apply_input(a, b)

    # 3. Parameterized Inputs
    async def test_edge_cases():
        # Test edge cases for data width (kernel needs to handle it)
        for data_width in [4, 8, 16, 32]:
            tb = await cocotb.start_simulation(self.dut)
            await tb._test_multiplier_tb(data_width)  

    # 4. Asynchronous Master Clock
    await cocotb.create_task(asynchronous_clock_stimulus(dut)) 
    while True:
        await RisingEdge(dut.clk)

async def asynchronous_clock_stimulus(dut):
    MASTER_CLOCK_PERIOD = random.randint(15, 25)
    master_clock = Clock(dut.async_clk, period=MASTER_CLOCK_PERIOD, units="ns")
    await master_clock.start()

    # Simulate asynchronous input changes unrelated to the DUT clk
    await RisingEdge(dut.async_clk)
    dut.a.value = BinaryValue(random.randint(-2**15, 2**15 - 1), width=16)

    while True:
        await RisingEdge(master_clock)
        dut.b.value = BinaryValue(random.randint(-2**15, 2**15 - 1), width=16) 

# Helper functions to adjust test parameters
def change_data_width(tb, new_data_width):
    tb.dut.a.width = new_data_width
    tb.dut.b.width = new_data_width
    tb.dut.product.width = 2 * new_data_width

async def _test_multiplier_tb(self, data_width):
    await change_data_width(self, data_width)
    await cocotb.test._multiplier_tb(self.dut)  # Call the original test

    if cocotb.result.TestFailure:
        return  # Don't run assertions if test failed because of bug

    # Generate and print coverage report after each data width run
    print("\n=== Coverage Report for Data Width {} ===".format(data_width))
    for cover_group in [product_cover, input_cover]:
        coverage_data = cover_group.accumulate_coverage()
        reporter = SummaryReporter()
        reporter.report(coverage_data, include_no_branch=False)

        report_file = Path("coverage_{}_{}.txt".format(data_width, cover_group.name))
        with open(report_file, "w") as f:
            Reporter().report(coverage_data, show_missing=True, file=f)

        print("Detailed Report for {} saved to {}".format(cover_group.name, report_file))

# Termination
if cocotb.result.TestFailure:
    print("!!! Testbench Failed !!!")
else:
    print("Testbench Completed Successfully") 