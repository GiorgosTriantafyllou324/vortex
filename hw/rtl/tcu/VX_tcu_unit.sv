// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

module VX_tcu_unit import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    input wire              clk,
    input wire              reset,

    // Inputs
    VX_dispatch_if.slave    dispatch_if [`ISSUE_WIDTH],

`ifdef TCU_OP
    VX_lsu_mem_if.master    tcu_lsu_mem_if,
    VX_txbar_bus_if.master  txbar_bus_if,
`endif

    // Outputs
    VX_commit_if.master     commit_if [`ISSUE_WIDTH]
);
    localparam BLOCK_SIZE = `NUM_TCU_BLOCKS;
    localparam NUM_LANES  = `NUM_TCU_LANES;

    `STATIC_ASSERT (BLOCK_SIZE == `ISSUE_WIDTH, ("must be full issue execution"));
    `STATIC_ASSERT (NUM_LANES == `NUM_THREADS, ("must be full warp execution"));
    `SCOPE_IO_SWITCH (BLOCK_SIZE);

    VX_execute_if #(
        .data_t (tcu_execute_t)
    ) per_block_execute_if[BLOCK_SIZE]();

    VX_lane_dispatch #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (3)
    ) lane_dispatch (
        .clk        (clk),
        .reset      (reset),
        .dispatch_if(dispatch_if),
        .execute_if (per_block_execute_if)
    );

    VX_result_if #(
        .data_t (tcu_result_t)
    ) per_block_result_if[BLOCK_SIZE]();

`ifdef TCU_OP
    VX_txbar_bus_if per_block_txbar_if[BLOCK_SIZE]();
`endif

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin : g_blocks
    `ifdef TCU_OP
        VX_tcu_op_core #(
            .INSTANCE_ID (`SFORMATF(("%s-op_core%0d", INSTANCE_ID, block_idx)))
        ) tcu_fp (
            `SCOPE_IO_BIND (block_idx)
            .clk            (clk),
            .reset          (reset),
            .execute_if     (per_block_execute_if[block_idx]),
            .tcu_lsu_mem_if (tcu_lsu_mem_if),
            .txbar_bus_if   (per_block_txbar_if[block_idx]),
            .result_if      (per_block_result_if[block_idx])
        );
    `else
        VX_tcu_core #(
            .INSTANCE_ID (`SFORMATF(("%s-fused%0d", INSTANCE_ID, block_idx)))
        ) tcu_core (
            `SCOPE_IO_BIND (block_idx)
            .clk        (clk),
            .reset      (reset),
            .execute_if (per_block_execute_if[block_idx]),
            .result_if  (per_block_result_if[block_idx])
        );
    `endif
    end

`ifdef TCU_OP
    VX_txbar_arb #(
        .NUM_REQS (BLOCK_SIZE),
        .ARBITER  ("R"),
        .OUT_BUF  (0)
    ) txbar_arb (
        .clk       (clk),
        .reset     (reset),
        .bus_in_if (per_block_txbar_if),
        .bus_out_if(txbar_bus_if)
    );
`endif

    VX_lane_gather #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_BUF    (3)
    ) lane_gather (
        .clk       (clk),
        .reset     (reset),
        .result_if (per_block_result_if),
        .commit_if (commit_if)
    );

    // Debugging
    always_ff @(posedge clk) begin
        if (~reset && per_block_execute_if[0].valid && per_block_execute_if[0].ready) begin
        `ifdef TCU_OP
            if (per_block_execute_if[0].data.op_type == INST_TCU_MMA_OP) begin
                `TRACE(1, ("%t: [tcu_unit]: Activated (outer-product)\n", $time));
            end
        `else
            if (per_block_execute_if[0].data.op_type == INST_TCU_WMMA) begin
                `TRACE(1, ("%t: [tcu_unit]: Activated (inner-product dense)\n", $time)); 
            end
            `ifdef TCU_SPARSE_ENABLE
            else if (per_block_execute_if[0].data.op_type == INST_TCU_WMMA_SP) begin
                `TRACE(1, ("%t: [tcu_unit]: Activated (inner-product sparse)\n", $time));
            end
            `endif
        `endif
        end
    end

endmodule
