"""Minimal example: run the ChipMATE loop on one toy problem.

Usage::

    # 1. Pick any LLM provider you have access to. Examples:
    #
    #    # Local vLLM serving one of the ChipMATE models:
    #    vllm serve core12345/ChipMATE-V-9B --port 8000
    #    export CHIPMATE_BASE_URL=http://localhost:8000/v1
    #    export CHIPMATE_API_KEY=dummy
    #    export CHIPMATE_MODEL=core12345/ChipMATE-V-9B
    #
    #    # DeepSeek API:
    #    export CHIPMATE_BASE_URL=https://api.deepseek.com
    #    export CHIPMATE_API_KEY=sk-...
    #    export CHIPMATE_MODEL=deepseek-chat
    #
    # 2. Run:
    python examples/single_problem.py
"""
import os
import sys

from chipmate import make_backend, run_problem


QUESTION = (
    "Implement a Verilog module top_module with one 8-bit input 'd', a clock "
    "'clk', an active-high synchronous 'reset', and an 8-bit registered output 'q' "
    "that captures 'd' on the rising edge of clk. When reset is high, q is cleared "
    "to 0 on the next rising edge."
)

REF_SV = """
module top_module (
    input        clk,
    input        reset,
    input  [7:0] d,
    output reg [7:0] q
);
endmodule
"""


def main():
    backend = make_backend(
        provider="openai-compat",
        model=os.environ.get("CHIPMATE_MODEL", "deepseek-chat"),
        api_key=os.environ.get("CHIPMATE_API_KEY"),
        base_url=os.environ.get("CHIPMATE_BASE_URL", "https://api.deepseek.com"),
    )
    res = run_problem(
        task_id="demo_register_8b",
        question=QUESTION,
        ref_sv=REF_SV,
        v_backend=backend,
        # n=10, max_turns=5 -- the defaults from the paper's grid sweep.
    )

    print(f"matched={res.matched}  best_match_rate={res.best_match_rate:.3f}  "
          f"best_turn={res.best_turn}  turns_used={res.turns_used}",
          file=sys.stderr)
    print("\n=== Verilog ===\n" + res.verilog)
    print("\n=== Python reference ===\n" + res.python)


if __name__ == "__main__":
    main()
