"""Per-turn feedback formatters for the Verilog and Python agents.

Each agent receives a structured natural-language diff of the mismatches its
peer surfaced via cross_verify(). The format follows the paper's
"waveform-to-natural-language" converter, listing the first few mismatched
input vectors and outputs.
"""


def format_v_feedback(verify):
    if verify.get("meta_error"):
        return f"Could not run verification (meta: {verify['meta_error']})."
    if verify.get("sv_error"):
        return f"Your Verilog failed to compile/run:\n{verify['sv_error']}"
    if verify.get("py_error"):
        return "Your peer's Python failed to execute (not your fault, but you can't get cross-verify this turn)."
    mm = verify["mismatches"]
    if not mm:
        return ""
    lines = [
        f"Verilog vs Python: {verify['mismatched']}/{verify['total_checks']} mismatches "
        f"across {verify['num_tests']} test vectors.",
        "First mismatches (got = your Verilog, exp = peer Python):",
    ]
    for ex in mm[:5]:
        lines.append(f"  Test {ex['test']}, signal '{ex['signal']}': got={ex['verilog']}, exp={ex['python']}")
        ins = ex.get("inputs", {})
        if ins:
            ins_s = ", ".join(f"{k}={v}" for k, v in sorted(ins.items()))
            lines.append(f"    (inputs: {ins_s})")
    lines.append(
        "Check your logic carefully. Either you or the Python agent is wrong — "
        "only change your code if you think your previous code is wrong."
    )
    return "\n".join(lines)


def format_p_feedback(verify):
    if verify.get("meta_error"):
        return f"Could not run verification (meta: {verify['meta_error']})."
    if verify.get("py_error"):
        return f"Your Python failed to execute:\n{verify['py_error']}"
    if verify.get("sv_error"):
        return "Your peer's Verilog failed to compile/run (not your fault, but you can't get cross-verify this turn)."
    mm = verify["mismatches"]
    if not mm:
        return ""
    lines = [
        f"Python vs Verilog: {verify['mismatched']}/{verify['total_checks']} mismatches "
        f"across {verify['num_tests']} test vectors.",
        "First mismatches (got = your Python, exp = peer Verilog):",
    ]
    for ex in mm[:5]:
        lines.append(f"  Test {ex['test']}, signal '{ex['signal']}': got={ex['python']}, exp={ex['verilog']}")
        ins = ex.get("inputs", {})
        if ins:
            ins_s = ", ".join(f"{k}={v}" for k, v in sorted(ins.items()))
            lines.append(f"    (inputs: {ins_s})")
    lines.append(
        "Check your logic carefully. Either you or the Verilog agent is wrong — "
        "only change your code if you think your previous code is wrong."
    )
    return "\n".join(lines)
