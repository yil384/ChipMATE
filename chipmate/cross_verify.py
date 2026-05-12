"""Cross-verify a Verilog DUT against a Python reference model.

Both implementations are driven with the SAME random input stimuli; outputs are
compared cycle by cycle. No golden testbench or reference implementation is
consulted -- only the port list of a reference Verilog file is read so we can
build a stimulus harness that matches the target interface.

This module is the run-time core of the multi-agent workflow described in the
ChipMATE paper (Section 2.2): the two agents agree on correctness iff
match_rate reaches 1.0 on a shared set of random vectors.

External requirements: `iverilog` and `vvp` must be on PATH.
"""
import json
import random
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path


def normalize_module_name(sv_code, target="TopModule"):
    """Rename the first `module <name>` declaration to a canonical name.

    The Verilog prompts ask the model to write `module top_module(...)` but the
    stimulus harness instantiates `TopModule dut(...)`. We bridge the two by
    rewriting the user's top-level module name to the canonical form here.
    Sub-module instantiations inside the DUT are left untouched.
    """
    m = re.search(r'\bmodule\s+(\w+)', sv_code)
    if not m:
        return sv_code
    old = m.group(1)
    if old == target:
        return sv_code
    # Rename only the first `module <old>` declaration.
    sv_code = re.sub(r'\bmodule\s+' + re.escape(old), f'module {target}',
                     sv_code, count=1)
    # SystemVerilog `endmodule : <old>` style labels.
    sv_code = re.sub(r'endmodule\s*:\s*' + re.escape(old) + r'\b',
                     f'endmodule : {target}', sv_code)
    return sv_code


def parse_ports(verilog_code):
    """Extract (name, width_in_bits) for each input/output port.

    Returns (inputs, outputs). Both are lists of (name, width).
    """
    code = re.sub(r'//[^\n]*', '', verilog_code)
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    inputs, outputs = [], []
    for m in re.finditer(
        r'(input|output)\s*(?:reg|wire|logic)?\s*(?:\[([^\]]+)\])?\s*(\w+)', code
    ):
        direction, wexpr, name = m.group(1), m.group(2), m.group(3)
        w = 1
        if wexpr:
            parts = wexpr.split(':')
            if len(parts) == 2:
                try:
                    w = abs(eval(parts[0].strip()) - eval(parts[1].strip())) + 1
                except Exception:
                    w = 32
        (inputs if direction == 'input' else outputs).append((name, w))
    return inputs, outputs


def is_reset(n):
    n = n.lower()
    return 'reset' in n or 'rst' in n


def is_active_low(n):
    n_ = n.lower()
    return n_.endswith('_n') or n_.endswith('_b') or n_.endswith('_l') or n_.startswith('n_')


def gen_stimuli(inputs, num_tests, seed=42):
    """Sample random input vectors. clk / reset are driven by the harness."""
    rng = random.Random(seed)
    stims = []
    for _ in range(num_tests):
        s = {}
        for n, w in inputs:
            if n.lower() == 'clk' or is_reset(n):
                continue
            s[n] = rng.randint(0, (1 << w) - 1)
        stims.append(s)
    return stims


def build_stim_testbench(inputs, outputs, stimuli, has_clk, resets):
    lines = [
        "`timescale 1ns/1ps",
        "module stim_tb();",
    ]
    for n, w in inputs:
        bits = f"[{w-1}:0] " if w > 1 else ""
        lines.append(f"  reg {bits}{n};")
    for n, w in outputs:
        bits = f"[{w-1}:0] " if w > 1 else ""
        lines.append(f"  wire {bits}{n};")
    conns = ", ".join(f".{n}({n})" for n, _ in inputs + outputs)
    lines.append(f"  TopModule dut ({conns});")
    if has_clk:
        lines.append("  initial clk = 0;")
        lines.append("  always #5 clk = ~clk;")
    lines.append("  initial begin")
    for n, _ in inputs:
        if n.lower() == 'clk':
            continue
        lines.append(f"    {n} = 0;")
    # Assert reset for three cycles (matches Python harness exactly).
    for r, _, active_low in resets:
        asserted = 0 if active_low else 1
        lines.append(f"    {r} = {asserted};")
    if has_clk:
        lines.append("    @(posedge clk); #1;")
        lines.append("    @(posedge clk); #1;")
        lines.append("    @(posedge clk); #1;")
    for r, _, active_low in resets:
        deasserted = 1 if active_low else 0
        lines.append(f"    {r} = {deasserted};")
    if has_clk:
        lines.append("    @(posedge clk); #1;")
    for idx, stim in enumerate(stimuli):
        assigns = " ".join(f"{n} = 'h{v:x};" for n, v in stim.items())
        lines.append(f"    {assigns}")
        if has_clk:
            lines.append("    @(posedge clk); #1;")
        else:
            lines.append("    #1;")
        fmts = " ".join(f"{n}=%0d" for n, _ in outputs)
        args = ", ".join(n for n, _ in outputs)
        lines.append(f'    $display("TEST_{idx} {fmts}", {args});')
    lines.append("    $finish;")
    lines.append("  end")
    lines.append("  initial begin #100000 $display(\"TIMEOUT\"); $finish; end")
    lines.append("endmodule")
    return "\n".join(lines)


def simulate_verilog(dut_sv, inputs, outputs, stimuli, has_clk, resets, timeout=30):
    """Compile and run the DUT against the generated stimulus testbench.

    Returns (per_test_outputs, error). per_test_outputs is a list of dicts
    mapping output port name to integer value; error is None on success.
    """
    tb = build_stim_testbench(inputs, outputs, stimuli, has_clk, resets)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "dut.sv").write_text(dut_sv)
        (td / "tb.sv").write_text(tb)
        try:
            cp = subprocess.run(
                ["iverilog", "-g2012", "-o", "sim.vvp", "-s", "stim_tb",
                 str(td / "dut.sv"), str(td / "tb.sv")],
                capture_output=True, text=True, timeout=timeout, errors='replace', cwd=td,
            )
        except subprocess.TimeoutExpired:
            return None, "sv_compile_timeout"
        if cp.returncode != 0:
            err = (cp.stdout + "\n" + cp.stderr).strip()[:1500]
            return None, f"sv_compile_fail: {err}"
        try:
            rp = subprocess.run(
                ["vvp", "-n", "sim.vvp"], capture_output=True, text=True,
                timeout=timeout, errors='replace', cwd=td,
            )
        except subprocess.TimeoutExpired:
            return None, "sv_sim_timeout"
        out = rp.stdout
        if "TIMEOUT" in out:
            return None, "sv_sim_hang"
        results = [{} for _ in stimuli]
        for line in out.splitlines():
            m = re.match(r"TEST_(\d+)\s+(.*)", line.strip())
            if not m:
                continue
            idx = int(m.group(1))
            if idx >= len(results):
                continue
            rest = m.group(2)
            for pair in re.finditer(r"(\w+)=(-?\d+)", rest):
                results[idx][pair.group(1)] = int(pair.group(2))
        return results, None


def simulate_python(dut_py, inputs, outputs, stimuli, has_clk, resets, timeout=30):
    """Execute the Python reference model in an isolated subprocess.

    Returns (per_test_outputs, error) in the same shape as simulate_verilog().
    The user code is expected to define `class TopModule` with a single
    `eval(self, inputs: dict) -> dict` entry point.
    """
    harness = textwrap.dedent("""
        import sys, json, os
        _USER_CODE = open(sys.argv[1]).read()
        _STIMS    = json.load(open(sys.argv[2]))
        _RESETS   = json.load(open(sys.argv[3]))
        _OUTPUTS  = json.load(open(sys.argv[4]))
        _HAS_CLK  = bool(int(sys.argv[5]))

        _NS = {}
        exec(_USER_CODE, _NS)
        if "TopModule" not in _NS:
            print("PY_ERR: no TopModule class", file=sys.stderr); sys.exit(2)
        dut = _NS["TopModule"]()

        if _HAS_CLK and _RESETS:
            for _ in range(3):
                reset_inputs = {}
                for name, active_low in _RESETS:
                    reset_inputs[name] = 0 if active_low else 1
                for stim in _STIMS[:1]:
                    for k in stim:
                        reset_inputs.setdefault(k, 0)
                try:
                    dut.eval(reset_inputs)
                except Exception as e:
                    print(f"PY_ERR_RESET: {type(e).__name__}: {e}", file=sys.stderr); sys.exit(3)
            deassert = {name: (1 if active_low else 0) for name, active_low in _RESETS}
            for stim in _STIMS[:1]:
                for k in stim:
                    deassert.setdefault(k, 0)
            try:
                dut.eval(deassert)
            except Exception as e:
                print(f"PY_ERR_DEASSERT: {type(e).__name__}: {e}", file=sys.stderr); sys.exit(3)

        results = []
        for idx, stim in enumerate(_STIMS):
            inp = dict(stim)
            for name, active_low in _RESETS:
                inp.setdefault(name, 1 if active_low else 0)
            try:
                out = dut.eval(inp)
            except Exception as e:
                print(f"PY_ERR_EVAL_{idx}: {type(e).__name__}: {e}", file=sys.stderr); sys.exit(4)
            if not isinstance(out, dict):
                print(f"PY_ERR_EVAL_{idx}: returned non-dict: {type(out).__name__}", file=sys.stderr); sys.exit(5)
            row = {}
            for pname in _OUTPUTS:
                v = out.get(pname, None)
                if v is None:
                    row[pname] = None
                else:
                    try:
                        row[pname] = int(v)
                    except Exception:
                        row[pname] = None
            results.append(row)
        print("PY_RESULTS " + json.dumps(results))
    """).lstrip()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "harness.py").write_text(harness)
        (td / "user.py").write_text(dut_py)
        (td / "stims.json").write_text(json.dumps(stimuli))
        (td / "resets.json").write_text(json.dumps([[r, al] for r, _, al in resets]))
        (td / "outputs.json").write_text(json.dumps([n for n, _ in outputs]))
        try:
            cp = subprocess.run(
                ["python3", str(td / "harness.py"), str(td / "user.py"),
                 str(td / "stims.json"), str(td / "resets.json"),
                 str(td / "outputs.json"), "1" if has_clk else "0"],
                capture_output=True, text=True, timeout=timeout, errors='replace',
            )
        except subprocess.TimeoutExpired:
            return None, "py_exec_timeout"
        if cp.returncode != 0:
            err = cp.stderr.strip()[:800]
            return None, f"py_exec_fail: {err}"
        for line in cp.stdout.splitlines():
            if line.startswith("PY_RESULTS "):
                try:
                    return json.loads(line[len("PY_RESULTS "):]), None
                except Exception as e:
                    return None, f"py_parse_fail: {e}"
        return None, "py_no_results"


def cross_verify(ref_sv, dut_sv, dut_py, num_tests=30, seed=42):
    """Run dut_sv and dut_py on identical random stimuli and compare outputs.

    Arguments
    ---------
    ref_sv     : Verilog source whose port list defines the interface. Its body
                 is NOT evaluated -- only port names / widths are read.
    dut_sv     : Verilog source produced by the Verilog agent.
    dut_py     : Python source produced by the Python reference-model agent.
    num_tests  : Number of random input vectors to apply.
    seed       : RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
      match_rate     : float in [0, 1]; fraction of (test, output_signal) cells
                       that agreed across the two implementations.
      mismatches     : up to 5 example (test, signal, vlog, py, inputs) records.
      sv_error       : compile/simulate error from the Verilog side, or None.
      py_error       : execute error from the Python side, or None.
      num_tests, total_checks, mismatched, has_clk, meta_error
    """
    inputs, outputs = parse_ports(ref_sv)
    if not outputs:
        return {"match_rate": 0.0, "mismatches": [], "sv_error": None,
                "py_error": None, "num_tests": 0, "has_clk": False,
                "meta_error": "no output ports parsed"}
    has_clk = any(n.lower() == 'clk' for n, _ in inputs)
    resets = [(n, w, is_active_low(n)) for n, w in inputs
              if is_reset(n) and n.lower() != 'clk']
    stimuli = gen_stimuli(inputs, num_tests, seed)
    dut_sv = normalize_module_name(dut_sv)
    sv_out, sv_err = simulate_verilog(dut_sv, inputs, outputs, stimuli, has_clk, resets)
    if sv_err:
        return {"match_rate": 0.0, "mismatches": [], "sv_error": sv_err,
                "py_error": None, "num_tests": num_tests, "has_clk": has_clk}
    py_out, py_err = simulate_python(dut_py, inputs, outputs, stimuli, has_clk, resets)
    if py_err:
        return {"match_rate": 0.0, "mismatches": [], "sv_error": None,
                "py_error": py_err, "num_tests": num_tests, "has_clk": has_clk}
    mismatches = []
    total = 0
    mismatched = 0
    for i, (sv_row, py_row) in enumerate(zip(sv_out, py_out)):
        for oname, _ in outputs:
            total += 1
            sv_v = sv_row.get(oname)
            py_v = py_row.get(oname)
            if sv_v != py_v:
                mismatched += 1
                if len(mismatches) < 5:
                    mismatches.append({
                        "test": i, "signal": oname,
                        "verilog": sv_v, "python": py_v,
                        "inputs": stimuli[i],
                    })
    match_rate = 1.0 - (mismatched / total) if total > 0 else 0.0
    return {"match_rate": match_rate, "mismatches": mismatches,
            "sv_error": None, "py_error": None,
            "num_tests": num_tests, "total_checks": total,
            "mismatched": mismatched, "has_clk": has_clk}
