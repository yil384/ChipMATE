"""System prompts and answer extraction for the Verilog and Python agents.

The prompt templates mirror the structured code-skeleton format described in
the paper (combinational / sequential separation + few-shot example). They
match the prompts used to produce the ChipMATE benchmark numbers in Table 1.
"""
import re


V_PROMPT_TEMPLATE = (
    "You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer."
    "The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to write verilog code. After thinking, when you finally reach a conclusion, enclose the final verilog code in ```verilog ``` within <answer> </answer> tags. i.e., <answer> ```verilog\n module top_module(in, out, ...); ... ``` </answer>.\n"
    " IMPORTANT: You MUST always output the COMPLETE Verilog code inside ```verilog``` code blocks. Never respond with only text explanations. Always include the full module code{maybe_even_if}.\n"
    "Question:\n{question}\n\n"
    "{previous_code}"
    "{previous_error_log}"
    "{refine_instr}"
    "Answer:\n"
)

P_PROMPT_TEMPLATE = (
    """You are a helpful assistant.\n\nYour task is to write **Python code**, NOT Verilog.\n\nYou must implement a Python module that simulates the hardware DUT behavior.

Requirements:
1. Class: Define class TopModule.
2. Interface: Implement eval(self, inputs: dict) -> dict.
   - inputs/return keys are port names (str), values are integers.
3. Logic:
   - Sequential: Store state in self variables. Each eval() call represents one posedge clock.
   - Combinational: No state storage needed.
4. Bit-width: You MUST manually apply bit-masks to simulate hardware width (e.g., val & 0xFF).
5. Output: RETURN ONLY VALID PYTHON CODE in a single block. No explanations or text.

Reference Implementation:
```python
class TopModule:
    def __init__(self):
        self.q = 0
    def eval(self, inputs: dict) -> dict:
        d = inputs.get("d", 0)
        rst = inputs.get("reset", 0)
        if rst:
            self.q = 0
        else:
            self.q = d & 0xFF
        return {{"q": self.q}}
```"""
    "The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to write python code. After thinking, when you finally reach a conclusion, enclose the final python code in ```python ``` within <answer> </answer> tags. i.e., <answer> ```python\n class TopModule: ... ``` </answer>.\n"
    "Question:\n{question}\n\n"
    "{previous_code}"
    "{previous_error_log}"
    "{refine_instr}"
    "Answer:\n"
)


def build_v_prompt(question, turn_idx, v_history, v_error):
    previous_code = ""
    if turn_idx > 0 and v_history:
        valid = [h for h in v_history if h and not h.startswith("We can not extract")]
        recent = valid[-2:]
        parts = []
        for i, prev in enumerate(recent):
            label = f"Attempt {len(valid) - len(recent) + i + 1}"
            if len(prev) > 1500:
                prev = prev[:1500] + "\n// ...(code truncated)..."
            parts.append(f"{label}:\n```verilog\n{prev}\n```")
        if parts:
            previous_code = "Your previous Verilog attempts:\n" + "\n\n".join(parts) + "\n\n"
    previous_error_log = ""
    if turn_idx > 0 and v_error:
        err = v_error if len(v_error) <= 2000 else v_error[:2000] + "\n...(truncated)..."
        previous_error_log = f"Previous verification error:\n{err}\n\n"
    refine = ("Please refine your Verilog code to improve correctness and quality. "
              "You MUST output the complete module code in ```verilog``` blocks.\n\n") if turn_idx > 0 else ""
    return V_PROMPT_TEMPLATE.format(
        question=question,
        maybe_even_if=", even if no changes are needed" if turn_idx > 0 else "",
        previous_code=previous_code,
        previous_error_log=previous_error_log,
        refine_instr=refine,
    )


def build_p_prompt(question, turn_idx, p_history, p_error):
    previous_code = ""
    if turn_idx > 0 and p_history:
        valid = [h for h in p_history if h and not h.startswith("We can not extract")]
        recent = valid[-2:]
        parts = []
        for i, prev in enumerate(recent):
            label = f"Attempt {len(valid) - len(recent) + i + 1}"
            if len(prev) > 1500:
                prev = prev[:1500] + "\n# ...(code truncated)..."
            parts.append(f"{label}:\n```python\n{prev}\n```")
        if parts:
            previous_code = "Your previous Python attempts:\n" + "\n\n".join(parts) + "\n\n"
    previous_error_log = ""
    if turn_idx > 0 and p_error:
        err = p_error if len(p_error) <= 2000 else p_error[:2000] + "\n...(truncated)..."
        previous_error_log = f"Previous verification error:\n{err}\n\n"
    refine = ("Please refine your Python code based on the mismatch feedback above. "
              "Output complete class TopModule in ```python``` blocks.\n\n") if turn_idx > 0 else ""
    return P_PROMPT_TEMPLATE.format(
        question=question,
        previous_code=previous_code,
        previous_error_log=previous_error_log,
        refine_instr=refine,
    )


def extract_verilog(response):
    if not response:
        return ""
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", response, re.DOTALL)
    if m:
        body = m.group(1)
        ms = re.findall(r"```(?:verilog|systemverilog|sv)\s*\n(.*?)```", body, re.DOTALL)
        if not ms:
            ms = re.findall(r"```(?:verilog|systemverilog|sv)\s*\n(.*)", body, re.DOTALL)
        if ms:
            return ms[-1].strip()
        gs = re.findall(r"```(.*?)```", body, re.DOTALL)
        if gs:
            return gs[-1].strip()
        return body.strip()
    ms = re.findall(r"```verilog(.*?)```", response, re.DOTALL)
    if ms:
        return ms[-1].strip()
    ms = re.findall(r"```(.*?)```", response, re.DOTALL)
    if ms:
        return ms[-1].strip()
    m = re.search(r"```(?:verilog|systemverilog|sv)?\s*\n(.*)", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def extract_python(response):
    if not response:
        return ""
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", response, re.DOTALL)
    if m:
        body = m.group(1)
        ms = re.findall(r"```python\s*\n(.*?)```", body, re.DOTALL)
        if not ms:
            ms = re.findall(r"```python\s*\n(.*)", body, re.DOTALL)
        if ms:
            return ms[-1].strip()
        gs = re.findall(r"```(.*?)```", body, re.DOTALL)
        if gs:
            return gs[-1].strip()
        return body.strip()
    ms = re.findall(r"```python(.*?)```", response, re.DOTALL)
    if ms:
        return ms[-1].strip()
    ms = re.findall(r"```(.*?)```", response, re.DOTALL)
    if ms:
        return ms[-1].strip()
    return ""
