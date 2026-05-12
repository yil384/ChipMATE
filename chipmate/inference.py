"""Multi-agent inference loop with best-of-N sampling and cross-verification.

For each problem, both agents independently sample `n` paired candidates per
turn. The pairs (V_i, P_i) are cross-verified; the highest-match-rate pair
is selected as the shared prefix for the next turn (the "backtrack" mechanism
of the paper -- a turn whose best pair is worse than what we have is rejected
and the previous best is kept). The loop terminates when match_rate reaches
1.0 or when `max_turns` is exhausted.

Defaults (`n=10`, `max_turns=5`) are picked from the inference-grid sweep
reported alongside the paper.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .backends import Backend
from .cross_verify import cross_verify
from .feedback import format_p_feedback, format_v_feedback
from .prompts import (
    build_p_prompt,
    build_v_prompt,
    extract_python,
    extract_verilog,
)


DEFAULT_N: int = 10
DEFAULT_MAX_TURNS: int = 5
DEFAULT_NUM_VERIFY_TESTS: int = 30
DEFAULT_TEMPERATURE: float = 0.6
DEFAULT_SEED: int = 42

_INVALID_V = "We can not extract the Verilog code in the output."
_INVALID_P = "We can not extract the Python code in the output."


@dataclass
class TurnLog:
    turn: int
    n: int
    best_pair: int
    best_match_rate: float
    all_match_rates: List[float]
    v_lens: List[int]
    p_lens: List[int]
    best_v_so_far: str = ""
    best_mr_so_far: float = -1.0


@dataclass
class MultiAgentResult:
    """Outcome of running the multi-agent loop on one problem."""

    task_id: str
    verilog: str                # the Verilog implementation the user should keep
    python: str                 # the Python reference model that paired with it
    matched: bool               # True iff a turn reached match_rate == 1.0
    best_match_rate: float      # best cross-verify match rate observed
    best_turn: int              # index of the turn that produced the best pair
    turns_used: int             # number of turns actually executed
    turn_logs: List[TurnLog] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "verilog": self.verilog,
            "python": self.python,
            "matched": self.matched,
            "best_match_rate": self.best_match_rate,
            "best_turn": self.best_turn,
            "turns_used": self.turns_used,
            "turn_logs": [vars(t) for t in self.turn_logs],
        }


def _verify_pair(ref_sv: str, v_code: str, p_code: str,
                 num_tests: int, seed: int) -> Dict[str, Any]:
    if v_code.startswith("We can not") or p_code.startswith("We can not"):
        return {
            "match_rate": 0.0, "mismatches": [], "sv_error": None,
            "py_error": None, "num_tests": 0, "has_clk": False,
            "meta_error": "invalid extracted code",
        }
    return cross_verify(ref_sv, v_code, p_code, num_tests=num_tests, seed=seed)


def run_problem(
    *,
    task_id: str,
    question: str,
    ref_sv: str,
    v_backend: Backend,
    p_backend: Optional[Backend] = None,
    n: int = DEFAULT_N,
    max_turns: int = DEFAULT_MAX_TURNS,
    num_verify_tests: int = DEFAULT_NUM_VERIFY_TESTS,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
) -> MultiAgentResult:
    """Run the multi-agent loop on a single problem.

    Parameters
    ----------
    task_id           : Identifier carried through to the result; not interpreted.
    question          : Natural-language specification given to both agents.
    ref_sv            : Verilog source whose port list defines the target interface.
                        Only its port declarations are consulted.
    v_backend         : Backend used to sample Verilog candidates.
    p_backend         : Backend used to sample Python candidates. Defaults to v_backend.
    n                 : Number of (V, P) candidate pairs sampled per turn (best-of-N).
    max_turns         : Maximum number of cross-verification turns.
    num_verify_tests  : Number of random input vectors used per cross-verify call.
    temperature       : Sampling temperature passed to both backends.
    seed              : RNG seed for stimulus generation (also offsets per turn).

    Returns
    -------
    MultiAgentResult
    """
    if p_backend is None:
        p_backend = v_backend
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if max_turns < 1:
        raise ValueError(f"max_turns must be >= 1, got {max_turns}")

    state: Dict[str, Any] = {
        "v_carry": "", "p_carry": "",
        "v_err_carry": "", "p_err_carry": "",
        "matched": False, "turns": 0,
        "best_v": "", "best_p": "", "best_match_rate": -1.0,
        "best_turn": -1, "best_pair_idx": -1,
        "turn_logs": [],
    }

    for turn in range(max_turns):
        v_prompt = build_v_prompt(
            question, turn,
            v_history=[state["v_carry"]] if state["v_carry"] else [],
            v_error=state["v_err_carry"],
        )
        p_prompt = build_p_prompt(
            question, turn,
            p_history=[state["p_carry"]] if state["p_carry"] else [],
            p_error=state["p_err_carry"],
        )

        # n V's and n P's drawn in parallel; pair index-wise into n candidates.
        with ThreadPoolExecutor(max_workers=2) as outer:
            fv = outer.submit(v_backend.generate, v_prompt, n, temperature)
            fp = outer.submit(p_backend.generate, p_prompt, n, temperature)
            v_outs = fv.result()
            p_outs = fp.result()

        v_codes = [extract_verilog(t or "") or _INVALID_V for t in v_outs]
        p_codes = [extract_python(t or "") or _INVALID_P for t in p_outs]

        with ThreadPoolExecutor(max_workers=n) as vpool:
            futs = {
                vpool.submit(_verify_pair, ref_sv, v_codes[i], p_codes[i],
                             num_verify_tests, seed + turn): i
                for i in range(n)
            }
            results: List[Optional[Dict[str, Any]]] = [None] * n
            for f in as_completed(futs):
                i = futs[f]
                results[i] = f.result()

        best_i = -1
        best_mr = -1.0
        for i, vr in enumerate(results):
            mr = (vr or {}).get("match_rate", -1.0) or -1.0
            if mr > best_mr:
                best_mr = mr
                best_i = i

        bv, bp = v_codes[best_i], p_codes[best_i]
        bverify = results[best_i] or {}
        state["turns"] = turn + 1
        # Backtrack rule: update best only if the new pair strictly improves.
        if best_mr > state["best_match_rate"]:
            state["best_match_rate"] = best_mr
            state["best_v"] = bv
            state["best_p"] = bp
            state["best_turn"] = turn
            state["best_pair_idx"] = best_i

        state["v_carry"] = bv if not bv.startswith("We can not") else state["v_carry"]
        state["p_carry"] = bp if not bp.startswith("We can not") else state["p_carry"]
        state["v_err_carry"] = format_v_feedback(bverify) if bverify.get("match_rate", 0.0) < 1.0 else ""
        state["p_err_carry"] = format_p_feedback(bverify) if bverify.get("match_rate", 0.0) < 1.0 else ""

        turn_log = TurnLog(
            turn=turn, n=n, best_pair=best_i, best_match_rate=best_mr,
            all_match_rates=[(r or {}).get("match_rate", -1.0) for r in results],
            v_lens=[len(c) for c in v_codes],
            p_lens=[len(c) for c in p_codes],
            best_v_so_far=state["best_v"],
            best_mr_so_far=state["best_match_rate"],
        )
        state["turn_logs"].append(turn_log)

        if best_mr >= 1.0:
            state["matched"] = True
            break

    verilog = state["best_v"] or state["v_carry"] or ""
    python = state["best_p"] or state["p_carry"] or ""
    return MultiAgentResult(
        task_id=task_id,
        verilog=verilog,
        python=python,
        matched=state["matched"],
        best_match_rate=state["best_match_rate"],
        best_turn=state["best_turn"],
        turns_used=state["turns"],
        turn_logs=state["turn_logs"],
    )
