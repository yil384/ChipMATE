"""Command-line entry point for batch ChipMATE inference.

Reads problems from a JSONL file (one problem per line), runs the multi-agent
loop on each, and writes per-problem results to an output JSONL.

Expected input row shape::

    {"task_id": "...", "question": "...", "ref_sv": "..."}

Where:

- ``question`` is the natural-language spec given to both agents.
- ``ref_sv``  is reference Verilog whose port list defines the interface.
              Its body is not consulted (only the port declarations are read).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .backends import make_backend
from .inference import (
    DEFAULT_MAX_TURNS,
    DEFAULT_N,
    DEFAULT_NUM_VERIFY_TESTS,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    run_problem,
)


def _iter_jsonl(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="chipmate",
        description="Multi-agent inference with cross-verification "
                    "(Verilog agent + Python reference-model agent).",
    )
    ap.add_argument("--input", required=True,
                    help="Input JSONL with one {task_id, question, ref_sv} per line.")
    ap.add_argument("--out", required=True,
                    help="Output JSONL; one MultiAgentResult dict per line.")

    # Backend selection. Both agents share one backend by default; pass a
    # separate --p-* set of flags if you want the Python agent to use a
    # different model (e.g. ChipMATE-V-9B + ChipMATE-P-9B served from two
    # different vLLM endpoints).
    ap.add_argument("--provider", default="openai-compat",
                    choices=["openai-compat", "anthropic"],
                    help="Backend family. (default: openai-compat)")
    ap.add_argument("--model", required=True,
                    help="Model id for the Verilog agent (and Python agent "
                         "unless --p-model is also passed).")
    ap.add_argument("--api-key", default=None,
                    help="API key. Defaults to provider's SDK env var "
                         "(OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY).")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible base URL. Required for DeepSeek "
                         "/ Gemini / local vLLM. Ignored for anthropic.")
    ap.add_argument("--p-provider", default=None)
    ap.add_argument("--p-model", default=None)
    ap.add_argument("--p-api-key", default=None)
    ap.add_argument("--p-base-url", default=None)

    # Inference hyperparameters.
    ap.add_argument("-n", "--n", type=int, default=DEFAULT_N,
                    help="Best-of-N: candidate pairs per turn. (default: %(default)d)")
    ap.add_argument("-t", "--max-turns", type=int, default=DEFAULT_MAX_TURNS,
                    help="Maximum cross-verification turns. (default: %(default)d)")
    ap.add_argument("--num-verify-tests", type=int, default=DEFAULT_NUM_VERIFY_TESTS,
                    help="Random vectors used per cross-verify call.")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)

    # Driver knobs.
    ap.add_argument("--workers", type=int, default=4,
                    help="Number of problems run concurrently.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Take only the first N problems from --input.")
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    # Resolve API key from environment if not given. Convenience for the most
    # common cases; the SDKs also auto-read these themselves.
    if args.provider == "openai-compat" and args.api_key is None:
        args.api_key = (os.environ.get("OPENAI_API_KEY")
                        or os.environ.get("DEEPSEEK_API_KEY"))
    if args.provider == "anthropic" and args.api_key is None:
        args.api_key = os.environ.get("ANTHROPIC_API_KEY")

    v_backend = make_backend(
        provider=args.provider, model=args.model,
        api_key=args.api_key, base_url=args.base_url,
    )
    if args.p_model:
        p_backend = make_backend(
            provider=(args.p_provider or args.provider), model=args.p_model,
            api_key=(args.p_api_key or args.api_key),
            base_url=(args.p_base_url or args.base_url),
        )
    else:
        p_backend = v_backend

    problems = list(_iter_jsonl(args.input))
    if args.offset:
        problems = problems[args.offset:]
    if args.limit is not None:
        problems = problems[:args.limit]
    print(f"Loaded {len(problems)} problems; n={args.n} max_turns={args.max_turns}",
          file=sys.stderr)

    Path(os.path.dirname(os.path.abspath(args.out)) or ".").mkdir(parents=True, exist_ok=True)
    f_out = open(args.out, "w")

    def _work(row):
        return run_problem(
            task_id=row["task_id"],
            question=row["question"],
            ref_sv=row["ref_sv"],
            v_backend=v_backend,
            p_backend=p_backend,
            n=args.n,
            max_turns=args.max_turns,
            num_verify_tests=args.num_verify_tests,
            temperature=args.temperature,
            seed=args.seed,
        )

    t0 = time.time()
    done = 0
    n_match = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_work, row): row for row in problems}
        for f in as_completed(futs):
            row = futs[f]
            try:
                res = f.result()
                f_out.write(json.dumps(res.to_dict()) + "\n")
                f_out.flush()
                if res.matched:
                    n_match += 1
            except Exception as e:
                f_out.write(json.dumps({
                    "task_id": row.get("task_id", "?"),
                    "verilog": "", "python": "", "matched": False,
                    "best_match_rate": -1.0, "best_turn": -1,
                    "turns_used": 0, "turn_logs": [], "error": str(e),
                }) + "\n")
                f_out.flush()
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed else 0.0
            eta = (len(problems) - done) / rate if rate else 0.0
            print(f"  {done}/{len(problems)}  matched={n_match}  "
                  f"rate={rate:.2f}/s  eta={eta:.0f}s",
                  file=sys.stderr)

    f_out.close()
    print(f"Done in {time.time()-t0:.0f}s. matched={n_match}/{len(problems)} -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
