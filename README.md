# ChipMATE Inference

Multi-agent inference framework for RTL generation with cross-verification.

This is the **inference framework released alongside the ChipMATE paper**
(*ChipMATE: Multi-Agent Training via Reinforcement Learning for Enhanced RTL
Generation*). It implements the multi-agent workflow for RTL generation. 

| Model              | VerilogEval V2 pass@1 |
|--------------------|-----------------------|
| ChipMATE-Agents-4B | **75.0%**             |
| ChipMATE-Agents-9B | **80.1%**             |

## What this repo gives you

- `chipmate.run_problem(...)` — programmatic entry point for the multi-agent
  loop. One call per problem; returns the best Verilog implementation, the
  paired Python reference model, the per-turn cross-verify trace, and a
  `matched` flag.
- `chipmate` CLI — batch runner over a JSONL of problems.
- A pluggable backend layer: any OpenAI-compatible endpoint (OpenAI,
  DeepSeek, Gemini's OpenAI shim, a local `vllm serve` instance, …) or
  Anthropic's native Claude API.
- `cross_verify` — the standalone port-aware random-stimulus harness used
  by the loop. It compiles the DUT via Icarus Verilog and executes the
  Python reference model in an isolated subprocess.

## Install

```bash
pip install chipmate-inference

# Anthropic backend is optional:
pip install "chipmate-inference[anthropic]"
```

You also need `iverilog` (and its `vvp` runtime) on `PATH`. On Debian/Ubuntu:

```bash
sudo apt-get install iverilog
```

Or use the Docker image, which ships iverilog out of the box (see below).

## Quick start

### Option A: hosted LLM API (DeepSeek, OpenAI, Gemini, Claude, …)

```python
from chipmate import make_backend, run_problem

# DeepSeek — the API used to produce the paper's API-based baselines.
backend = make_backend(
    provider="openai-compat",
    model="deepseek-chat",
    api_key="sk-...",                         # or set OPENAI_API_KEY
    base_url="https://api.deepseek.com",
)

result = run_problem(
    task_id="my_problem",
    question="Implement a Verilog module that ...",
    ref_sv="module top_module(input clk, ...);\nendmodule\n",
    v_backend=backend,
    # n=10, max_turns=5  — defaults from the inference grid sweep
)

print(result.verilog)            # final Verilog source
print(result.matched)            # True iff cross-verify reached match_rate == 1.0
print(result.best_match_rate)    # best agreement observed
```

Endpoints for other providers:

| Provider           | `provider`         | `base_url`                                                             |
|--------------------|--------------------|------------------------------------------------------------------------|
| OpenAI / GPT       | `openai-compat`    | `https://api.openai.com/v1`                                            |
| DeepSeek           | `openai-compat`    | `https://api.deepseek.com`                                             |
| Gemini             | `openai-compat`    | `https://generativelanguage.googleapis.com/v1beta/openai/`             |
| Anthropic / Claude | `anthropic`        | *(native)*                                                             |
| Local vLLM server  | `openai-compat`    | `http://localhost:8000/v1`                                             |

### Option B: open-source ChipMATE weights on your own GPU

Run a ChipMATE checkpoint locally with [vLLM](https://github.com/vllm-project/vllm),
which exposes an OpenAI-compatible endpoint:

```bash
# Verilog agent on GPU 0
vllm serve core12345/ChipMATE-V-9B --port 8001 &
# Python reference-model agent on GPU 1
CUDA_VISIBLE_DEVICES=1 vllm serve core12345/ChipMATE-P-9B --port 8002 &
```

```python
from chipmate import make_backend, run_problem

v_backend = make_backend(model="core12345/ChipMATE-V-9B",
                         base_url="http://localhost:8001/v1", api_key="dummy")
p_backend = make_backend(model="core12345/ChipMATE-P-9B",
                         base_url="http://localhost:8002/v1", api_key="dummy")

result = run_problem(
    task_id="my_problem", question="...", ref_sv="...",
    v_backend=v_backend, p_backend=p_backend,
)
```

Both 4B and 9B variants are released on HuggingFace:

- [core12345/ChipMATE-V-4B](https://huggingface.co/core12345/ChipMATE-V-4B) — Verilog agent, 4B
- [core12345/ChipMATE-P-4B](https://huggingface.co/core12345/ChipMATE-P-4B) — Python reference-model agent, 4B
- [core12345/ChipMATE-V-9B](https://huggingface.co/core12345/ChipMATE-V-9B) — Verilog agent, 9B
- [core12345/ChipMATE-P-9B](https://huggingface.co/core12345/ChipMATE-P-9B) — Python reference-model agent, 9B

### Option C: batch CLI

```bash
chipmate \
  --input problems.jsonl \
  --out   results.jsonl \
  --provider openai-compat \
  --model    deepseek-chat \
  --base-url https://api.deepseek.com \
  --api-key  $DEEPSEEK_API_KEY \
  -n 10 -t 5 \
  --workers 4
```

Input JSONL shape (one object per line):

```json
{"task_id": "demo_01", "question": "...natural-language spec...", "ref_sv": "module top_module(...);\nendmodule\n"}
```

`ref_sv` is only used to read the port list (names, widths, clk/reset
conventions). Its body is never executed and never compared against.

## Defaults

`n=10` and `max_turns=5` are the defaults; lower `--n` to cut cost, raise
`--max-turns` to give the agents more chances to converge on hard problems.

## Docker

```bash
docker build -t chipmate-inference .

docker run --rm \
  -e DEEPSEEK_API_KEY=sk-... \
  -v "$PWD":/work \
  chipmate-inference \
    --input /work/problems.jsonl \
    --out   /work/results.jsonl \
    --provider openai-compat \
    --model    deepseek-chat \
    --base-url https://api.deepseek.com \
    -n 10 -t 5
```

The image ships `iverilog` so you don't need to install it on the host.

## Citation

If you use this framework, please cite the ChipMATE paper:

```bibtex
@inproceedings{chipmate2026,
  title     = {ChipMATE: Multi-Agent Training via Reinforcement Learning for Enhanced RTL Generation},
  author    = {ChipMATE authors},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {Under review}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
