"""LLM backends used by the inference loop.

Two backend families are supported:

1.  `openai-compat` -- any service that speaks the OpenAI Chat Completions
    protocol. This covers, among others:

      * OpenAI / Azure OpenAI         (`base_url="https://api.openai.com/v1"`)
      * DeepSeek                       (`base_url="https://api.deepseek.com"`)
      * Google Gemini OpenAI shim      (`base_url="https://generativelanguage.googleapis.com/v1beta/openai/"`)
      * A local `vllm serve` instance  (`base_url="http://localhost:8000/v1"`)
      * Any other OpenAI-compatible gateway.

    To run a downloaded HuggingFace model on your own GPUs, start vLLM as a
    local server, then point the backend at it:

        vllm serve core12345/ChipMATE-V-9B --port 8000

2.  `anthropic` -- Anthropic's native Messages API for Claude. Selected when
    `provider="anthropic"`. Requires `pip install chipmate-inference[anthropic]`.

A backend is a callable returning `List[str]`: given a prompt and a sample
count `n`, it returns `n` completions. The inference loop never sees provider
specifics beyond this.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional


class Backend:
    """Abstract backend interface."""

    def generate(self, prompt: str, n: int = 1, temperature: float = 0.6,
                 max_tokens: int = 4096) -> List[str]:
        raise NotImplementedError


class OpenAICompatBackend(Backend):
    """Backend for any OpenAI Chat-Completions-compatible endpoint."""

    def __init__(self, model: str, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, timeout: int = 900,
                 max_retries: int = 4):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai>=1.0 is required for the openai-compat backend. "
                "Install it with `pip install openai`."
            ) from e
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        kw = {}
        if api_key:
            kw["api_key"] = api_key
        if base_url:
            kw["base_url"] = base_url
        self._client = OpenAI(**kw)
        # Some reasoning models (e.g. `*-reasoner`) require a larger output budget.
        self._is_reasoner = "reasoner" in model.lower() or "o1" in model.lower()

    def _call_once(self, prompt: str, temperature: float, max_tokens: int) -> str:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": self.timeout,
        }
        if self._is_reasoner:
            kwargs["max_tokens"] = max(max_tokens, 32768)
        else:
            kwargs.update(temperature=temperature, top_p=0.95, max_tokens=max_tokens)
        last_err: Optional[Exception] = None
        for i in range(self.max_retries):
            try:
                r = self._client.chat.completions.create(**kwargs)
                return r.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                time.sleep(2 ** i)
        raise RuntimeError(f"openai-compat call failed after {self.max_retries} retries: {last_err}")

    def generate(self, prompt, n=1, temperature=0.6, max_tokens=4096):
        if n <= 1:
            return [self._call_once(prompt, temperature, max_tokens)]
        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = [pool.submit(self._call_once, prompt, temperature, max_tokens)
                    for _ in range(n)]
            return [f.result() for f in futs]


class AnthropicBackend(Backend):
    """Backend for Anthropic's native Messages API (Claude)."""

    def __init__(self, model: str, api_key: Optional[str] = None,
                 timeout: int = 900, max_retries: int = 4):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic>=0.34 is required for the anthropic backend. "
                "Install it with `pip install chipmate-inference[anthropic]`."
            ) from e
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def _call_once(self, prompt: str, temperature: float, max_tokens: int) -> str:
        last_err: Optional[Exception] = None
        for i in range(self.max_retries):
            try:
                r = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.timeout,
                )
                # Concatenate all text blocks.
                parts = [b.text for b in r.content if getattr(b, "type", "") == "text"]
                return "".join(parts)
            except Exception as e:
                last_err = e
                time.sleep(2 ** i)
        raise RuntimeError(f"anthropic call failed after {self.max_retries} retries: {last_err}")

    def generate(self, prompt, n=1, temperature=0.6, max_tokens=4096):
        if n <= 1:
            return [self._call_once(prompt, temperature, max_tokens)]
        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = [pool.submit(self._call_once, prompt, temperature, max_tokens)
                    for _ in range(n)]
            return [f.result() for f in futs]


def make_backend(provider: str = "openai-compat", *, model: str,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: int = 900,
                 max_retries: int = 4) -> Backend:
    """Construct a backend by name.

    Parameters
    ----------
    provider  : "openai-compat" (default) or "anthropic".
    model     : Model id understood by the chosen provider.
                  - openai-compat: e.g. "deepseek-chat", "gpt-4o", "core12345/ChipMATE-V-9B".
                  - anthropic:     e.g. "claude-opus-4-7", "claude-haiku-4-5-20251001".
    api_key   : Provider API key. If None, falls back to the SDK's default
                env var (OPENAI_API_KEY / ANTHROPIC_API_KEY).
    base_url  : Required when targeting a non-OpenAI endpoint
                (DeepSeek / Gemini / vLLM). Ignored by the anthropic backend.

    Returns
    -------
    Backend
    """
    provider = (provider or "openai-compat").lower()
    if provider in ("openai-compat", "openai", "deepseek", "gemini", "vllm"):
        return OpenAICompatBackend(model=model, api_key=api_key,
                                   base_url=base_url, timeout=timeout,
                                   max_retries=max_retries)
    if provider == "anthropic":
        return AnthropicBackend(model=model, api_key=api_key,
                                timeout=timeout, max_retries=max_retries)
    raise ValueError(f"Unknown provider: {provider!r}. "
                     "Expected 'openai-compat' or 'anthropic'.")
