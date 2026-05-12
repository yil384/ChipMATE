"""ChipMATE inference: multi-agent cross-verification for RTL generation."""

from .inference import run_problem, MultiAgentResult
from .backends import make_backend, Backend

__all__ = ["run_problem", "MultiAgentResult", "make_backend", "Backend"]
__version__ = "0.1.0"
