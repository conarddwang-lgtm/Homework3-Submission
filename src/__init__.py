"""
Week 3: Pretraining Data Collection & Voice Agents - Shared Modules

This package contains reusable code for all notebooks.

Modules with heavy dependencies (torch, transformers, whisper, etc.)
are imported on-demand in each notebook rather than eagerly here.
"""

__version__ = "3.0.0"

# Core modules (lightweight deps only: requests, anthropic, datetime)
from .llm_client import LLMClient
from .cost_tracker import CostTracker
from .utils import estimate_tokens, estimate_cost, format_response, save_task_output, append_to_reflection

__all__ = [
    'LLMClient',
    'CostTracker',
    'estimate_tokens',
    'estimate_cost',
    'format_response',
    'save_task_output',
    'append_to_reflection',
]

# Week 3 modules are imported directly in notebooks:
#   from src.scraping_utils import ...
#   from src.ocr_utils import ...
#   from src.audio_utils import ...
#   from src.data_pipeline import ...
