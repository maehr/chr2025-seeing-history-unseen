"""
OpenRouter VLM Benchmarking Module

This module provides tools for benchmarking vision-language models
available through the OpenRouter API.
"""

from .openrouter_benchmark import (
    aggregate_results,
    benchmark_model,
    list_models,
)

__all__ = [
    "list_models",
    "benchmark_model",
    "aggregate_results",
]
