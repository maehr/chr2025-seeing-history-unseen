"""
Entry point for running the benchmarking CLI as a module.

Usage:
    python -m src.benchmarking list-models
    python -m src.benchmarking benchmark --model MODEL_NAME
"""

from .cli import cli

if __name__ == "__main__":
    cli()
