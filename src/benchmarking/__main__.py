"""
Entry point for running the benchmarking CLI as a module.

Usage:
    python -m src.benchmarking list-models
    python -m src.benchmarking benchmark --model MODEL_NAME
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
