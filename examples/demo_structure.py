#!/usr/bin/env python3
"""
Example demonstrating the structure and usage of the benchmarking module.

This example shows the API structure without making actual API calls.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmarking.tasks import (
    get_all_tasks,
    get_detailed_analysis_tasks,
    get_simple_description_tasks,
    get_wcag_alttext_tasks,
)


def main() -> None:
    """Demonstrate the benchmarking module structure."""
    print("=" * 70)
    print("OpenRouter VLM Benchmarking - Module Structure Demo")
    print("=" * 70)

    # Show available task sets
    print("\n📝 Available Task Sets:")
    print("-" * 70)

    wcag_tasks = get_wcag_alttext_tasks()
    simple_tasks = get_simple_description_tasks()
    detailed_tasks = get_detailed_analysis_tasks()
    all_tasks = get_all_tasks()

    print(f"\n1. WCAG Alt-text Tasks ({len(wcag_tasks)} tasks)")
    print("   Focus: WCAG 2.2-compliant alt-text generation")
    for i, task in enumerate(wcag_tasks, 1):
        print(f"\n   Task {i}: {task['description']}")
        print(f"   Prompt preview: {task['prompt'][:80]}...")

    print(f"\n2. Simple Description Tasks ({len(simple_tasks)} tasks)")
    print("   Focus: Basic image description and object identification")
    for i, task in enumerate(simple_tasks, 1):
        print(f"\n   Task {i}: {task['description']}")
        print(f"   Prompt: {task['prompt']}")

    print(f"\n3. Detailed Analysis Tasks ({len(detailed_tasks)} tasks)")
    print("   Focus: Comprehensive historical and accessibility analysis")
    for i, task in enumerate(detailed_tasks, 1):
        print(f"\n   Task {i}: {task['description']}")
        print(f"   Prompt preview: {task['prompt'][:80]}...")

    print(f"\n4. All Tasks Combined: {len(all_tasks)} total tasks")

    # Show CLI commands
    print("\n\n💻 CLI Commands:")
    print("-" * 70)
    print("""
# List available VLM models
uv run python -m src.benchmarking list-models

# Benchmark a single model with WCAG tasks
uv run python -m src.benchmarking benchmark \\
    --model "openai/gpt-4-vision-preview" \\
    --task-set wcag \\
    --output results.json \\
    --verbose

# Compare multiple models
uv run python -m src.benchmarking benchmark-multiple \\
    --models "model1,model2,model3" \\
    --task-set all \\
    --output comparison.json
    """)

    # Show Python API
    print("\n\n🐍 Python API:")
    print("-" * 70)
    print("""
from src.benchmarking import list_models, benchmark_model, aggregate_results
from src.benchmarking.tasks import get_wcag_alttext_tasks

# List models
models = list_models(api_key)

# Benchmark single model
tasks = get_wcag_alttext_tasks()
results = benchmark_model("model-name", tasks, api_key)

# Compare multiple models
all_results = [
    benchmark_model(model, tasks, api_key) 
    for model in ["model1", "model2"]
]
summary = aggregate_results(all_results)
    """)

    # Show expected results structure
    print("\n\n📊 Expected Results Structure:")
    print("-" * 70)
    print("""
benchmark_model() returns:
{
  "model_name": str,
  "total_tasks": int,
  "successful_tasks": int,
  "failed_tasks": int,
  "success_rate": float (0.0-1.0),
  "total_response_time_ms": float,
  "avg_response_time_ms": float,
  "total_prompt_tokens": int,
  "total_completion_tokens": int,
  "total_tokens": int,
  "task_results": [
    {
      "task_index": int,
      "prompt": str,
      "response": str,
      "response_time_ms": float,
      "tokens_used": dict,
      "success": bool,
      "error": str | None
    },
    ...
  ]
}

aggregate_results() returns:
{
  "total_models": int,
  "total_tasks": int,
  "total_successful_tasks": int,
  "overall_success_rate": float,
  "fastest_model": str,
  "slowest_model": str,
  "models_summary": [
    {
      "model_name": str,
      "success_rate": float,
      "avg_response_time_ms": float,
      "total_tokens": int
    },
    ...
  ]
}
    """)

    print("\n" + "=" * 70)
    print("✓ Demo completed successfully!")
    print("=" * 70)
    print("\nFor full documentation, see: src/benchmarking/README.md")


if __name__ == "__main__":
    main()
