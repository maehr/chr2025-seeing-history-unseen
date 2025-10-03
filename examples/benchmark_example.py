#!/usr/bin/env python3
"""
Example script demonstrating OpenRouter VLM benchmarking.

This script shows how to use the benchmarking module programmatically.
Note: This is a demonstration script. To actually run it, you need:
1. A valid OPENROUTER_API_KEY in your .env file
2. The required Python dependencies installed (requests, python-dotenv)

Usage:
    python examples/benchmark_example.py
"""

import os
import sys
from typing import Any, Dict, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.benchmarking import aggregate_results, benchmark_model, list_models
from src.benchmarking.tasks import get_wcag_alttext_tasks


def main() -> None:
    """Run example benchmarking workflow."""
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY not found in environment")
        print("\nTo run this example:")
        print("1. Copy example.env to .env")
        print("2. Add your OpenRouter API key to .env")
        print("3. Run this script again")
        return

    print("=" * 70)
    print("OpenRouter VLM Benchmarking Example")
    print("=" * 70)

    # Example 1: List available vision models
    print("\n📋 Example 1: Listing available vision models...")
    print("-" * 70)
    try:
        models = list_models(api_key, filter_vllm=True)
        print(f"✓ Found {len(models)} vision-language models")
        print(f"\nFirst 5 models:")
        for model in models[:5]:
            print(f"  • {model}")
        if len(models) > 5:
            print(f"  ... and {len(models) - 5} more")
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return

    # Example 2: Benchmark a single model
    print("\n\n🔬 Example 2: Benchmarking a single model...")
    print("-" * 70)

    # Get WCAG alt-text tasks
    tasks = get_wcag_alttext_tasks()
    print(f"Using {len(tasks)} WCAG-compliant alt-text tasks")

    # For demonstration, we'll use a hypothetical model
    # In practice, replace with an actual model from the list
    example_model = "openai/gpt-4-vision-preview"  # Example model
    print(f"\nNote: To actually run benchmarks, you would use:")
    print(f"  python -m src.benchmarking benchmark --model {example_model}")

    print("\n📊 Example benchmark results structure:")
    print(
        """
{
  "model_name": "openai/gpt-4-vision-preview",
  "total_tasks": 5,
  "successful_tasks": 5,
  "failed_tasks": 0,
  "success_rate": 1.0,
  "total_response_time_ms": 6172.83,
  "avg_response_time_ms": 1234.57,
  "total_prompt_tokens": 850,
  "total_completion_tokens": 420,
  "total_tokens": 1270,
  "task_results": [
    {
      "task_index": 0,
      "prompt": "Generate WCAG-compliant alt-text...",
      "response": "Historical street scene, Zurich 1920s...",
      "response_time_ms": 1234.56,
      "tokens_used": {"prompt_tokens": 170, "completion_tokens": 84},
      "success": true,
      "error": null
    },
    ...
  ]
}
    """
    )

    # Example 3: Compare multiple models
    print("\n\n🏆 Example 3: Comparing multiple models...")
    print("-" * 70)
    print("To compare multiple models, use:")
    print(
        "  python -m src.benchmarking benchmark-multiple "
        '--models "model1,model2,model3" --output comparison.json'
    )

    print("\n📊 Example comparison results structure:")
    print(
        """
{
  "total_models": 3,
  "total_tasks": 15,
  "total_successful_tasks": 14,
  "overall_success_rate": 0.933,
  "fastest_model": "anthropic/claude-3-opus",
  "slowest_model": "openai/gpt-4-vision-preview",
  "models_summary": [
    {
      "model_name": "anthropic/claude-3-opus",
      "success_rate": 1.0,
      "avg_response_time_ms": 987.65,
      "total_tokens": 1150
    },
    ...
  ]
}
    """
    )

    # Example 4: Task sets
    print("\n\n📝 Example 4: Available task sets...")
    print("-" * 70)
    from src.benchmarking.tasks import (
        get_all_tasks,
        get_detailed_analysis_tasks,
        get_simple_description_tasks,
    )

    print(f"• WCAG alt-text tasks: {len(get_wcag_alttext_tasks())} tasks")
    print(f"• Simple description tasks: {len(get_simple_description_tasks())} tasks")
    print(
        f"• Detailed analysis tasks: {len(get_detailed_analysis_tasks())} tasks"
    )
    print(f"• All tasks combined: {len(get_all_tasks())} tasks")

    print("\nExample WCAG task:")
    wcag_task = get_wcag_alttext_tasks()[0]
    print(f"  Prompt: {wcag_task['prompt'][:100]}...")
    print(f"  Description: {wcag_task['description']}")

    # Summary
    print("\n\n" + "=" * 70)
    print("💡 Next Steps")
    print("=" * 70)
    print("""
1. Set up your OpenRouter API key in .env file
2. List available models:
   python -m src.benchmarking list-models

3. Benchmark a single model:
   python -m src.benchmarking benchmark --model MODEL_NAME --task-set wcag

4. Compare multiple models:
   python -m src.benchmarking benchmark-multiple --models "model1,model2"

5. View detailed documentation:
   cat src/benchmarking/README.md
    """)

    print("=" * 70)
    print("Example completed successfully! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
