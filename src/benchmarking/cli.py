#!/usr/bin/env python3
"""
Command-line interface for OpenRouter VLM benchmarking.

Usage:
    python -m src.benchmarking.cli list-models
    python -m src.benchmarking.cli benchmark --model MODEL_NAME --task-set wcag
    python -m src.benchmarking.cli benchmark-multiple --models model1,model2 --task-set all
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from .openrouter_benchmark import aggregate_results, benchmark_model, list_models
from .tasks import (
    get_all_tasks,
    get_detailed_analysis_tasks,
    get_simple_description_tasks,
    get_wcag_alttext_tasks,
)


def get_api_key() -> str:
    """
    Get OpenRouter API key from environment.

    Returns:
        API key string

    Raises:
        ValueError: If API key is not found
    """
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found. Please set it in your .env file or environment."
        )
    return api_key


def get_task_set(task_set_name: str) -> List[Dict[str, Any]]:
    """
    Get tasks based on the specified task set name.

    Args:
        task_set_name: Name of the task set (wcag, simple, detailed, all)

    Returns:
        List of task dictionaries

    Raises:
        ValueError: If task set name is invalid
    """
    task_sets = {
        "wcag": get_wcag_alttext_tasks,
        "simple": get_simple_description_tasks,
        "detailed": get_detailed_analysis_tasks,
        "all": get_all_tasks,
    }

    if task_set_name not in task_sets:
        raise ValueError(
            f"Invalid task set: {task_set_name}. "
            f"Choose from: {', '.join(task_sets.keys())}"
        )

    return task_sets[task_set_name]()


def cmd_list_models(args: argparse.Namespace) -> int:
    """
    List available VLM models.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success)
    """
    try:
        api_key = get_api_key()
        models = list_models(api_key, filter_vllm=not args.all_models)

        print(f"Found {len(models)} models:")
        for model in models:
            print(f"  - {model}")

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """
    Benchmark a single model.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success)
    """
    try:
        api_key = get_api_key()
        tasks = get_task_set(args.task_set)

        print(f"Benchmarking model: {args.model}")
        print(f"Task set: {args.task_set} ({len(tasks)} tasks)")
        print("-" * 60)

        results = benchmark_model(args.model, tasks, api_key, args.max_tokens)

        print(f"\nResults for {results['model_name']}:")
        print(f"  Total tasks: {results['total_tasks']}")
        print(f"  Successful: {results['successful_tasks']}")
        print(f"  Failed: {results['failed_tasks']}")
        print(f"  Success rate: {results['success_rate']:.1%}")
        print(
            f"  Average response time: {results['avg_response_time_ms']:.2f}ms"
        )
        print(f"  Total tokens used: {results['total_tokens']}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nDetailed results saved to: {args.output}")

        if args.verbose:
            print("\nTask-by-task results:")
            for task_result in results["task_results"]:
                print(f"\n  Task {task_result['task_index']}:")
                print(f"    Prompt: {task_result['prompt'][:80]}...")
                if task_result["success"]:
                    print(
                        f"    Response: {task_result['response'][:100]}..."
                    )
                    print(
                        f"    Time: {task_result['response_time_ms']:.2f}ms"
                    )
                else:
                    print(f"    Error: {task_result['error']}")

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_benchmark_multiple(args: argparse.Namespace) -> int:
    """
    Benchmark multiple models and compare results.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success)
    """
    try:
        api_key = get_api_key()
        tasks = get_task_set(args.task_set)
        model_names = [m.strip() for m in args.models.split(",")]

        print(f"Benchmarking {len(model_names)} models")
        print(f"Task set: {args.task_set} ({len(tasks)} tasks)")
        print("-" * 60)

        all_results = []
        for model_name in model_names:
            print(f"\nBenchmarking: {model_name}")
            results = benchmark_model(model_name, tasks, api_key, args.max_tokens)
            all_results.append(results)

            print(f"  Success rate: {results['success_rate']:.1%}")
            print(
                f"  Avg response time: {results['avg_response_time_ms']:.2f}ms"
            )

        # Aggregate and display summary
        summary = aggregate_results(all_results)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total models benchmarked: {summary['total_models']}")
        print(f"Overall success rate: {summary['overall_success_rate']:.1%}")
        print(f"Fastest model: {summary['fastest_model']}")
        print(f"Slowest model: {summary['slowest_model']}")

        print("\nModel Rankings (by speed):")
        for i, model_info in enumerate(summary["models_summary"], 1):
            print(
                f"  {i}. {model_info['model_name']}: "
                f"{model_info['avg_response_time_ms']:.2f}ms "
                f"(success: {model_info['success_rate']:.1%})"
            )

        if args.output:
            output_data = {
                "summary": summary,
                "detailed_results": all_results,
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nDetailed results saved to: {args.output}")

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """
    Main CLI entry point.

    Returns:
        Exit code (0 for success)
    """
    parser = argparse.ArgumentParser(
        description="Benchmark VLM models via OpenRouter API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List models command
    list_parser = subparsers.add_parser(
        "list-models", help="List available VLM models"
    )
    list_parser.add_argument(
        "--all-models",
        action="store_true",
        help="Show all models, not just vision models",
    )

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Benchmark a single model"
    )
    benchmark_parser.add_argument(
        "--model", required=True, help="Model identifier to benchmark"
    )
    benchmark_parser.add_argument(
        "--task-set",
        default="wcag",
        choices=["wcag", "simple", "detailed", "all"],
        help="Task set to use (default: wcag)",
    )
    benchmark_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens for responses (default: 1000)",
    )
    benchmark_parser.add_argument(
        "--output", "-o", help="Save detailed results to JSON file"
    )
    benchmark_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )

    # Benchmark multiple command
    multiple_parser = subparsers.add_parser(
        "benchmark-multiple", help="Benchmark multiple models and compare"
    )
    multiple_parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated list of model identifiers",
    )
    multiple_parser.add_argument(
        "--task-set",
        default="wcag",
        choices=["wcag", "simple", "detailed", "all"],
        help="Task set to use (default: wcag)",
    )
    multiple_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum tokens for responses (default: 1000)",
    )
    multiple_parser.add_argument(
        "--output", "-o", help="Save detailed results to JSON file"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "list-models":
        return cmd_list_models(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    elif args.command == "benchmark-multiple":
        return cmd_benchmark_multiple(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
