#!/usr/bin/env python3
"""
Command-line interface for OpenRouter VLM benchmarking.

Usage:
    python -m src.benchmarking list-models
    python -m src.benchmarking benchmark --model MODEL_NAME --task-set wcag
    python -m src.benchmarking benchmark-multiple --models model1,model2 --task-set all
"""

import json
import os
import sys
from typing import Any, Dict, List

import click
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


@click.group()
def cli() -> None:
    """Benchmark VLM models via OpenRouter API."""
    pass


@cli.command("list-models")
@click.option(
    "--all-models",
    is_flag=True,
    help="Show all models, not just vision models",
)
def list_models_cmd(all_models: bool) -> None:
    """List available VLM models."""
    try:
        api_key = get_api_key()
        models = list_models(api_key, filter_vllm=not all_models)

        click.echo(f"Found {len(models)} models:")
        for model in models:
            click.echo(f"  - {model}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--model", required=True, help="Model identifier to benchmark")
@click.option(
    "--task-set",
    type=click.Choice(["wcag", "simple", "detailed", "all"]),
    default="wcag",
    help="Task set to use (default: wcag)",
)
@click.option(
    "--max-tokens",
    type=int,
    default=1000,
    help="Maximum tokens for responses (default: 1000)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Save detailed results to JSON file",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def benchmark(
    model: str, task_set: str, max_tokens: int, output: str, verbose: bool
) -> None:
    """Benchmark a single model."""
    try:
        api_key = get_api_key()
        tasks = get_task_set(task_set)

        click.echo(f"Benchmarking model: {model}")
        click.echo(f"Task set: {task_set} ({len(tasks)} tasks)")
        click.echo("-" * 60)

        results = benchmark_model(model, tasks, api_key, max_tokens)

        click.echo(f"\nResults for {results['model_name']}:")
        click.echo(f"  Total tasks: {results['total_tasks']}")
        click.echo(f"  Successful: {results['successful_tasks']}")
        click.echo(f"  Failed: {results['failed_tasks']}")
        click.echo(f"  Success rate: {results['success_rate']:.1%}")
        click.echo(
            f"  Average response time: {results['avg_response_time_ms']:.2f}ms"
        )
        click.echo(f"  Total tokens used: {results['total_tokens']}")

        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            click.echo(f"\nDetailed results saved to: {output}")

        if verbose:
            click.echo("\nTask-by-task results:")
            for task_result in results["task_results"]:
                click.echo(f"\n  Task {task_result['task_index']}:")
                click.echo(f"    Prompt: {task_result['prompt'][:80]}...")
                if task_result["success"]:
                    click.echo(f"    Response: {task_result['response'][:100]}...")
                    click.echo(f"    Time: {task_result['response_time_ms']:.2f}ms")
                else:
                    click.echo(f"    Error: {task_result['error']}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("benchmark-multiple")
@click.option(
    "--models",
    required=True,
    help="Comma-separated list of model identifiers",
)
@click.option(
    "--task-set",
    type=click.Choice(["wcag", "simple", "detailed", "all"]),
    default="wcag",
    help="Task set to use (default: wcag)",
)
@click.option(
    "--max-tokens",
    type=int,
    default=1000,
    help="Maximum tokens for responses (default: 1000)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Save detailed results to JSON file",
)
def benchmark_multiple(
    models: str, task_set: str, max_tokens: int, output: str
) -> None:
    """Benchmark multiple models and compare."""
    try:
        api_key = get_api_key()
        tasks = get_task_set(task_set)
        model_names = [m.strip() for m in models.split(",")]

        click.echo(f"Benchmarking {len(model_names)} models")
        click.echo(f"Task set: {task_set} ({len(tasks)} tasks)")
        click.echo("-" * 60)

        all_results = []
        for model_name in model_names:
            click.echo(f"\nBenchmarking: {model_name}")
            results = benchmark_model(model_name, tasks, api_key, max_tokens)
            all_results.append(results)

            click.echo(f"  Success rate: {results['success_rate']:.1%}")
            click.echo(f"  Avg response time: {results['avg_response_time_ms']:.2f}ms")

        # Aggregate and display summary
        summary = aggregate_results(all_results)

        click.echo("\n" + "=" * 60)
        click.echo("SUMMARY")
        click.echo("=" * 60)
        click.echo(f"Total models benchmarked: {summary['total_models']}")
        click.echo(f"Overall success rate: {summary['overall_success_rate']:.1%}")
        click.echo(f"Fastest model: {summary['fastest_model']}")
        click.echo(f"Slowest model: {summary['slowest_model']}")

        click.echo("\nModel Rankings (by speed):")
        for i, model_info in enumerate(summary["models_summary"], 1):
            click.echo(
                f"  {i}. {model_info['model_name']}: "
                f"{model_info['avg_response_time_ms']:.2f}ms "
                f"(success: {model_info['success_rate']:.1%})"
            )

        if output:
            output_data = {
                "summary": summary,
                "detailed_results": all_results,
            }
            with open(output, "w") as f:
                json.dump(output_data, f, indent=2)
            click.echo(f"\nDetailed results saved to: {output}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
