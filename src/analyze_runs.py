#!/usr/bin/env python3

"""Analyse alt-text generation runs for descriptive statistics."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def create_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Creates boxplots for throughput, latency, and cost metrics.

    Args:
        df: The DataFrame containing run data.
        output_dir: Directory to save the boxplot images.
    """
    # Calculate throughput per item
    df_plot = df.copy()
    df_plot["throughput_items_per_sec"] = 1 / df_plot["latency_seconds"]

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Throughput boxplot
    df_plot.boxplot(
        column="throughput_items_per_sec",
        by="model",
        ax=axes[0],
        grid=False,
    )
    axes[0].set_title("Throughput by Model")
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Items per Second")
    axes[0].tick_params(axis="x", rotation=45)

    # Latency boxplot
    df_plot.boxplot(
        column="latency_seconds",
        by="model",
        ax=axes[1],
        grid=False,
    )
    axes[1].set_title("Latency by Model")
    axes[1].set_xlabel("Model")
    axes[1].set_ylabel("Seconds")
    axes[1].tick_params(axis="x", rotation=45)

    # Cost boxplot
    df_plot.boxplot(
        column="usage_cost",
        by="model",
        ax=axes[2],
        grid=False,
    )
    axes[2].set_title("Cost by Model")
    axes[2].set_xlabel("Model")
    axes[2].set_ylabel("USD")
    axes[2].tick_params(axis="x", rotation=45)

    # Remove the automatic suptitle from pandas
    plt.suptitle("")

    # Adjust layout and save
    plt.tight_layout()
    boxplot_path = output_dir / "run_analysis_boxplots.png"
    plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Boxplots saved to: {boxplot_path}")


def analyze_run(run_csv_path: Path, output_dir: Path) -> None:
    """
    Analyzes a long-format run CSV to calculate descriptive statistics for each model.

    Args:
        run_csv_path: The path to the alt_text_runs_*_long.csv file.
        output_dir: Directory to save the analysis summary.
    """
    if not run_csv_path.exists():
        print(f"Error: File not found at {run_csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(run_csv_path)

    # Ensure required columns exist
    required_cols = [
        "model",
        "content",
        "error",
        "latency_seconds",
        "usage_cost",
    ]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(
            f"Error: Missing required columns in CSV: {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    summary = df.groupby("model").agg(
        total_requests=("model", "size"),
        successful_responses=("error", lambda x: x.isna().sum()),
        non_empty_responses=(
            "content",
            lambda x: (x.notna() & (x != "")).sum(),
        ),
        total_cost=("usage_cost", "sum"),
        mean_cost=("usage_cost", "mean"),
        median_cost=("usage_cost", "median"),
        min_cost=("usage_cost", "min"),
        max_cost=("usage_cost", "max"),
        min_latency=("latency_seconds", "min"),
        max_latency=("latency_seconds", "max"),
        mean_latency=("latency_seconds", "mean"),
        median_latency=("latency_seconds", "median"),
    )

    # Calculate coverage and throughput statistics
    summary["coverage_pct"] = (
        (summary["non_empty_responses"] / summary["total_requests"]) * 100
    ).round(2)
    summary["mean_throughput_items_per_sec"] = (1 / summary["mean_latency"]).round(2)
    summary["median_throughput_items_per_sec"] = (1 / summary["median_latency"]).round(
        2
    )
    summary["min_throughput_items_per_sec"] = (1 / summary["max_latency"]).round(2)
    summary["max_throughput_items_per_sec"] = (1 / summary["min_latency"]).round(2)

    # Reorder and select columns for the final output
    summary = summary[
        [
            "total_requests",
            "successful_responses",
            "non_empty_responses",
            "coverage_pct",
            "mean_throughput_items_per_sec",
            "median_throughput_items_per_sec",
            "min_throughput_items_per_sec",
            "max_throughput_items_per_sec",
            "total_cost",
            "mean_cost",
            "median_cost",
            "min_cost",
            "max_cost",
            "mean_latency",
            "median_latency",
            "min_latency",
            "max_latency",
        ]
    ]

    print("--- Model Performance Summary ---")
    print(summary)

    # Save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run_analysis.csv"
    summary.to_csv(summary_path)

    print(f"\nAnalysis summary saved to: {summary_path}")

    # Create boxplots for throughput, latency, and cost
    create_boxplots(df, output_dir)


def main() -> None:
    """Main function to parse arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description="Calculate descriptive statistics from alt-text generation runs."
    )
    parser.add_argument(
        "run_csv",
        type=Path,
        nargs="?",
        default=Path("runs/20251021_233530/alt_text_runs_20251021_233933_long.csv"),
        help="Path to the alt_text_runs_*_long.csv file from a generation run (default: most recent run).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./analysis"),
        help="Directory to save the analysis summary CSV file (default: ./analysis).",
    )
    args = parser.parse_args()

    analyze_run(args.run_csv, args.output_dir)


if __name__ == "__main__":
    main()
