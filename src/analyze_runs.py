#!/usr/bin/env python3

"""Analyse alt-text generation runs for descriptive statistics."""

import argparse
import sys
from pathlib import Path

import pandas as pd


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
        min_latency=("latency_seconds", "min"),
        max_latency=("latency_seconds", "max"),
        mean_latency=("latency_seconds", "mean"),
        median_latency=("latency_seconds", "median"),
    )

    # Calculate coverage and throughput
    summary["coverage_pct"] = (
        (summary["non_empty_responses"] / summary["total_requests"]) * 100
    ).round(2)
    summary["throughput_items_per_sec"] = (1 / summary["mean_latency"]).round(2)

    # Reorder and select columns for the final output
    summary = summary[
        [
            "total_requests",
            "successful_responses",
            "non_empty_responses",
            "coverage_pct",
            "throughput_items_per_sec",
            "total_cost",
            "mean_cost",
            "median_cost",
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


def main() -> None:
    """Main function to parse arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description="Calculate descriptive statistics from alt-text generation runs."
    )
    parser.add_argument(
        "run_csv",
        type=Path,
        help="Path to the alt_text_runs_*_long.csv file from a generation run.",
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
