"""
Analyze survey completion times by objectid and submission_email.

This script reads the processed survey rankings CSV and calculates
min, median, max, and mean time elapsed for each objectid and submission_email.
"""

import pandas as pd
from pathlib import Path


def analyze_time_by_objectid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time statistics grouped by objectid.

    Args:
        df: DataFrame with survey rankings

    Returns:
        DataFrame with time statistics per objectid
    """
    stats = df.groupby("objectid")["time_elapsed_s"].agg(
        min_time="min",
        median_time="median",
        max_time="max",
        mean_time="mean",
        count="count",
    )
    return stats.sort_values("mean_time", ascending=False)


def analyze_time_by_submission(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time statistics grouped by submission_seed.

    Args:
        df: DataFrame with survey rankings

    Returns:
        DataFrame with time statistics per submission
    """
    stats = df.groupby("submission_seed")["time_elapsed_s"].agg(
        min_time="min",
        median_time="median",
        max_time="max",
        mean_time="mean",
        count="count",
    )
    return stats.sort_values("mean_time", ascending=False)


def main():
    """Main function to analyze survey time data."""
    # Input CSV file
    input_path = (
        Path(__file__).parent.parent / "data" / "processed" / "survey_rankings.csv"
    )

    print(f"Reading data from: {input_path}")
    df = pd.read_csv(input_path)

    print(f"\nTotal records: {len(df)}")
    print(f"Unique objects: {df['objectid'].nunique()}")
    print(f"Unique submissions: {df['submission_seed'].nunique()}")

    # Overall statistics
    print("\n" + "=" * 70)
    print("OVERALL TIME STATISTICS")
    print("=" * 70)
    print(f"Min time: {df['time_elapsed_s'].min():.2f}s")
    print(f"Median time: {df['time_elapsed_s'].median():.2f}s")
    print(f"Max time: {df['time_elapsed_s'].max():.2f}s")
    print(f"Mean time: {df['time_elapsed_s'].mean():.2f}s")

    # By objectid
    print("\n" + "=" * 70)
    print("TIME STATISTICS BY OBJECTID")
    print("=" * 70)
    by_object = analyze_time_by_objectid(df)
    print(by_object.to_string())

    # By submission
    print("\n" + "=" * 70)
    print("TIME STATISTICS BY SUBMISSION SEED")
    print("=" * 70)
    by_submission = analyze_time_by_submission(df)
    print(by_submission.to_string())

    # Save results
    output_dir = Path(__file__).parent.parent / "analysis"
    output_dir.mkdir(exist_ok=True)

    by_object.to_csv(output_dir / "time_stats_by_object.csv")
    by_submission.to_csv(output_dir / "time_stats_by_submission.csv")

    print("\n\nResults saved to:")
    print(f"  - {output_dir / 'time_stats_by_object.csv'}")
    print(f"  - {output_dir / 'time_stats_by_submission.csv'}")


if __name__ == "__main__":
    main()
