"""
Process Formspree survey submissions and extract model rankings.

This script reads the JSON export from Formspree and produces a CSV/DataFrame
with objectid, submission email, and the rank for each model.
"""

import json
import pandas as pd
from pathlib import Path


def extract_rankings(json_path: str | Path) -> pd.DataFrame:
    """
    Extract rankings from Formspree JSON export.

    Args:
        json_path: Path to the JSON file

    Returns:
        DataFrame with columns: objectid, submission_seed, time_elapsed_s, and rank_* for each model
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    # Process each submission
    for submission in data["submissions"]:
        submission_seed = submission.get("seed", "")

        # Process each answer (object) in the submission
        for answer in submission.get("answers", []):
            objectid = answer.get("objectid", "")
            final_order_ids = answer.get("final_order_ids", {})

            # Calculate time elapsed
            time_initial_str = answer.get("timestamp_initial_utc")
            time_final_str = answer.get("timestamp_final_utc")
            time_elapsed_s = None
            if time_initial_str and time_final_str:
                time_initial = pd.to_datetime(time_initial_str)
                time_final = pd.to_datetime(time_final_str)
                time_elapsed_s = (time_final - time_initial).total_seconds()

            # Create a record with the objectid and seed
            record = {
                "objectid": objectid,
                "submission_seed": submission_seed,
                "time_elapsed_s": time_elapsed_s,
            }

            # Extract model names and their ranks from final_order_ids
            # final_order_ids maps rank (as string) to model_id
            for rank_str, model_id in final_order_ids.items():
                # Parse model name from model_id format: "provider__model__content"
                # Example: "google__gemini-2.5-flash-lite__content"
                model_parts = model_id.replace("__content", "").split("__")
                if len(model_parts) >= 2:
                    model_name = "/".join(
                        model_parts[:2]
                    )  # "google/gemini-2.5-flash-lite"
                    rank_column = f"rank_{model_name}"
                    record[rank_column] = int(rank_str)

            records.append(record)

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Sort columns: objectid, submission_seed, then rank columns alphabetically
    base_columns = ["objectid", "submission_seed", "time_elapsed_s"]
    rank_columns = sorted([col for col in df.columns if col.startswith("rank_")])
    df = df[base_columns + rank_columns]

    return df


def main():
    """Main function to process the survey data."""
    # Input JSON file
    json_path = (
        Path(__file__).parent.parent
        / "data"
        / "processed"
        / "processed_survey_submissions.json"
    )

    # Output CSV file
    output_path = (
        Path(__file__).parent.parent / "data" / "processed" / "survey_rankings.csv"
    )

    print(f"Reading data from: {json_path}")
    df = extract_rankings(json_path)

    print(
        f"\nExtracted {len(df)} rankings from {len(df['submission_seed'].unique())} unique submissions"
    )
    print(f"\nColumns: {', '.join(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nSaved rankings to: {output_path}")

    return df


if __name__ == "__main__":
    df = main()
