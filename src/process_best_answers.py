"""
Process Formspree survey submissions and extract aggregated rankings for each object.

This script reads the JSON export from Formspree and produces a CSV/DataFrame
with objectid, best ranked model, mean rank, median rank, and best ranked answer.
"""

import json
import pandas as pd
from pathlib import Path


def extract_best_answers(json_path: str | Path) -> pd.DataFrame:
    """
    Extract aggregated ranking statistics for each unique object.

    Args:
        json_path: Path to the JSON file

    Returns:
        DataFrame with columns: objectid, best_ranked_model, mean_rank, median_rank, best_ranked_answer
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect all rankings for each object
    object_data = {}

    # Process each submission
    for submission in data["submissions"]:
        # Process each answer (object) in the submission
        for answer in submission.get("answers", []):
            objectid = answer.get("objectid", "")
            final_order_ids = answer.get("final_order_ids", {})
            final_order_texts = answer.get("final_order_texts", {})

            if objectid not in object_data:
                object_data[objectid] = {
                    "model_ranks": {},  # model -> list of ranks
                    "rank_1_models": [],  # list of models that got rank 1
                    "rank_1_texts": {},  # model -> text when it got rank 1
                }

            # Process each model's rank in this submission
            for rank_str, model_id in final_order_ids.items():
                rank = int(rank_str)

                # Parse model name
                model_parts = model_id.replace("__content", "").split("__")
                if len(model_parts) >= 2:
                    model_name = "/".join(model_parts[:2])
                else:
                    model_name = model_id

                # Store the rank for this model
                if model_name not in object_data[objectid]["model_ranks"]:
                    object_data[objectid]["model_ranks"][model_name] = []
                object_data[objectid]["model_ranks"][model_name].append(rank)

                # If this is rank 1, track it
                if rank == 1:
                    object_data[objectid]["rank_1_models"].append(model_name)
                    if rank_str in final_order_texts:
                        object_data[objectid]["rank_1_texts"][model_name] = (
                            final_order_texts[rank_str]
                        )

    # Aggregate statistics for each object
    records = []
    for objectid, data in object_data.items():
        # Find the model with the best (lowest) mean rank
        model_mean_ranks = {
            model: sum(ranks) / len(ranks)
            for model, ranks in data["model_ranks"].items()
        }
        best_model = min(model_mean_ranks.items(), key=lambda x: x[1])[0]
        mean_rank = model_mean_ranks[best_model]

        # Calculate median rank for the best model
        best_model_ranks = sorted(data["model_ranks"][best_model])
        n = len(best_model_ranks)
        if n % 2 == 0:
            median_rank = (best_model_ranks[n // 2 - 1] + best_model_ranks[n // 2]) / 2
        else:
            median_rank = best_model_ranks[n // 2]

        # Count occurrences of each rank (1st, 2nd, 3rd, 4th) for the best model
        rank_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for rank in data["model_ranks"][best_model]:
            rank_counts[rank] += 1

        # Get the answer text for the best model (use first occurrence when it was rank 1)
        best_answer = data["rank_1_texts"].get(best_model, "")

        record = {
            "objectid": objectid,
            "best_ranked_model": best_model,
            "mean_rank": round(mean_rank, 2),
            "median_rank": median_rank,
            "count_rank_1": rank_counts[1],
            "count_rank_2": rank_counts[2],
            "count_rank_3": rank_counts[3],
            "count_rank_4": rank_counts[4],
            "best_ranked_answer": best_answer,
        }

        records.append(record)

    # Convert to DataFrame and sort by objectid
    df = pd.DataFrame(records)
    df = df.sort_values("objectid").reset_index(drop=True)

    return df


def main():
    """Main function to process the survey data."""
    # Input JSON file
    json_path = (
        Path(__file__).parent.parent
        / "data"
        / "raw"
        / "formspree_xrbyjror_2025-10-27T08_46_50_export.json"
    )

    # Output CSV file
    output_path = (
        Path(__file__).parent.parent / "data" / "processed" / "best_answers.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading data from: {json_path}")
    df = extract_best_answers(json_path)

    print(f"\nExtracted aggregated rankings for {len(df)} unique objects")
    print("\nBest ranked model distribution:")
    print(df["best_ranked_model"].value_counts())
    print("\nFirst few rows:")
    print(df.head(10))

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nSaved aggregated rankings to: {output_path}")

    return df


if __name__ == "__main__":
    df = main()
