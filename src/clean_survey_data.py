"""
Script to clean survey data by removing specific submissions and email addresses.

This script:
1. Removes submissions with seed values 1316969232 and 1723420473
2. Removes all email addresses from the data (seed remains as unique identifier)
"""

import json
from pathlib import Path


def clean_survey_data(
    input_path: Path,
    output_path: Path,
    seeds_to_remove: list[str],
) -> dict[str, int]:
    """
    Clean survey data by removing specific submissions and email addresses.

    Args:
        input_path: Path to the input JSON file
        output_path: Path to save the cleaned JSON file
        seeds_to_remove: List of seed values to remove

    Returns:
        Dictionary with statistics about the cleaning process
    """
    # Load the data
    with input_path.open() as f:
        data = json.load(f)

    # Track statistics
    original_submission_count = len(data["submissions"])
    emails_removed = 0

    # Remove the top-level email field
    if "email" in data:
        del data["email"]

    # Remove email from fields list if present
    if "fields" in data and "email" in data["fields"]:
        data["fields"] = [field for field in data["fields"] if field != "email"]

    # Filter submissions and remove emails within them
    filtered_submissions = []
    for submission in data["submissions"]:
        # Skip submissions with seeds to remove
        if submission.get("seed") in seeds_to_remove:
            continue

        # Remove the email field from the submission
        if "email" in submission:
            del submission["email"]
            emails_removed += 1

        filtered_submissions.append(submission)

    # Update the data with filtered submissions
    data["submissions"] = filtered_submissions

    # Save the cleaned data
    with output_path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Return statistics
    return {
        "original_submissions": original_submission_count,
        "filtered_submissions": len(filtered_submissions),
        "removed_submissions": original_submission_count - len(filtered_submissions),
        "emails_removed": emails_removed,
    }


def main():
    """Main entry point for the script."""
    # Define paths
    input_file = (
        Path(__file__).parent.parent
        / "data"
        / "raw"
        / "formspree_xrbyjror_2025-10-27T12_40_38_export.json"
    )
    output_file = (
        Path(__file__).parent.parent
        / "data"
        / "processed"
        / "processed_survey_submissions.json"
    )

    # Seeds to remove
    seeds_to_remove = ["1316969232", "1723420473"]

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Clean the data
    print(f"Processing: {input_file}")
    print(f"Removing submissions with seeds: {', '.join(seeds_to_remove)}")

    stats = clean_survey_data(input_file, output_file, seeds_to_remove)

    # Print statistics
    print("\nCleaning complete!")
    print(f"Original submissions: {stats['original_submissions']}")
    print(f"Filtered submissions: {stats['filtered_submissions']}")
    print(f"Removed submissions: {stats['removed_submissions']}")
    print(f"Emails removed: {stats['emails_removed']}")
    print(f"\nCleaned data saved to: {output_file}")


if __name__ == "__main__":
    main()
