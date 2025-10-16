"""
Image-based benchmarking for VLM models via OpenRouter API.

This module provides functionality for benchmarking vision-language models
on image understanding tasks, with results stored in pandas DataFrames.
"""

import base64
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


class ImageBenchmark:
    """
    Image-based benchmarking for VLM models.
    
    This class handles running benchmarks on vision-language models using
    actual image files, collecting results in a pandas DataFrame.
    """

    # Specific models to benchmark
    SELECTED_MODELS = [
        "google/gemini-2.0-flash-exp:free",
        "openai/gpt-4o-mini",
        "mistralai/pixtral-12b",
        "meta-llama/llama-3.2-90b-vision-instruct",
    ]

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str) -> None:
        """
        Initialize the image benchmark.

        Args:
            api_key: OpenRouter API key for authentication
        """
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.results: List[Dict[str, Any]] = []

    def encode_image(self, image_path: Path) -> str:
        """
        Encode an image file to base64.

        Args:
            image_path: Path to the image file

        Returns:
            Base64-encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_image_url(self, image_path: Path) -> str:
        """
        Get a data URL for an image.

        Args:
            image_path: Path to the image file

        Returns:
            Data URL for the image
        """
        # Determine MIME type from file extension
        extension = image_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(extension, "image/jpeg")
        
        encoded = self.encode_image(image_path)
        return f"data:{mime_type};base64,{encoded}"

    def run_model_on_image(
        self,
        model: str,
        image_path: Path,
        prompt: str,
        image_type: str,
        max_tokens: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run a single model on a single image with a prompt.

        Args:
            model: Model identifier
            image_path: Path to the image file
            prompt: Text prompt to send with the image
            image_type: Category of the image (e.g., "artwork", "photograph")
            max_tokens: Maximum tokens in the response

        Returns:
            Dictionary containing the result metadata
        """
        # Prepare the message with image
        image_url = self.get_image_url(image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ]

        # Make API request with timing
        start_time = time.time()
        try:
            url = f"{self.BASE_URL}/chat/completions"
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            response = requests.post(
                url, headers=self.headers, json=payload, timeout=120
            )
            response.raise_for_status()
            end_time = time.time()

            result_data = response.json()
            response_text = (
                result_data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            usage = result_data.get("usage", {})

            return {
                "model": model,
                "image_path": str(image_path),
                "image_name": image_path.name,
                "image_type": image_type,
                "prompt": prompt,
                "response": response_text,
                "response_time_s": end_time - start_time,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "success": True,
                "error": None,
            }
        except Exception as e:
            end_time = time.time()
            return {
                "model": model,
                "image_path": str(image_path),
                "image_name": image_path.name,
                "image_type": image_type,
                "prompt": prompt,
                "response": None,
                "response_time_s": end_time - start_time,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "success": False,
                "error": str(e),
            }

    def benchmark_images(
        self,
        image_paths: List[Path],
        image_types: List[str],
        prompt: str,
        models: Optional[List[str]] = None,
        max_tokens: int = 1000,
    ) -> pd.DataFrame:
        """
        Benchmark multiple models on multiple images.

        Args:
            image_paths: List of paths to image files
            image_types: List of image type categories (same length as image_paths)
            prompt: Text prompt to use for all images
            models: List of model identifiers (defaults to SELECTED_MODELS)
            max_tokens: Maximum tokens in responses

        Returns:
            pandas DataFrame with all benchmark results
        """
        if models is None:
            models = self.SELECTED_MODELS

        if len(image_paths) != len(image_types):
            raise ValueError("image_paths and image_types must have the same length")

        results = []

        # Iterate over each image and each model
        for image_path, image_type in zip(image_paths, image_types):
            for model in models:
                print(f"Testing {model} on {image_path.name}...")
                result = self.run_model_on_image(
                    model, image_path, prompt, image_type, max_tokens
                )
                results.append(result)

        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Add columns for human ranking (to be filled in later)
        df["expert_rank"] = None
        df["expert_notes"] = None
        
        return df

    def export_to_csv(self, df: pd.DataFrame, output_path: Path) -> None:
        """
        Export benchmark results to a CSV file.

        Args:
            df: DataFrame containing benchmark results
            output_path: Path where CSV file should be saved
        """
        df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")

    def compute_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute descriptive statistics from benchmark results with rankings.

        Args:
            df: DataFrame with benchmark results including expert_rank column

        Returns:
            DataFrame with statistics per model
        """
        # Filter to successful runs only
        successful = df[df["success"] == True]

        # Group by model and compute statistics
        stats = successful.groupby("model").agg(
            {
                "response_time_s": ["mean", "std", "min", "max"],
                "total_tokens": ["mean", "std", "min", "max"],
                "success": "count",  # Number of successful runs
                "expert_rank": ["mean", "std", "min", "max"],  # Ranking statistics
            }
        )

        # Flatten column names
        stats.columns = ["_".join(col).strip() for col in stats.columns.values]
        stats = stats.reset_index()

        # Rename columns for clarity
        stats = stats.rename(
            columns={
                "success_count": "total_runs",
                "response_time_s_mean": "avg_response_time_s",
                "response_time_s_std": "std_response_time_s",
                "response_time_s_min": "min_response_time_s",
                "response_time_s_max": "max_response_time_s",
                "total_tokens_mean": "avg_total_tokens",
                "total_tokens_std": "std_total_tokens",
                "total_tokens_min": "min_total_tokens",
                "total_tokens_max": "max_total_tokens",
                "expert_rank_mean": "avg_expert_rank",
                "expert_rank_std": "std_expert_rank",
                "expert_rank_min": "best_rank",
                "expert_rank_max": "worst_rank",
            }
        )

        return stats


# Image type categories
IMAGE_TYPES = [
    "artwork",
    "photograph_object",
    "photograph_archaeological_site",
    "scan_newspaper",
    "scan_poster",
    "drawing",
    "map",
    "statistical_figure",
    "diagram",
]


def get_default_prompt() -> str:
    """
    Get the default prompt for image understanding benchmarking.

    Returns:
        Default prompt string
    """
    return (
        "Please provide a detailed description of this image. "
        "Include information about what is depicted, the style or medium "
        "(if applicable), any text visible, and any historical or cultural "
        "context that can be inferred from the visual elements."
    )
