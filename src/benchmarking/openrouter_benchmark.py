"""
OpenRouter VLM Benchmarking Implementation

This module implements benchmarking functionality for vision-language models
available through the OpenRouter API.
"""

import json
import time
from typing import Any, Dict, List, Optional

import requests


class OpenRouterClient:
    """Client for interacting with the OpenRouter API."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str) -> None:
        """
        Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key for authentication
        """
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def get_models(self) -> List[Dict[str, Any]]:
        """
        Fetch list of available models from OpenRouter API.

        Returns:
            List of model information dictionaries

        Raises:
            requests.RequestException: If the API request fails
        """
        url = f"{self.BASE_URL}/models"
        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()
        return response.json().get("data", [])

    def generate_completion(
        self, model: str, messages: List[Dict[str, str]], max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate a completion using the specified model.

        Args:
            model: Model identifier
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens in the response

        Returns:
            API response dictionary containing the completion

        Raises:
            requests.RequestException: If the API request fails
        """
        url = f"{self.BASE_URL}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        response = requests.post(url, headers=self.headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()


def list_models(api_key: str, filter_vllm: bool = True) -> List[str]:
    """
    List available VLM models from OpenRouter API.

    Args:
        api_key: OpenRouter API key for authentication
        filter_vllm: If True, filter to only vision-language models

    Returns:
        List of model identifiers

    Example:
        >>> api_key = "your-api-key"
        >>> models = list_models(api_key)
        >>> print(models)
        ['anthropic/claude-3-opus', 'openai/gpt-4-vision-preview', ...]
    """
    client = OpenRouterClient(api_key)
    models = client.get_models()

    model_ids = []
    for model in models:
        model_id = model.get("id", "")
        # Filter for vision-language models if requested
        if filter_vllm:
            # Vision models typically have "vision" in the name or support vision
            supports_vision = (
                "vision" in model_id.lower()
                or "vision" in model.get("name", "").lower()
                or model.get("architecture", {}).get("modality") == "multimodal"
            )
            if supports_vision:
                model_ids.append(model_id)
        else:
            model_ids.append(model_id)

    return model_ids


def benchmark_model(
    model_name: str,
    tasks: List[Dict[str, Any]],
    api_key: str,
    max_tokens: int = 1000,
) -> Dict[str, Any]:
    """
    Benchmark a specific model on a set of tasks.

    Args:
        model_name: Model identifier to benchmark
        tasks: List of task dictionaries, each containing 'prompt' and optional 'image_url'
        api_key: OpenRouter API key for authentication
        max_tokens: Maximum tokens for model responses

    Returns:
        Dictionary containing benchmark results with timing and cost information

    Example:
        >>> tasks = [
        ...     {"prompt": "Describe this image", "image_url": "https://..."},
        ...     {"prompt": "What objects are in this image?"}
        ... ]
        >>> results = benchmark_model("openai/gpt-4-vision-preview", tasks, api_key)
        >>> print(f"Average response time: {results['avg_response_time_ms']:.2f}ms")
    """
    client = OpenRouterClient(api_key)
    task_results = []

    for i, task in enumerate(tasks):
        prompt = task.get("prompt", "")
        image_url = task.get("image_url")

        # Construct message based on whether image is provided
        if image_url:
            message_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        else:
            message_content = prompt

        messages = [{"role": "user", "content": message_content}]

        # Time the API call
        start_time = time.time()
        try:
            response = client.generate_completion(model_name, messages, max_tokens)
            end_time = time.time()

            response_time_ms = (end_time - start_time) * 1000
            response_text = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            tokens_used = response.get("usage", {})

            task_results.append(
                {
                    "task_index": i,
                    "prompt": prompt,
                    "response": response_text,
                    "response_time_ms": response_time_ms,
                    "tokens_used": tokens_used,
                    "success": True,
                    "error": None,
                }
            )
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000

            task_results.append(
                {
                    "task_index": i,
                    "prompt": prompt,
                    "response": None,
                    "response_time_ms": response_time_ms,
                    "tokens_used": {},
                    "success": False,
                    "error": str(e),
                }
            )

    # Calculate aggregate statistics
    successful_tasks = [r for r in task_results if r["success"]]
    total_response_time = sum(r["response_time_ms"] for r in task_results)
    avg_response_time = (
        total_response_time / len(task_results) if task_results else 0
    )
    success_rate = len(successful_tasks) / len(task_results) if task_results else 0

    total_prompt_tokens = sum(
        r["tokens_used"].get("prompt_tokens", 0) for r in successful_tasks
    )
    total_completion_tokens = sum(
        r["tokens_used"].get("completion_tokens", 0) for r in successful_tasks
    )

    return {
        "model_name": model_name,
        "total_tasks": len(tasks),
        "successful_tasks": len(successful_tasks),
        "failed_tasks": len(task_results) - len(successful_tasks),
        "success_rate": success_rate,
        "total_response_time_ms": total_response_time,
        "avg_response_time_ms": avg_response_time,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "task_results": task_results,
    }


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results from multiple model benchmarks.

    Args:
        results: List of benchmark result dictionaries from benchmark_model()

    Returns:
        Dictionary containing aggregated statistics across all models

    Example:
        >>> results = [
        ...     benchmark_model("model1", tasks, api_key),
        ...     benchmark_model("model2", tasks, api_key),
        ... ]
        >>> summary = aggregate_results(results)
        >>> print(f"Best model: {summary['fastest_model']}")
    """
    if not results:
        return {
            "total_models": 0,
            "total_tasks": 0,
            "overall_success_rate": 0.0,
            "models_summary": [],
        }

    models_summary = []
    for result in results:
        models_summary.append(
            {
                "model_name": result["model_name"],
                "success_rate": result["success_rate"],
                "avg_response_time_ms": result["avg_response_time_ms"],
                "total_tokens": result["total_tokens"],
            }
        )

    # Sort by average response time
    models_summary_sorted = sorted(
        models_summary, key=lambda x: x["avg_response_time_ms"]
    )

    total_successful = sum(r["successful_tasks"] for r in results)
    total_tasks = sum(r["total_tasks"] for r in results)
    overall_success_rate = total_successful / total_tasks if total_tasks > 0 else 0.0

    return {
        "total_models": len(results),
        "total_tasks": total_tasks,
        "total_successful_tasks": total_successful,
        "overall_success_rate": overall_success_rate,
        "fastest_model": (
            models_summary_sorted[0]["model_name"] if models_summary_sorted else None
        ),
        "slowest_model": (
            models_summary_sorted[-1]["model_name"] if models_summary_sorted else None
        ),
        "models_summary": models_summary_sorted,
    }
