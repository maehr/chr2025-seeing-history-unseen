"""
Example benchmark tasks for VLM evaluation.

This module provides sample tasks for benchmarking vision-language models
on alt-text generation for digital heritage collections.
"""

from typing import Any, Dict, List


def get_wcag_alttext_tasks() -> List[Dict[str, Any]]:
    """
    Get benchmark tasks focused on WCAG-compliant alt-text generation.

    Returns:
        List of task dictionaries with prompts and optional image URLs
    """
    return [
        {
            "prompt": (
                "You are an accessibility assistant for a digital heritage collection. "
                "Generate a concise, WCAG 2.2-compliant alt-text for a historical photograph "
                "showing a street scene from 1920s Zurich. The alt-text should be under 120 "
                "characters, not start with 'image of' or 'photo of', and describe the key "
                "visual elements relevant for understanding the historical context."
            ),
            "description": "Alt-text generation for historical street scene",
        },
        {
            "prompt": (
                "Generate WCAG-compliant alt-text for a complex historical diagram showing "
                "a medieval castle's floor plan. Provide both a short descriptive alt-text "
                "(under 120 characters) and indicate that a longer description would be needed "
                "to fully convey the diagram's details."
            ),
            "description": "Alt-text for complex diagram",
        },
        {
            "prompt": (
                "Create alt-text for a historical document image containing handwritten text "
                "in German from the 18th century. Follow WCAG guidelines: indicate it's a "
                "historical document, note the language and approximate date, and mention that "
                "a full transcription would be provided separately."
            ),
            "description": "Alt-text for historical document with text",
        },
        {
            "prompt": (
                "Generate accessible alt-text for a museum artifact photo showing a ceramic "
                "vase from ancient Greece with painted decorations. The alt-text should describe "
                "the object type, approximate age, and notable visual features without being "
                "overly verbose. Aim for 80-120 characters."
            ),
            "description": "Alt-text for museum artifact",
        },
        {
            "prompt": (
                "Write WCAG-compliant alt-text for a historical map showing European trade "
                "routes in the 16th century. Since this is a complex informative image, provide "
                "a brief summary alt-text and note that detailed route information would be "
                "provided in an accompanying description."
            ),
            "description": "Alt-text for historical map",
        },
    ]


def get_simple_description_tasks() -> List[Dict[str, Any]]:
    """
    Get simple image description tasks for basic benchmarking.

    Returns:
        List of task dictionaries with basic prompts
    """
    return [
        {
            "prompt": "Describe what you see in this image in one sentence.",
            "description": "Simple one-sentence description",
        },
        {
            "prompt": "List the main objects visible in this image.",
            "description": "Object identification",
        },
        {
            "prompt": "What is the setting or location shown in this image?",
            "description": "Setting identification",
        },
        {
            "prompt": "Are there any people in this image? If so, describe them briefly.",
            "description": "Person detection and description",
        },
        {
            "prompt": "What time period or era does this image appear to be from?",
            "description": "Historical period identification",
        },
    ]


def get_detailed_analysis_tasks() -> List[Dict[str, Any]]:
    """
    Get detailed analysis tasks for comprehensive evaluation.

    Returns:
        List of task dictionaries requiring detailed responses
    """
    return [
        {
            "prompt": (
                "Provide a detailed analysis of this historical image including: "
                "1) Visual composition and elements, "
                "2) Historical context and period, "
                "3) Cultural or social significance, "
                "4) Any text or inscriptions visible."
            ),
            "description": "Comprehensive historical analysis",
        },
        {
            "prompt": (
                "Analyze this image for accessibility purposes. Identify: "
                "1) Main subject and key visual elements, "
                "2) Important details for understanding the image's purpose, "
                "3) Any challenges in creating accessible descriptions, "
                "4) Suggestions for making this content more accessible."
            ),
            "description": "Accessibility-focused analysis",
        },
        {
            "prompt": (
                "Examine this heritage image and identify: "
                "1) Type of artifact or document, "
                "2) Estimated age or time period, "
                "3) Notable features or characteristics, "
                "4) Potential research or educational value."
            ),
            "description": "Heritage artifact evaluation",
        },
    ]


def get_all_tasks() -> List[Dict[str, Any]]:
    """
    Get all available benchmark tasks.

    Returns:
        Combined list of all task types
    """
    return (
        get_wcag_alttext_tasks()
        + get_simple_description_tasks()
        + get_detailed_analysis_tasks()
    )
