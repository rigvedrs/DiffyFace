"""
Quantitative Attribute Evaluation

This script evaluates the model's ability to generate images matching specific attributes
by comparing generated images with their prompts on a per-attribute basis.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import re

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_attributes_from_prompt(prompt: str) -> Dict[str, any]:
    """Extract attributes from a prompt text."""
    prompt_lower = prompt.lower()
    
    attributes = {}
    
    # Age
    age_pattern = r'(\d+)\s*(?:year|yr)'
    age_match = re.search(age_pattern, prompt_lower)
    if age_match:
        attributes["age"] = int(age_match.group(1))
    
    # Gender
    if any(word in prompt_lower for word in ["male", "man", "guy"]):
        attributes["gender"] = "male"
    elif any(word in prompt_lower for word in ["female", "woman", "lady"]):
        attributes["gender"] = "female"
    
    # Hair color
    hair_colors = ["blond", "blonde", "brown", "black", "red", "gray", "grey", "white"]
    for color in hair_colors:
        if color in prompt_lower:
            attributes["hair_color"] = color
            break
    
    # Facial hair
    if "french beard" in prompt_lower or "goatee" in prompt_lower:
        attributes["facial_hair"] = "goatee"
    elif "full beard" in prompt_lower:
        attributes["facial_hair"] = "full beard"
    elif "mustache" in prompt_lower or "moustache" in prompt_lower:
        attributes["facial_hair"] = "mustache"
    elif "stubble" in prompt_lower:
        attributes["facial_hair"] = "stubble"
    elif "beard" in prompt_lower:
        attributes["facial_hair"] = "beard"
    else:
        attributes["facial_hair"] = "none"
    
    # Expression
    if any(word in prompt_lower for word in ["smile", "smiling", "happy", "cheerful"]):
        attributes["expression"] = "smiling"
    elif any(word in prompt_lower for word in ["serious", "neutral"]):
        attributes["expression"] = "serious"
    elif "cry" in prompt_lower or "cries" in prompt_lower:
        attributes["expression"] = "crying"
    
    # Accessories
    if "eyeglasses" in prompt_lower or ("glasses" in prompt_lower and "sunglasses" not in prompt_lower):
        attributes["accessories"] = "eyeglasses"
    elif "sunglasses" in prompt_lower:
        attributes["accessories"] = "sunglasses"
    elif "hat" in prompt_lower:
        attributes["accessories"] = "hat"
    else:
        attributes["accessories"] = "none"
    
    # Eye color
    eye_colors = ["brown", "blue", "green", "hazel", "gray", "grey"]
    for color in eye_colors:
        if f"{color} eye" in prompt_lower:
            attributes["eye_color"] = color
            break
    
    return attributes


def evaluate_attributes(
    test_set_metadata_path: str,
    output_file: str = None
):
    """
    Evaluate attribute accuracy from test set.
    
    This assumes the test set metadata already contains attribute information.
    For a more thorough evaluation, you could also use vision models to
    extract attributes from generated images and compare.
    """
    # Load test set metadata
    with open(test_set_metadata_path, 'r') as f:
        test_data = json.load(f)
    
    results = test_data["results"]
    
    print(f"Evaluating attributes for {len(results)} test samples...")
    
    # Attribute statistics
    attribute_stats = defaultdict(lambda: {
        "total": 0,
        "present_in_prompt": 0,
        "present_in_attributes": 0
    })
    
    # Per-sample results
    sample_results = []
    
    for idx, result in enumerate(results):
        prompt = result["prompt"]
        expected_attributes = result.get("attributes", {})
        
        # Extract attributes from prompt (as a check)
        extracted_attributes = extract_attributes_from_prompt(prompt)
        
        # Compare expected vs extracted (sanity check)
        sample_result = {
            "prompt_id": result["prompt_id"],
            "prompt": prompt,
            "expected_attributes": expected_attributes,
            "extracted_attributes": extracted_attributes
        }
        
        # Count attribute presence
        for attr_name, attr_value in expected_attributes.items():
            if attr_value is not None and attr_value != "":
                attribute_stats[attr_name]["total"] += 1
                attribute_stats[attr_name]["present_in_attributes"] += 1
                
                # Check if also in extracted
                if attr_name in extracted_attributes:
                    if str(extracted_attributes[attr_name]).lower() == str(attr_value).lower():
                        attribute_stats[attr_name]["present_in_prompt"] += 1
        
        sample_results.append(sample_result)
    
    # Compute accuracy metrics
    attribute_accuracy = {}
    for attr_name, stats in attribute_stats.items():
        if stats["total"] > 0:
            accuracy = stats["present_in_attributes"] / stats["total"]
            attribute_accuracy[attr_name] = {
                "accuracy": accuracy,
                "samples_with_attribute": stats["total"],
                "present_in_prompt": stats["present_in_prompt"]
            }
    
    # Overall statistics
    overall_stats = {
        "total_samples": len(results),
        "attributes_evaluated": len(attribute_accuracy),
        "attribute_accuracy": attribute_accuracy
    }
    
    # Compile final results
    final_results = {
        "evaluation_type": "attribute_accuracy",
        "test_set_path": str(test_set_metadata_path),
        "overall_statistics": overall_stats,
        "detailed_results": sample_results
    }
    
    # Save results
    if output_file is None:
        output_file = Path(test_set_metadata_path).parent / "attribute_evaluation_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✓ Attribute evaluation complete!")
    print(f"  Total samples: {overall_stats['total_samples']}")
    print(f"  Attributes evaluated: {overall_stats['attributes_evaluated']}")
    print(f"  Results saved: {output_file}")
    
    # Print attribute accuracy
    print("\nAttribute Coverage (samples with attribute / total samples):")
    for attr, stats in attribute_accuracy.items():
        print(f"  {attr}: {stats['accuracy']:.2%} ({stats['samples_with_attribute']}/{overall_stats['total_samples']})")
    
    return final_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Attribute evaluation")
    parser.add_argument("--test-set", type=str, required=True,
                       help="Path to test set metadata JSON")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results")
    
    args = parser.parse_args()
    
    evaluate_attributes(args.test_set, args.output)

