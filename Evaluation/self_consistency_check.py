"""
Self-Consistency Evaluation: Groq Round-Trip Check

This script evaluates model consistency by:
1. Generating images from test prompts
2. Feeding generated images back to Groq API to get descriptions
3. Comparing original prompts with regenerated descriptions
4. Computing similarity metrics
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import re
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Data.groq_caption_generator import GroqCaptionGenerator


def extract_attributes(text: str) -> Dict[str, any]:
    """
    Extract key attributes from a description text.
    
    This is a simplified extractor - in practice, you might use
    more sophisticated NLP techniques or structured prompts.
    """
    text_lower = text.lower()
    
    attributes = {
        "age": None,
        "gender": None,
        "hair_color": None,
        "facial_hair": None,
        "expression": None,
        "accessories": None,
        "eye_color": None
    }
    
    # Age extraction (look for patterns like "25 year old", "30-year-old")
    age_pattern = r'(\d+)\s*(?:year|yr)'
    age_match = re.search(age_pattern, text_lower)
    if age_match:
        attributes["age"] = age_match.group(1)
    
    # Gender
    if any(word in text_lower for word in ["male", "man", "guy", "gentleman"]):
        attributes["gender"] = "male"
    elif any(word in text_lower for word in ["female", "woman", "lady"]):
        attributes["gender"] = "female"
    
    # Hair color
    hair_colors = ["blond", "blonde", "brown", "black", "red", "gray", "grey", "white", "auburn"]
    for color in hair_colors:
        if color in text_lower:
            attributes["hair_color"] = color
            break
    
    # Facial hair
    if "beard" in text_lower:
        if "french beard" in text_lower or "goatee" in text_lower:
            attributes["facial_hair"] = "goatee"
        elif "full beard" in text_lower:
            attributes["facial_hair"] = "full beard"
        else:
            attributes["facial_hair"] = "beard"
    elif "mustache" in text_lower or "moustache" in text_lower:
        attributes["facial_hair"] = "mustache"
    elif "stubble" in text_lower:
        attributes["facial_hair"] = "stubble"
    else:
        attributes["facial_hair"] = "none"
    
    # Expression
    if any(word in text_lower for word in ["smile", "smiling", "happy", "cheerful"]):
        attributes["expression"] = "smiling"
    elif any(word in text_lower for word in ["serious", "neutral", "stoic"]):
        attributes["expression"] = "serious"
    elif "cry" in text_lower or "sad" in text_lower:
        attributes["expression"] = "sad"
    
    # Accessories
    if "glasses" in text_lower or "eyeglasses" in text_lower:
        attributes["accessories"] = "eyeglasses"
    elif "sunglasses" in text_lower:
        attributes["accessories"] = "sunglasses"
    elif "hat" in text_lower:
        attributes["accessories"] = "hat"
    else:
        attributes["accessories"] = "none"
    
    # Eye color
    eye_colors = ["brown", "blue", "green", "hazel", "gray", "grey"]
    for color in eye_colors:
        if f"{color} eye" in text_lower:
            attributes["eye_color"] = color
            break
    
    return attributes


def compare_attributes(original: Dict, regenerated: Dict) -> Dict[str, bool]:
    """Compare two attribute dictionaries and return matches."""
    matches = {}
    
    for key in original.keys():
        if key in regenerated:
            orig_val = str(original[key]).lower() if original[key] else None
            regen_val = str(regenerated[key]).lower() if regenerated[key] else None
            
            # Handle None cases
            if orig_val is None and regen_val is None:
                matches[key] = True
            elif orig_val is None or regen_val is None:
                matches[key] = False
            else:
                # Fuzzy matching for some attributes
                if key == "age":
                    # Allow ±5 years
                    try:
                        orig_age = int(orig_val)
                        regen_age = int(regen_val)
                        matches[key] = abs(orig_age - regen_age) <= 5
                    except:
                        matches[key] = orig_val == regen_val
                else:
                    matches[key] = orig_val == regen_val or orig_val in regen_val or regen_val in orig_val
        else:
            matches[key] = False
    
    return matches


def evaluate_self_consistency(
    test_set_metadata_path: str,
    output_file: str = None,
    use_cached: bool = False
):
    """
    Perform self-consistency evaluation.
    
    Args:
        test_set_metadata_path: Path to test set metadata JSON
        output_file: Path to save evaluation results
        use_cached: If True, skip Groq API calls and use cached descriptions
    """
    # Load test set metadata
    with open(test_set_metadata_path, 'r') as f:
        test_data = json.load(f)
    
    results = test_data["results"]
    
    print(f"Evaluating self-consistency for {len(results)} images...")
    
    # Initialize Groq caption generator
    try:
        groq_gen = GroqCaptionGenerator()
    except Exception as e:
        print(f"Error initializing Groq: {e}")
        print("Note: This script requires GROQ_API_KEY environment variable")
        return None
    
    # Evaluation results
    evaluation_results = []
    attribute_matches = defaultdict(lambda: {"total": 0, "matched": 0})
    
    for idx, result in enumerate(results):
        prompt_id = result["prompt_id"]
        original_prompt = result["prompt"]
        original_attributes = result.get("attributes", {})
        image_path = result["image_path"]
        
        print(f"\n[{idx+1}/{len(results)}] Evaluating: {prompt_id}")
        print(f"  Original: {original_prompt[:60]}...")
        
        # Check if image exists
        if not Path(image_path).exists():
            print(f"  ✗ Image not found: {image_path}")
            continue
        
        try:
            # Get regenerated description from Groq
            print("  → Sending to Groq API...")
            regenerated_description = groq_gen.generate_caption(image_path)
            print(f"  → Received: {regenerated_description[:60]}...")
            
            # Extract attributes from regenerated description
            regenerated_attributes = extract_attributes(regenerated_description)
            
            # Compare attributes
            matches = compare_attributes(original_attributes, regenerated_attributes)
            
            # Count matches
            total_attrs = len([v for v in original_attributes.values() if v is not None])
            matched_attrs = sum(matches.values())
            match_rate = matched_attrs / total_attrs if total_attrs > 0 else 0
            
            # Update attribute-level statistics
            for attr_name, is_match in matches.items():
                if original_attributes.get(attr_name) is not None:
                    attribute_matches[attr_name]["total"] += 1
                    if is_match:
                        attribute_matches[attr_name]["matched"] += 1
            
            eval_result = {
                "prompt_id": prompt_id,
                "original_prompt": original_prompt,
                "original_attributes": original_attributes,
                "regenerated_description": regenerated_description,
                "regenerated_attributes": regenerated_attributes,
                "attribute_matches": matches,
                "match_rate": match_rate,
                "total_attributes": total_attrs,
                "matched_attributes": matched_attrs
            }
            evaluation_results.append(eval_result)
            
            print(f"  ✓ Match rate: {match_rate:.2%} ({matched_attrs}/{total_attrs})")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    # Compute overall statistics
    overall_stats = {
        "total_samples": len(evaluation_results),
        "average_match_rate": sum(r["match_rate"] for r in evaluation_results) / len(evaluation_results) if evaluation_results else 0,
        "attribute_accuracy": {
            attr: {
                "accuracy": stats["matched"] / stats["total"] if stats["total"] > 0 else 0,
                "matched": stats["matched"],
                "total": stats["total"]
            }
            for attr, stats in attribute_matches.items()
        }
    }
    
    # Compile final results
    final_results = {
        "evaluation_type": "self_consistency",
        "test_set_path": str(test_set_metadata_path),
        "overall_statistics": overall_stats,
        "detailed_results": evaluation_results
    }
    
    # Save results
    if output_file is None:
        output_file = Path(test_set_metadata_path).parent / "self_consistency_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✓ Self-consistency evaluation complete!")
    print(f"  Average match rate: {overall_stats['average_match_rate']:.2%}")
    print(f"  Results saved: {output_file}")
    
    # Print attribute-level accuracy
    print("\nAttribute-level Accuracy:")
    for attr, stats in overall_stats["attribute_accuracy"].items():
        if stats["total"] > 0:
            print(f"  {attr}: {stats['accuracy']:.2%} ({stats['matched']}/{stats['total']})")
    
    return final_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-consistency evaluation")
    parser.add_argument("--test-set", type=str, required=True,
                       help="Path to test set metadata JSON")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results")
    
    args = parser.parse_args()
    
    evaluate_self_consistency(args.test_set, args.output)

