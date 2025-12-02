"""
Generate a fixed test set of images for evaluation.

This script generates images from a fixed set of test prompts and saves them
with metadata for consistent evaluation across different model versions.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Generation.model import DiffyFaceModel


def load_test_prompts(prompts_file: str = None):
    """Load test prompts from JSON file."""
    if prompts_file is None:
        prompts_file = PROJECT_ROOT / "Evaluation" / "test_prompts.json"
    
    with open(prompts_file, 'r') as f:
        data = json.load(f)
    
    return data["test_prompts"]


def generate_test_set(
    output_dir: str = None,
    num_samples: int = None,
    seed: int = 42,
    num_inference_steps: int = 50
):
    """
    Generate test set images from fixed prompts.
    
    Args:
        output_dir: Directory to save generated images and metadata
        num_samples: Number of prompts to use (None = use all)
        seed: Random seed for reproducibility
        num_inference_steps: Number of inference steps
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "Evaluation" / "test_set_output"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test prompts
    test_prompts = load_test_prompts()
    
    if num_samples is not None:
        test_prompts = test_prompts[:num_samples]
    
    print(f"Generating {len(test_prompts)} test images...")
    print(f"Output directory: {output_dir}")
    
    # Initialize model
    print("\nLoading DiffyFace model...")
    model = DiffyFaceModel(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate images
    results = []
    for idx, prompt_data in enumerate(test_prompts):
        prompt_id = prompt_data["id"]
        prompt = prompt_data["prompt"]
        
        print(f"\n[{idx+1}/{len(test_prompts)}] Generating: {prompt_id}")
        print(f"  Prompt: {prompt[:80]}...")
        
        try:
            # Generate image with fixed seed for reproducibility
            image = model.generate(
                prompt=prompt,
                negative_prompt="blurry, distorted, low quality, deformed",
                num_inference_steps=num_inference_steps,
                seed=seed + idx,  # Different seed per image but deterministic
                guidance_scale=7.5
            )
            
            # Save image
            image_filename = f"{prompt_id}.png"
            image_path = output_dir / image_filename
            image.save(image_path)
            
            # Store result
            result = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "attributes": prompt_data.get("attributes", {}),
                "image_path": str(image_path),
                "image_filename": image_filename,
                "generation_params": {
                    "seed": seed + idx,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": 7.5
                },
                "timestamp": datetime.now().isoformat()
            }
            results.append(result)
            
            print(f"  ✓ Saved: {image_filename}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    # Save metadata
    metadata = {
        "test_set_info": {
            "total_prompts": len(test_prompts),
            "successful_generations": len(results),
            "generation_date": datetime.now().isoformat(),
            "model": "rigvedrs/DiffyFace",
            "base_model": "rigvedrs/Diffy-2-1"
        },
        "results": results
    }
    
    metadata_path = output_dir / "test_set_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Test set generation complete!")
    print(f"  Generated: {len(results)}/{len(test_prompts)} images")
    print(f"  Metadata saved: {metadata_path}")
    
    return results, metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate test set for evaluation")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for test images")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of prompts to generate (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of inference steps")
    
    args = parser.parse_args()
    
    generate_test_set(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        num_inference_steps=args.steps
    )

