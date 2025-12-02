"""
Batch Caption Generator Script

This script processes datasets in batches for distributed processing.
Each team member can run this script with different start/end indices
to process subsets of the dataset in parallel.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from groq_caption_generator import GroqCaptionGenerator


def get_image_paths(dataset_dir: str, image_extensions: set = None) -> List[str]:
    """
    Get all image paths from a dataset directory.
    
    Args:
        dataset_dir: Directory containing images
        image_extensions: Set of valid image extensions
        
    Returns:
        Sorted list of image paths
    """
    if image_extensions is None:
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    dataset_dir = Path(dataset_dir)
    image_paths = []
    
    # Support both flat directory and nested structure
    for ext in image_extensions:
        image_paths.extend(dataset_dir.rglob(f"*{ext}"))
        image_paths.extend(dataset_dir.rglob(f"*{ext.upper()}"))
    
    # Sort for consistent ordering across different machines
    image_paths = sorted([str(p) for p in image_paths])
    
    return image_paths


def process_dataset_batch(
    dataset_dir: str,
    output_file: str,
    start_index: int = 0,
    end_index: int = None,
    api_key: str = None,
    model: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
):
    """
    Process a batch of images from a dataset.
    
    Args:
        dataset_dir: Directory containing images
        output_file: Output JSONL file path
        start_index: Starting index for this batch
        end_index: Ending index for this batch (None = process all)
        api_key: Groq API key (optional, uses env var if not provided)
        model: Model name to use
    """
    print(f"="*80)
    print(f"Batch Caption Generation")
    print(f"="*80)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output file: {output_file}")
    print(f"Start index: {start_index}")
    print(f"End index: {end_index if end_index else 'end of dataset'}")
    print(f"="*80)
    
    # Get all image paths
    print("Scanning for images...")
    image_paths = get_image_paths(dataset_dir)
    total_images = len(image_paths)
    
    print(f"Found {total_images} images")
    
    if end_index is None:
        end_index = total_images
    
    if start_index >= total_images:
        print(f"Error: Start index {start_index} is beyond dataset size {total_images}")
        return
    
    if end_index > total_images:
        print(f"Warning: End index {end_index} exceeds dataset size {total_images}. Using {total_images}")
        end_index = total_images
    
    # Initialize caption generator
    print("Initializing Groq API client...")
    generator = GroqCaptionGenerator(api_key=api_key, model=model)
    
    # Process batch
    batch_paths = image_paths[start_index:end_index]
    print(f"Processing {len(batch_paths)} images...")
    
    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process images
    results = []
    for idx, image_path in enumerate(batch_paths):
        try:
            image_name = Path(image_path).name
            print(f"\n[{idx+1}/{len(batch_paths)}] Processing: {image_name}")
            
            caption = generator.generate_caption(image_path)
            
            result = {
                "file_name": image_name,
                "text": caption
            }
            results.append(result)
            
            # Save incrementally
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            print(f"✓ Caption generated: {caption[:100]}...")
            
        except Exception as e:
            print(f"✗ Error processing {image_path}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print(f"Batch processing complete!")
    print(f"Processed: {len(results)}/{len(batch_paths)} images")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate captions for dataset images using Groq API (distributed processing)"
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Directory containing images to process"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting index for this batch (for distributed processing)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Ending index for this batch (None = process all remaining)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Groq API key (or set GROQ_API_KEY environment variable)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/llama-4-maverick-17b-128e-instruct",
        help="Groq model to use"
    )
    
    args = parser.parse_args()
    
    # Check API key
    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: Groq API key is required.")
        print("Set GROQ_API_KEY environment variable or use --api-key argument")
        sys.exit(1)
    
    # Process batch
    process_dataset_batch(
        dataset_dir=args.dataset_dir,
        output_file=args.output,
        start_index=args.start,
        end_index=args.end,
        api_key=api_key,
        model=args.model
    )


if __name__ == "__main__":
    main()

