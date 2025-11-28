"""
EasyPortrait Dataset Download and Preparation Script

Downloads and prepares EasyPortrait dataset (39,000 images)
for caption generation and finetuning.

EasyPortrait dataset: https://github.com/hukenovs/easyportrait
"""

import os
import sys
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Data.data_loaders import prepare_imagefolder_dataset


def download_easyportrait(download_dir: str):
    """
    Download EasyPortrait dataset.
    
    Args:
        download_dir: Directory to download to
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("EasyPortrait Dataset Download")
    print("="*80)
    print("\nEasyPortrait dataset can be downloaded from:")
    print("https://github.com/hukenovs/easyportrait")
    print("\nOr use git clone:")
    print("git clone https://github.com/hukenovs/easyportrait.git")
    print("\nThe dataset images are typically in the 'images' or 'data' directory")
    print("\nPlease download/clone the dataset and place it in:")
    print(f"{download_dir}")
    print("="*80)


def extract_easyportrait(source_dir: str, extract_dir: str):
    """
    Extract and organize EasyPortrait dataset.
    
    Args:
        source_dir: Source directory (could be git repo or zip extract)
        extract_dir: Directory to organize images
    """
    source_dir = Path(source_dir)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    
    print(f"Organizing EasyPortrait images from {source_dir}...")
    
    # Find all images (could be in various subdirectories)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = []
    
    # Common EasyPortrait directory structures
    possible_dirs = [
        source_dir / "images",
        source_dir / "data",
        source_dir / "dataset",
        source_dir
    ]
    
    for possible_dir in possible_dirs:
        if possible_dir.exists():
            image_files.extend([
                f for f in possible_dir.rglob('*')
                if f.suffix.lower() in image_extensions
            ])
    
    if not image_files:
        raise ValueError(f"No images found in {source_dir}")
    
    print(f"Found {len(image_files)} images")
    
    # Copy images to extract_dir
    for image_file in tqdm(image_files, desc="Organizing images"):
        try:
            dest_file = extract_dir / image_file.name
            # Handle duplicate names
            counter = 1
            while dest_file.exists():
                stem = image_file.stem
                suffix = image_file.suffix
                dest_file = extract_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            shutil.copy2(image_file, dest_file)
        except Exception as e:
            print(f"Error copying {image_file}: {e}")
            continue
    
    print(f"✓ Organized {len(list(extract_dir.glob('*')))} images to {extract_dir}")


def prepare_easyportrait_for_training(
    source_dir: str,
    output_dir: str
):
    """
    Complete EasyPortrait preparation pipeline.
    
    Args:
        source_dir: Directory containing EasyPortrait images
        output_dir: Output directory for prepared dataset
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    print("="*80)
    print("EasyPortrait Dataset Preparation")
    print("="*80)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print("="*80)
    
    # Prepare ImageFolder structure
    print("\nPreparing ImageFolder structure...")
    prepare_imagefolder_dataset(
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        metadata_file="metadata.jsonl"
    )
    
    print("\n" + "="*80)
    print("✓ EasyPortrait dataset prepared successfully!")
    print(f"  Images: {output_dir / 'images'}")
    print(f"  Metadata: {output_dir / 'metadata.jsonl'}")
    print(f"  Ready for caption generation and finetuning")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare EasyPortrait dataset")
    parser.add_argument(
        "--download-dir",
        type=str,
        default="./downloads",
        help="Directory to download dataset"
    )
    parser.add_argument(
        "--extract",
        type=str,
        default=None,
        help="Path to EasyPortrait source directory (git repo or extracted)"
    )
    parser.add_argument(
        "--prepare",
        type=str,
        default=None,
        help="Path to EasyPortrait images directory to prepare"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./easyportrait_processed",
        help="Output directory for prepared dataset"
    )
    
    args = parser.parse_args()
    
    if args.extract:
        # Extract/organize from source
        extract_dir = Path(args.extract).parent / "easyportrait_organized"
        extract_easyportrait(args.extract, str(extract_dir))
        print(f"\nOrganized to: {extract_dir}")
        print("Now run with --prepare flag pointing to organized directory")
        
    elif args.prepare:
        # Prepare dataset
        prepare_easyportrait_for_training(
            source_dir=args.prepare,
            output_dir=args.output
        )
    else:
        # Download instructions
        download_easyportrait(args.download_dir)


if __name__ == "__main__":
    main()

