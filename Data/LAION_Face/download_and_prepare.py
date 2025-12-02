"""
LAION-Face Dataset Download and Preparation Script

Downloads and prepares LAION-Face subset (156,000 images)
for caption generation and finetuning.

LAION-Face dataset: https://huggingface.co/datasets/FacePerceiver/laion-face
"""

import os
import sys
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm
from PIL import Image

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Data.data_loaders import prepare_imagefolder_dataset, HuggingFaceDatasetWrapper

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets package not installed. Install with: pip install datasets")


def download_laion_face_huggingface(download_dir: str, num_samples: int = None):
    """
    Download LAION-Face dataset from HuggingFace.
    
    Args:
        download_dir: Directory to download to
        num_samples: Number of samples to download (None = all, 156k for subset)
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets package is required. Install with: pip install datasets")
    
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("LAION-Face Dataset Download (HuggingFace)")
    print("="*80)
    print(f"Downloading to: {download_dir}")
    if num_samples:
        print(f"Downloading {num_samples} samples (subset)")
    else:
        print("Downloading full dataset")
    print("="*80)
    
    try:
        # Load dataset from HuggingFace
        print("Loading dataset from HuggingFace...")
        dataset = load_dataset(
            "FacePerceiver/laion-face",
            split="train",
            cache_dir=str(download_dir / "cache")
        )
        
        print(f"Dataset loaded: {len(dataset)} samples")
        
        # Select subset if specified
        if num_samples and num_samples < len(dataset):
            print(f"Selecting {num_samples} samples...")
            dataset = dataset.select(range(num_samples))
        
        # Save images
        images_dir = download_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving {len(dataset)} images...")
        for idx, item in enumerate(tqdm(dataset, desc="Saving images")):
            try:
                image = item.get("image")
                if image is None:
                    continue
                
                # Convert to PIL if needed
                if not isinstance(image, Image.Image):
                    image = Image.open(image).convert('RGB')
                
                # Save image
                image_file = images_dir / f"laion_face_{idx:06d}.jpg"
                image.save(image_file, 'JPEG', quality=95)
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        print(f"✓ Downloaded {len(list(images_dir.glob('*')))} images to {images_dir}")
        return str(images_dir)
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nAlternative: Download manually from HuggingFace and extract")
        raise


def filter_laion_face_quality(
    source_dir: str,
    output_dir: str,
    min_size: tuple = (512, 512),
    max_size: tuple = (2048, 2048)
):
    """
    Filter LAION-Face images by quality criteria.
    
    Args:
        source_dir: Directory containing LAION-Face images
        output_dir: Directory to save filtered images
        min_size: Minimum image dimensions (width, height)
        max_size: Maximum image dimensions (width, height)
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Filtering LAION-Face images by quality...")
    print(f"Min size: {min_size}, Max size: {max_size}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [
        f for f in source_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    filtered_count = 0
    for image_file in tqdm(image_files, desc="Filtering"):
        try:
            img = Image.open(image_file).convert('RGB')
            width, height = img.size
            
            # Check size constraints
            if width < min_size[0] or height < min_size[1]:
                continue
            if width > max_size[0] or height > max_size[1]:
                continue
            
            # Copy to output
            shutil.copy2(image_file, output_dir / image_file.name)
            filtered_count += 1
            
        except Exception as e:
            continue
    
    print(f"✓ Filtered {filtered_count}/{len(image_files)} images to {output_dir}")


def prepare_laion_face_for_training(
    source_dir: str,
    output_dir: str,
    filter_quality: bool = True,
    num_samples: int = 156000
):
    """
    Complete LAION-Face preparation pipeline.
    
    Args:
        source_dir: Directory containing LAION-Face images
        output_dir: Output directory for prepared dataset
        filter_quality: Whether to filter by quality
        num_samples: Number of samples to include (for subset)
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    print("="*80)
    print("LAION-Face Dataset Preparation")
    print("="*80)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Filter quality: {filter_quality}")
    print(f"Target samples: {num_samples}")
    print("="*80)
    
    # Step 1: Filter by quality if needed
    if filter_quality:
        temp_dir = output_dir / "temp_filtered"
        filter_laion_face_quality(str(source_dir), str(temp_dir))
        source_dir = temp_dir
    
    # Step 2: Limit to subset if needed
    if num_samples:
        image_files = sorted([
            f for f in source_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        ])
        
        if len(image_files) > num_samples:
            print(f"\nSelecting {num_samples} samples from {len(image_files)}...")
            selected_files = image_files[:num_samples]
            subset_dir = output_dir / "temp_subset"
            subset_dir.mkdir(parents=True, exist_ok=True)
            
            for img_file in tqdm(selected_files, desc="Creating subset"):
                shutil.copy2(img_file, subset_dir / img_file.name)
            
            source_dir = subset_dir
    
    # Step 3: Prepare ImageFolder structure
    print("\nPreparing ImageFolder structure...")
    prepare_imagefolder_dataset(
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        metadata_file="metadata.jsonl"
    )
    
    # Clean up temp directories
    for temp_dir in [output_dir / "temp_filtered", output_dir / "temp_subset"]:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    print("\n" + "="*80)
    print("✓ LAION-Face dataset prepared successfully!")
    print(f"  Images: {output_dir / 'images'}")
    print(f"  Metadata: {output_dir / 'metadata.jsonl'}")
    print(f"  Ready for caption generation and finetuning")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare LAION-Face dataset")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download from HuggingFace"
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="./downloads",
        help="Directory to download dataset"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=156000,
        help="Number of samples to download/prepare (default: 156000 for subset)"
    )
    parser.add_argument(
        "--prepare",
        type=str,
        default=None,
        help="Path to LAION-Face images directory to prepare"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./laion_face_processed",
        help="Output directory for prepared dataset"
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip quality filtering"
    )
    
    args = parser.parse_args()
    
    if args.download:
        # Download from HuggingFace
        images_dir = download_laion_face_huggingface(
            args.download_dir,
            num_samples=args.num_samples
        )
        print(f"\nDownloaded to: {images_dir}")
        print("Now run with --prepare flag pointing to downloaded directory")
        
    elif args.prepare:
        # Prepare dataset
        prepare_laion_face_for_training(
            source_dir=args.prepare,
            output_dir=args.output,
            filter_quality=not args.no_filter,
            num_samples=args.num_samples
        )
    else:
        print("Use --download to download from HuggingFace")
        print("Or use --prepare to prepare existing images")


if __name__ == "__main__":
    main()

