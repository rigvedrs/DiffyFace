"""
FFHQ Dataset Download and Preparation Script

Downloads and prepares FFHQ dataset (70,000 images, 1024×1024 → 768×768)
for caption generation and finetuning.

FFHQ dataset: https://github.com/NVlabs/ffhq-dataset
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import argparse
from PIL import Image
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Data.data_loaders import prepare_imagefolder_dataset


def clone_and_download_ffhq(download_dir: str):
    """
    Clone FFHQ repository and download dataset using official script.
    
    Args:
        download_dir: Directory to download to
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("FFHQ Dataset Download (Official Method)")
    print("="*80)
    
    # Clone repository
    repo_dir = download_dir / "ffhq-dataset"
    if not repo_dir.exists():
        print("\nCloning FFHQ repository...")
        result = subprocess.run(
            ["git", "clone", "https://github.com/NVlabs/ffhq-dataset.git", str(repo_dir)],
            check=True,
            capture_output=True,
            text=True
        )
        print("✓ Repository cloned successfully")
    else:
        print(f"\nRepository already exists at {repo_dir}")
    
    # Change to repo directory and run download script
    print("\nDownloading FFHQ dataset (this will download ~89GB of images)...")
    print("This may take a while depending on your internet connection.")
    
    download_script = repo_dir / "download_ffhq.py"
    if not download_script.exists():
        raise FileNotFoundError(f"Download script not found: {download_script}")
    
    # Run the download script
    result = subprocess.run(
        [sys.executable, str(download_script), "--json", "--images"],
        cwd=str(repo_dir),
        check=True
    )
    
    print("\n✓ FFHQ dataset downloaded successfully!")
    print(f"Images location: {repo_dir / 'images1024x1024'}")
    print(f"\nTo prepare for training, run:")
    print(f"python download_and_prepare.py --prepare {repo_dir / 'images1024x1024'} --output ./ffhq_processed --target-size 768")
    
    return str(repo_dir / "images1024x1024")


def download_ffhq(download_dir: str, method: str = "manual"):
    """
    Download FFHQ dataset (legacy method - kept for backward compatibility).
    
    Args:
        download_dir: Directory to download to
        method: Download method ('manual' or 'wget')
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("FFHQ Dataset Download (Legacy Method)")
    print("="*80)
    print("\nNote: The recommended method is to use --clone-and-download")
    print("This will use the official FFHQ download script.")
    print("\nFor manual download, visit:")
    print("https://github.com/NVlabs/ffhq-dataset")
    print("="*80)
    
    if method == "wget":
        print("\nAttempting to download with wget...")
        os.system(f"cd {download_dir} && wget https://github.com/NVlabs/ffhq-dataset/releases/download/v1.0/ffhq-1024x1024.zip")
    else:
        print("\nPlease use --clone-and-download for the official download method")


def extract_ffhq(zip_path: str, extract_dir: str):
    """
    Extract FFHQ dataset from zip file.
    
    Args:
        zip_path: Path to FFHQ zip file
        extract_dir: Directory to extract to
    """
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    if not zip_path.exists():
        raise FileNotFoundError(f"FFHQ zip file not found: {zip_path}")
    
    print(f"Extracting {zip_path} to {extract_dir}...")
    
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get total files for progress bar
        file_list = zip_ref.namelist()
        print(f"Found {len(file_list)} files in archive")
        
        # Extract with progress bar
        for file in tqdm(file_list, desc="Extracting"):
            zip_ref.extract(file, extract_dir)
    
    print(f"✓ Extraction complete: {extract_dir}")


def resize_images(source_dir: str, target_dir: str, target_size: int = 768):
    """
    Resize FFHQ images from 1024×1024 to target size (default 768×768).
    Handles both flat directory structure and nested structure (images1024x1024/00000/00000.png).
    
    Args:
        source_dir: Directory containing original images (can be nested)
        target_dir: Directory to save resized images
        target_size: Target image size (default 768)
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files (handles nested structure)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [
        f for f in source_dir.rglob('*')
        if f.suffix.lower() in image_extensions and f.is_file()
    ]
    
    print(f"Found {len(image_files)} images")
    print(f"Resizing from 1024×1024 to {target_size}×{target_size}...")
    
    for image_file in tqdm(image_files, desc="Resizing"):
        try:
            # Load image
            img = Image.open(image_file).convert('RGB')
            
            # Resize to target size (FFHQ images are already square 1024x1024)
            img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            
            # Save resized image (preserve original name)
            target_file = target_dir / image_file.name
            # Ensure unique filename if there are duplicates in nested structure
            if target_file.exists():
                # Use relative path from source_dir to create unique name
                rel_path = image_file.relative_to(source_dir)
                target_file = target_dir / str(rel_path).replace('/', '_')
            
            img.save(target_file, 'PNG' if image_file.suffix.lower() == '.png' else 'JPEG', quality=95)
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue
    
    print(f"✓ Resizing complete: {len(list(target_dir.rglob('*.*')))} images in {target_dir}")


def prepare_ffhq_for_training(
    source_dir: str,
    output_dir: str,
    resize: bool = True,
    target_size: int = 768
):
    """
    Complete FFHQ preparation pipeline.
    
    Args:
        source_dir: Directory containing FFHQ images (1024×1024)
        output_dir: Output directory for prepared dataset
        resize: Whether to resize images to 768×768
        target_size: Target image size
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    print("="*80)
    print("FFHQ Dataset Preparation")
    print("="*80)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Resize: {resize} ({target_size}×{target_size})")
    print("="*80)
    
    # Step 1: Resize images if needed
    if resize:
        temp_dir = output_dir / "temp_resized"
        resize_images(str(source_dir), str(temp_dir), target_size)
        source_dir = temp_dir
    
    # Step 2: Prepare ImageFolder structure
    print("\nPreparing ImageFolder structure...")
    prepare_imagefolder_dataset(
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        metadata_file="metadata.jsonl"
    )
    
    # Clean up temp directory if created
    if resize and temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    print("\n" + "="*80)
    print("✓ FFHQ dataset prepared successfully!")
    print(f"  Images: {output_dir / 'images'}")
    print(f"  Metadata: {output_dir / 'metadata.jsonl'}")
    print(f"  Ready for caption generation and finetuning")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare FFHQ dataset")
    parser.add_argument(
        "--clone-and-download",
        action="store_true",
        help="Clone FFHQ repository and download dataset using official script (recommended)"
    )
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
        help="Path to FFHQ zip file to extract (legacy method)"
    )
    parser.add_argument(
        "--prepare",
        type=str,
        default=None,
        help="Path to FFHQ images directory to prepare (supports nested structure)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./ffhq_processed",
        help="Output directory for prepared dataset"
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Skip resizing (keep original 1024×1024)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=768,
        help="Target image size (default: 768)"
    )
    
    args = parser.parse_args()
    
    if args.clone_and_download:
        # Clone and download using official method
        images_dir = clone_and_download_ffhq(args.download_dir)
        print(f"\n✓ Download complete!")
        print(f"Images are located at: {images_dir}")
        print(f"\nNext step: Prepare the dataset for training")
        print(f"python download_and_prepare.py --prepare {images_dir} --output {args.output} --target-size {args.target_size}")
        
    elif args.extract:
        # Extract zip file (legacy method)
        extract_dir = Path(args.extract).parent / "ffhq_extracted"
        extract_ffhq(args.extract, str(extract_dir))
        print(f"\nExtracted to: {extract_dir}")
        print("Now run with --prepare flag pointing to extracted directory")
        
    elif args.prepare:
        # Prepare dataset
        prepare_ffhq_for_training(
            source_dir=args.prepare,
            output_dir=args.output,
            resize=not args.no_resize,
            target_size=args.target_size
        )
    else:
        # Show help
        print("FFHQ Dataset Download and Preparation")
        print("="*80)
        print("\nRecommended method (official):")
        print("  python download_and_prepare.py --clone-and-download --download-dir ./downloads")
        print("\nThen prepare for training:")
        print("  python download_and_prepare.py --prepare ./downloads/ffhq-dataset/images1024x1024 --output ./ffhq_processed")
        print("\nFor help, use: python download_and_prepare.py --help")


if __name__ == "__main__":
    main()

