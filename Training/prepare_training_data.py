"""
Training Data Preparation Script

Merges multiple metadata.jsonl files from team members into a single
training-ready dataset for each source dataset (FFHQ, EasyPortrait, LAION-Face).

Each dataset folder contains:
- images/ directory with all images
- metadata.jsonl file (single file, can be merged from multiple member files)
"""

import json
import shutil
import sys
from pathlib import Path
from typing import List, Dict
import argparse
from tqdm import tqdm


def merge_metadata_files(metadata_files: List[Path], output_file: Path) -> int:
    """
    Merge multiple metadata.jsonl files into one.
    
    Args:
        metadata_files: List of metadata.jsonl file paths
        output_file: Output merged metadata file
        
    Returns:
        Number of entries merged
    """
    all_entries = []
    seen_files = set()  # Track duplicate file names
    
    for metadata_file in metadata_files:
        if not metadata_file.exists():
            print(f"Warning: Metadata file not found: {metadata_file}")
            continue
        
        print(f"Loading metadata from: {metadata_file}")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        file_name = entry.get("file_name", "")
                        
                        # Handle duplicate file names by adding prefix
                        if file_name in seen_files:
                            # Extract name and extension
                            name_parts = Path(file_name).stem, Path(file_name).suffix
                            counter = 1
                            while file_name in seen_files:
                                file_name = f"{name_parts[0]}_{counter}{name_parts[1]}"
                                counter += 1
                            entry["file_name"] = file_name
                        
                        seen_files.add(file_name)
                        all_entries.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line in {metadata_file}: {e}")
                        continue
    
    # Write merged metadata
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✓ Merged {len(all_entries)} entries to {output_file}")
    return len(all_entries)


def copy_images(source_dirs: List[Path], output_images_dir: Path) -> int:
    """
    Copy images from multiple source directories to output directory.
    
    Args:
        source_dirs: List of source image directories
        output_images_dir: Output images directory
        
    Returns:
        Number of images copied
    """
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    copied_count = 0
    seen_files = set()
    
    for source_dir in source_dirs:
        if not source_dir.exists():
            print(f"Warning: Source directory not found: {source_dir}")
            continue
        
        print(f"Copying images from: {source_dir}")
        
        # Find all images
        image_files = [
            f for f in source_dir.rglob('*')
            if f.suffix.lower() in image_extensions
        ]
        
        for image_file in tqdm(image_files, desc=f"Copying from {source_dir.name}"):
            file_name = image_file.name
            
            # Handle duplicate file names
            if file_name in seen_files:
                name_parts = Path(file_name).stem, Path(file_name).suffix
                counter = 1
                while file_name in seen_files:
                    file_name = f"{name_parts[0]}_{counter}{name_parts[1]}"
                    counter += 1
            
            dest_file = output_images_dir / file_name
            try:
                shutil.copy2(image_file, dest_file)
                seen_files.add(file_name)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {image_file}: {e}")
                continue
    
    print(f"✓ Copied {copied_count} images to {output_images_dir}")
    return copied_count


def prepare_dataset(
    dataset_name: str,
    data_root: Path,
    output_dir: Path,
    metadata_files: List[Path] = None
):
    """
    Prepare a single dataset by merging metadata files and organizing images.
    
    Args:
        dataset_name: Name of the dataset (FFHQ, EasyPortrait, LAION_Face)
        data_root: Root directory containing dataset folder
        output_dir: Output directory for prepared dataset
        metadata_files: List of metadata.jsonl files to merge (if None, looks in data_root/dataset_name/)
    """
    dataset_dir = data_root / dataset_name
    output_dir = Path(output_dir)
    
    print("="*80)
    print(f"Preparing {dataset_name} Dataset")
    print("="*80)
    print(f"Source: {dataset_dir}")
    print(f"Output: {output_dir}")
    print("="*80)
    
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")
    
    # Find metadata files
    if metadata_files is None:
        # Look for metadata.jsonl in the dataset directory
        main_metadata = dataset_dir / "metadata.jsonl"
        if main_metadata.exists():
            metadata_files = [main_metadata]
        else:
            # Look for multiple metadata files (member1.jsonl, member2.jsonl, etc.)
            metadata_files = sorted([
                f for f in dataset_dir.glob("*.jsonl")
                if f.name.startswith("metadata") or "member" in f.name.lower()
            ])
    
    if not metadata_files:
        raise ValueError(f"No metadata files found in {dataset_dir}")
    
    print(f"\nFound {len(metadata_files)} metadata file(s)")
    
    # Find images directory
    images_dir = dataset_dir / "images"
    if not images_dir.exists():
        # Try to find images in subdirectories
        image_dirs = [
            d for d in dataset_dir.iterdir()
            if d.is_dir() and any(f.suffix.lower() in {'.jpg', '.jpeg', '.png'} for f in d.rglob('*'))
        ]
        if image_dirs:
            images_dir = image_dirs[0]
        else:
            raise ValueError(f"No images directory found in {dataset_dir}")
    
    print(f"Found images directory: {images_dir}")
    
    # Create output structure
    output_images_dir = output_dir / "images"
    output_metadata = output_dir / "metadata.jsonl"
    
    # Merge metadata
    print("\nMerging metadata files...")
    num_entries = merge_metadata_files(metadata_files, output_metadata)
    
    # Copy images
    print("\nCopying images...")
    num_images = copy_images([images_dir], output_images_dir)
    
    # Verify consistency
    print("\nVerifying dataset...")
    with open(output_metadata, 'r', encoding='utf-8') as f:
        metadata_entries = [json.loads(line) for line in f if line.strip()]
    
    # Check if all images in metadata exist
    missing_images = []
    for entry in metadata_entries:
        image_path = output_images_dir / entry["file_name"]
        if not image_path.exists():
            missing_images.append(entry["file_name"])
    
    if missing_images:
        print(f"Warning: {len(missing_images)} images referenced in metadata are missing")
        if len(missing_images) <= 10:
            for img in missing_images:
                print(f"  - {img}")
    else:
        print("✓ All images in metadata are present")
    
    print("\n" + "="*80)
    print(f"✓ {dataset_name} dataset prepared successfully!")
    print(f"  Images: {num_images}")
    print(f"  Metadata entries: {num_entries}")
    print(f"  Output: {output_dir}")
    print("="*80)
    
    return output_dir


def prepare_all_datasets(
    data_root: str,
    output_base: str,
    datasets: List[str] = None
):
    """
    Prepare all datasets for training.
    
    Args:
        data_root: Root directory containing dataset folders
        output_base: Base output directory
        datasets: List of dataset names to prepare (None = all)
    """
    data_root = Path(data_root)
    output_base = Path(output_base)
    
    if datasets is None:
        datasets = ["FFHQ", "EasyPortrait", "LAION_Face"]
    
    print("="*80)
    print("Preparing All Datasets for Training")
    print("="*80)
    print(f"Data root: {data_root}")
    print(f"Output base: {output_base}")
    print(f"Datasets: {', '.join(datasets)}")
    print("="*80)
    
    prepared_datasets = {}
    
    for dataset_name in datasets:
        try:
            output_dir = prepare_dataset(
                dataset_name=dataset_name,
                data_root=data_root,
                output_dir=output_base / dataset_name
            )
            prepared_datasets[dataset_name] = output_dir
        except Exception as e:
            print(f"\n✗ Error preparing {dataset_name}: {e}")
            continue
    
    print("\n" + "="*80)
    print("All Datasets Prepared!")
    print("="*80)
    for name, path in prepared_datasets.items():
        print(f"  {name}: {path}")
    print("\nNext step: Run training with the prepared datasets")
    print("="*80)
    
    return prepared_datasets


def merge_all_datasets(
    dataset_dirs: List[Path],
    output_dir: Path
):
    """
    Merge all prepared datasets into a single training dataset.
    
    Args:
        dataset_dirs: List of prepared dataset directories
        output_dir: Output directory for merged dataset
    """
    output_dir = Path(output_dir)
    output_images_dir = output_dir / "images"
    output_metadata = output_dir / "metadata.jsonl"
    
    print("="*80)
    print("Merging All Datasets")
    print("="*80)
    
    # Collect all metadata files
    metadata_files = []
    image_dirs = []
    
    for dataset_dir in dataset_dirs:
        dataset_dir = Path(dataset_dir)
        metadata_file = dataset_dir / "metadata.jsonl"
        images_dir = dataset_dir / "images"
        
        if metadata_file.exists():
            metadata_files.append(metadata_file)
        if images_dir.exists():
            image_dirs.append(images_dir)
    
    # Merge
    print("\nMerging metadata...")
    num_entries = merge_metadata_files(metadata_files, output_metadata)
    
    print("\nCopying images...")
    num_images = copy_images(image_dirs, output_images_dir)
    
    print("\n" + "="*80)
    print(f"✓ Merged dataset created!")
    print(f"  Total images: {num_images}")
    print(f"  Total entries: {num_entries}")
    print(f"  Output: {output_dir}")
    print("="*80)
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data from team member outputs"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./Training/data",
        help="Root directory containing dataset folders (FFHQ, EasyPortrait, LAION_Face)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./Training/data/prepared",
        help="Output directory for prepared datasets"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        choices=["FFHQ", "EasyPortrait", "LAION_Face"],
        help="Datasets to prepare (default: all)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all prepared datasets into one"
    )
    
    args = parser.parse_args()
    
    # Prepare individual datasets
    prepared = prepare_all_datasets(
        data_root=args.data_root,
        output_base=args.output,
        datasets=args.datasets
    )
    
    # Merge if requested
    if args.merge:
        print("\n" + "="*80)
        print("Merging All Datasets")
        print("="*80)
        merge_all_datasets(
            dataset_dirs=list(prepared.values()),
            output_dir=Path(args.output) / "merged"
        )


if __name__ == "__main__":
    main()
