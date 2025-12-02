"""
Dataset-Specific Processors

Processors for FFHQ, EasyPortrait, and LAION-Face datasets.
Handles dataset-specific formatting and preparation.
"""

import json
import shutil
from pathlib import Path
from typing import List, Optional

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loaders import create_metadata_jsonl, prepare_imagefolder_dataset
from groq_caption_generator import GroqCaptionGenerator


class FFHQProcessor:
    """Processor for FFHQ dataset (70k images, 1024x1024 -> 768x768)"""
    
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_structure(self):
        """Prepare ImageFolder structure from FFHQ source."""
        print("Preparing FFHQ dataset structure...")
        prepare_imagefolder_dataset(
            source_dir=str(self.source_dir),
            output_dir=str(self.output_dir),
            metadata_file="metadata.jsonl"
        )
        print("✓ FFHQ structure prepared")
    
    def generate_captions(self, api_key: Optional[str] = None, start_idx: int = 0, end_idx: Optional[int] = None):
        """Generate captions for FFHQ images."""
        images_dir = self.output_dir / "images"
        metadata_file = self.output_dir / "metadata.jsonl"
        
        # Load existing metadata
        image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        
        if end_idx is None:
            end_idx = len(image_files)
        
        generator = GroqCaptionGenerator(api_key=api_key)
        
        # Update metadata with captions
        results = []
        for idx in range(start_idx, min(end_idx, len(image_files))):
            image_file = image_files[idx]
            try:
                caption = generator.generate_caption(str(image_file))
                results.append({
                    "file_name": image_file.name,
                    "text": caption
                })
                print(f"[{idx+1}/{len(image_files)}] {image_file.name}: {caption[:80]}...")
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
                continue
        
        # Update metadata file
        with open(metadata_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"✓ Generated {len(results)} captions for FFHQ")


class EasyPortraitProcessor:
    """Processor for EasyPortrait dataset (39k images)"""
    
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_structure(self):
        """Prepare ImageFolder structure from EasyPortrait source."""
        print("Preparing EasyPortrait dataset structure...")
        prepare_imagefolder_dataset(
            source_dir=str(self.source_dir),
            output_dir=str(self.output_dir),
            metadata_file="metadata.jsonl"
        )
        print("✓ EasyPortrait structure prepared")
    
    def generate_captions(self, api_key: Optional[str] = None, start_idx: int = 0, end_idx: Optional[int] = None):
        """Generate captions for EasyPortrait images."""
        images_dir = self.output_dir / "images"
        metadata_file = self.output_dir / "metadata.jsonl"
        
        image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        
        if end_idx is None:
            end_idx = len(image_files)
        
        generator = GroqCaptionGenerator(api_key=api_key)
        
        results = []
        for idx in range(start_idx, min(end_idx, len(image_files))):
            image_file = image_files[idx]
            try:
                caption = generator.generate_caption(str(image_file))
                results.append({
                    "file_name": image_file.name,
                    "text": caption
                })
                print(f"[{idx+1}/{len(image_files)}] {image_file.name}: {caption[:80]}...")
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
                continue
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"✓ Generated {len(results)} captions for EasyPortrait")


class LAIONFaceProcessor:
    """Processor for LAION-Face subset (156k images)"""
    
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_structure(self):
        """Prepare ImageFolder structure from LAION-Face source."""
        print("Preparing LAION-Face dataset structure...")
        prepare_imagefolder_dataset(
            source_dir=str(self.source_dir),
            output_dir=str(self.output_dir),
            metadata_file="metadata.jsonl"
        )
        print("✓ LAION-Face structure prepared")
    
    def generate_captions(self, api_key: Optional[str] = None, start_idx: int = 0, end_idx: Optional[int] = None):
        """Generate captions for LAION-Face images."""
        images_dir = self.output_dir / "images"
        metadata_file = self.output_dir / "metadata.jsonl"
        
        image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        
        if end_idx is None:
            end_idx = len(image_files)
        
        generator = GroqCaptionGenerator(api_key=api_key)
        
        results = []
        for idx in range(start_idx, min(end_idx, len(image_files))):
            image_file = image_files[idx]
            try:
                caption = generator.generate_caption(str(image_file))
                results.append({
                    "file_name": image_file.name,
                    "text": caption
                })
                print(f"[{idx+1}/{len(image_files)}] {image_file.name}: {caption[:80]}...")
            except Exception as e:
                print(f"Error processing {image_file.name}: {e}")
                continue
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"✓ Generated {len(results)} captions for LAION-Face")


def merge_datasets(
    dataset_dirs: List[str],
    output_dir: str,
    metadata_file: str = "metadata.jsonl"
):
    """
    Merge multiple processed datasets into a single dataset.
    
    Args:
        dataset_dirs: List of dataset directories to merge
        output_dir: Output directory for merged dataset
        metadata_file: Name of metadata file
    """
    output_dir = Path(output_dir)
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    all_metadata = []
    
    for dataset_dir in dataset_dirs:
        dataset_dir = Path(dataset_dir)
        images_dir = dataset_dir / "images"
        metadata_path = dataset_dir / metadata_file
        
        if not metadata_path.exists():
            print(f"Warning: Metadata not found in {dataset_dir}, skipping...")
            continue
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    # Copy image
                    src_image = images_dir / entry["file_name"]
                    if src_image.exists():
                        dst_image = output_images_dir / entry["file_name"]
                        shutil.copy2(src_image, dst_image)
                        all_metadata.append(entry)
    
    # Write merged metadata
    output_metadata = output_dir / metadata_file
    with open(output_metadata, 'w', encoding='utf-8') as f:
        for entry in all_metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✓ Merged {len(all_metadata)} images from {len(dataset_dirs)} datasets")
    print(f"Output: {output_dir}")

