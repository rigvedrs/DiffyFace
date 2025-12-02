"""
Data Loaders for DiffyFace Dataset

Supports two data formats:
1. HuggingFace datasets with image and text columns
2. ImageFolder format with metadata.jsonl file
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets package not installed. Install with: pip install datasets")


class ImageFolderWithMetadata(Dataset):
    """
    Dataset loader for ImageFolder format with metadata.jsonl file.
    
    Expected structure:
        dataset_root/
            images/
                image1.jpg
                image2.jpg
                ...
            metadata.jsonl
    
    metadata.jsonl format (one JSON object per line):
        {"file_name": "image1.jpg", "text": "caption text"}
    """
    
    def __init__(
        self,
        root_dir: str,
        metadata_file: str = "metadata.jsonl",
        transform: Optional[transforms.Compose] = None,
        image_size: int = 768,
        center_crop: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing images and metadata
            metadata_file: Name of the metadata JSONL file
            transform: Optional torchvision transforms (if None, default transforms are used)
            image_size: Target image size for resizing
            center_crop: Whether to apply center crop
        """
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.metadata_file = self.root_dir / metadata_file
        
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.metadata_file.exists():
            raise ValueError(f"Metadata file not found: {self.metadata_file}")
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Setup transforms
        if transform is None:
            self.transform = self._get_default_transform(image_size, center_crop)
        else:
            self.transform = transform
    
    def _load_metadata(self) -> List[Dict[str, str]]:
        """Load metadata from JSONL file."""
        metadata = []
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    metadata.append(json.loads(line))
        return metadata
    
    def _get_default_transform(self, image_size: int, center_crop: bool) -> transforms.Compose:
        """Get default preprocessing transforms."""
        transform_list = [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
        
        if center_crop:
            transform_list.append(transforms.CenterCrop(image_size))
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample."""
        item = self.metadata[idx]
        file_name = item["file_name"]
        text = item["text"]
        
        # Load image
        image_path = self.images_dir / file_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            "pixel_values": image,
            "text": text,
            "file_name": file_name
        }


class HuggingFaceDatasetWrapper(Dataset):
    """
    Wrapper for HuggingFace datasets with image and text columns.
    
    Automatically handles image loading and preprocessing.
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        image_column: str = "image",
        text_column: str = "text",
        transform: Optional[transforms.Compose] = None,
        image_size: int = 768,
        center_crop: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the HuggingFace dataset wrapper.
        
        Args:
            dataset_name: HuggingFace dataset name or path
            split: Dataset split to use
            image_column: Name of the image column
            text_column: Name of the text column
            transform: Optional torchvision transforms
            image_size: Target image size
            center_crop: Whether to apply center crop
            cache_dir: Directory to cache the dataset
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets package is required. Install with: pip install datasets")
        
        # Load dataset
        self.dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        self.image_column = image_column
        self.text_column = text_column
        
        # Setup transforms
        if transform is None:
            self.transform = self._get_default_transform(image_size, center_crop)
        else:
            self.transform = transform
    
    def _get_default_transform(self, image_size: int, center_crop: bool) -> transforms.Compose:
        """Get default preprocessing transforms."""
        transform_list = [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
        
        if center_crop:
            transform_list.append(transforms.CenterCrop(image_size))
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample."""
        item = self.dataset[idx]
        
        # Get image and text
        image = item[self.image_column]
        text = item[self.text_column]
        
        # Convert PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            "pixel_values": image,
            "text": text,
            "file_name": item.get("file_name", f"image_{idx}.jpg")
        }


def create_metadata_jsonl(
    image_dir: str,
    output_file: str,
    captions: Optional[Dict[str, str]] = None
) -> None:
    """
    Create a metadata.jsonl file from a directory of images.
    
    Args:
        image_dir: Directory containing images
        output_file: Path to output JSONL file
        captions: Optional dictionary mapping image names to captions
    """
    image_dir = Path(image_dir)
    output_file = Path(output_file)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    # Create metadata entries
    metadata = []
    for image_file in image_files:
        entry = {
            "file_name": image_file.name,
            "text": captions.get(image_file.name, "") if captions else ""
        }
        metadata.append(entry)
    
    # Write to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Created metadata.jsonl with {len(metadata)} entries: {output_file}")


def prepare_imagefolder_dataset(
    source_dir: str,
    output_dir: str,
    metadata_file: str = "metadata.jsonl"
) -> None:
    """
    Prepare an ImageFolder dataset structure from a source directory.
    Handles both flat and nested directory structures.
    
    Args:
        source_dir: Source directory with images (can contain nested subdirectories)
        output_dir: Output directory for organized dataset
        metadata_file: Name of metadata file to create
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    
    # Create output directories
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images and create metadata (handles nested directories)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = [
        f for f in source_dir.rglob('*')
        if f.suffix.lower() in image_extensions and f.is_file()
    ]
    
    import shutil
    metadata = []
    for image_file in image_files:
        # Get relative path from source_dir to handle nested structures
        rel_path = image_file.relative_to(source_dir)
        
        # If image is directly in source_dir, use original name
        # Otherwise, create unique name from relative path
        if rel_path.parent == Path('.'):
            dest_file = images_dir / image_file.name
            file_name = image_file.name
        else:
            # Create unique filename from nested path (e.g., "00000_00000.png")
            unique_name = str(rel_path).replace('/', '_').replace('\\', '_')
            dest_file = images_dir / unique_name
            file_name = unique_name
        
        # Copy image
        shutil.copy2(image_file, dest_file)
        
        # Add to metadata (caption will be empty, to be filled later)
        metadata.append({
            "file_name": file_name,
            "text": ""
        })
    
    # Write metadata
    metadata_path = output_dir / metadata_file
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Prepared dataset: {len(metadata)} images in {output_dir}")
    print(f"Metadata file: {metadata_path}")

