# LAION-Face Dataset Preparation

This directory contains scripts to download, extract, and prepare the LAION-Face subset for caption generation and finetuning.

## Dataset Information

- **Size**: 156,000 images (filtered subset)
- **Source**: https://huggingface.co/datasets/FacePerceiver/laion-face
- **Characteristics**: High-quality face crops from LAION dataset

## Quick Start

### 1. Download from HuggingFace

```bash
python download_and_prepare.py \
    --download \
    --download-dir ./downloads \
    --num-samples 156000
```

### 2. Prepare for Training

```bash
python download_and_prepare.py \
    --prepare ./downloads/images \
    --output ./laion_face_processed \
    --num-samples 156000
```

## Complete Pipeline

```bash
# Step 1: Download from HuggingFace (156k subset)
python download_and_prepare.py \
    --download \
    --download-dir ./downloads \
    --num-samples 156000

# Step 2: Prepare with quality filtering
python download_and_prepare.py \
    --prepare ./downloads/images \
    --output ./laion_face_processed \
    --num-samples 156000
```

## Quality Filtering

The preparation script automatically filters images by:
- **Minimum size**: 512×512 pixels
- **Maximum size**: 2048×2048 pixels
- **Format validation**: Only valid image files

To skip quality filtering:
```bash
python download_and_prepare.py \
    --prepare ./downloads/images \
    --output ./laion_face_processed \
    --no-filter
```

## Output Structure

After preparation, you'll have:

```
laion_face_processed/
    images/
        laion_face_000000.jpg
        laion_face_000001.jpg
        ...
    metadata.jsonl  (empty captions, to be filled by caption generation)
```

## Next Steps

1. Generate captions using the caption generation pipeline
2. The dataset will be ready for finetuning

## Notes

- Images are downloaded from HuggingFace datasets
- Quality filtering ensures only suitable images are included
- Subset size (156k) matches the paper's specifications
- Images are renamed for consistency

