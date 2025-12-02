# EasyPortrait Dataset Preparation

This directory contains scripts to download, extract, and prepare the EasyPortrait dataset for caption generation and finetuning.

## Dataset Information

- **Size**: 39,000 well-framed portrait images
- **Source**: https://github.com/hukenovs/easyportrait
- **Characteristics**: Consistent lighting, minimal occlusion

## Quick Start

### 1. Download Dataset

```bash
# Option 1: Clone the repository
git clone https://github.com/hukenovs/easyportrait.git

# Option 2: Download manually from the GitHub repository
```

### 2. Extract/Organize Images

```bash
python download_and_prepare.py --extract /path/to/easyportrait/repo
```

### 3. Prepare for Training

```bash
python download_and_prepare.py \
    --prepare /path/to/organized/images \
    --output ./easyportrait_processed
```

## Complete Pipeline

```bash
# Step 1: Organize images from source
python download_and_prepare.py --extract ./easyportrait

# Step 2: Prepare for training
python download_and_prepare.py \
    --prepare ./easyportrait_organized \
    --output ./easyportrait_processed
```

## Output Structure

After preparation, you'll have:

```
easyportrait_processed/
    images/
        image1.jpg
        image2.jpg
        ...
    metadata.jsonl  (empty captions, to be filled by caption generation)
```

## Next Steps

1. Generate captions using the caption generation pipeline
2. The dataset will be ready for finetuning

## Notes

- Images are organized from various subdirectories into a single images folder
- Duplicate filenames are automatically handled
- Original image sizes are preserved (no resizing)

