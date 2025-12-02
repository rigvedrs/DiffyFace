# FFHQ Dataset Preparation

This directory contains scripts to download, extract, and prepare the FFHQ dataset for caption generation and finetuning.

## Dataset Information

- **Size**: 70,000 high-resolution face images
- **Original Resolution**: 1024×1024
- **Target Resolution**: 768×768 (resized for training)
- **Source**: https://github.com/NVlabs/ffhq-dataset
- **Download Size**: ~89GB (images) + ~254MB (metadata)

## Quick Start

### 1. Download Dataset (Official Method - Recommended)

```bash
# Clone repository and download using official script
python download_and_prepare.py --clone-and-download --download-dir ./downloads
```

This will:
- Clone the FFHQ repository from GitHub
- Run the official `download_ffhq.py` script with `--json --images` flags
- Download JSON metadata (~254MB) and 1024×1024 images (~89GB)
- Images will be saved in nested structure: `images1024x1024/00000/00000.png`

**Note:** The download may take several hours depending on your internet connection.

### 2. Prepare for Training

```bash
# Prepare dataset (resize to 768×768 and organize)
python download_and_prepare.py \
    --prepare ./downloads/ffhq-dataset/images1024x1024 \
    --output ./ffhq_processed \
    --target-size 768
```

## Complete Pipeline

```bash
# Step 1: Download using official method
python download_and_prepare.py --clone-and-download --download-dir ./downloads

# Step 2: Prepare (resize and organize)
python download_and_prepare.py \
    --prepare ./downloads/ffhq-dataset/images1024x1024 \
    --output ./ffhq_processed \
    --target-size 768
```

## Output Structure

After preparation, you'll have:

```
ffhq_processed/
    images/
        image1.jpg  (768×768)
        image2.jpg  (768×768)
        ...
    metadata.jsonl  (empty captions, to be filled by caption generation)
```

## Next Steps

1. Generate captions using the caption generation pipeline
2. The dataset will be ready for finetuning

## Notes

- Images are automatically resized from 1024×1024 to 768×768
- Center crop is applied to maintain square aspect ratio
- Original quality is preserved (JPEG quality 95)

