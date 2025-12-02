# Training Data Structure

This document describes the expected data structure for training.

## Expected Structure

```
Training/data/
├── FFHQ/
│   ├── images/          # All FFHQ images (from all team members)
│   │   ├── image_00001.jpg
│   │   ├── image_00002.jpg
│   │   └── ...
│   └── metadata.jsonl   # Single merged metadata file (from all team members)
├── EasyPortrait/
│   ├── images/          # All EasyPortrait images
│   │   ├── image_00001.jpg
│   │   └── ...
│   └── metadata.jsonl   # Single merged metadata file
└── LAION_Face/
    ├── images/          # All LAION-Face images
    │   ├── image_00001.jpg
    │   └── ...
    └── metadata.jsonl   # Single merged metadata file
```

## Data Collection Process

1. **Each team member generates captions** for their assigned batch
2. **All images are collected** into the `images/` folder
3. **All metadata files are merged** into a single `metadata.jsonl`

### Example: FFHQ Dataset (70,000 images)

- **Member 1**: Processes images 0-14,000 → generates `member1_metadata.jsonl`
- **Member 2**: Processes images 14,000-28,000 → generates `member2_metadata.jsonl`
- **Member 3**: Processes images 28,000-42,000 → generates `member3_metadata.jsonl`
- **Member 4**: Processes images 42,000-56,000 → generates `member4_metadata.jsonl`
- **Member 5**: Processes images 56,000-70,000 → generates `member5_metadata.jsonl`

**Then merge:**
```bash
# Copy all images to Training/data/FFHQ/images/
# Merge all metadata files:
cat member1_metadata.jsonl member2_metadata.jsonl member3_metadata.jsonl member4_metadata.jsonl member5_metadata.jsonl > Training/data/FFHQ/metadata.jsonl
```

## Metadata Format

Each `metadata.jsonl` file contains one JSON object per line:

```json
{"file_name": "image_00001.jpg", "text": "A happy 25 year old male with blond hair and a french beard smiles with visible teeth."}
{"file_name": "image_00002.jpg", "text": "A young woman with dark hair and glasses looks serious."}
```

## After Preparation

After running the data preparation script, the structure becomes:

```
Training/data/prepared/
├── FFHQ/
│   ├── images/          # All 70k images
│   └── metadata.jsonl   # All captions merged
├── EasyPortrait/
│   ├── images/          # All 39k images
│   └── metadata.jsonl   # All captions merged
├── LAION_Face/
│   ├── images/          # All 156k images
│   └── metadata.jsonl   # All captions merged
└── merged/              # All datasets combined
    ├── images/          # All images from all datasets
    └── metadata.jsonl   # All captions from all datasets
```

## Notes

- Each dataset has a **single** `metadata.jsonl` file (merged from all members)
- All images are in a **single** `images/` folder per dataset
- The preparation script can handle merging if you place individual metadata files in the dataset folder
- Images are verified against metadata entries during preparation
- Duplicate file names are automatically handled
