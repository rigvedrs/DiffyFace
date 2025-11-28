# Place Your Data Here

## Instructions for Team Members

After generating captions for your assigned batch, place your data in the appropriate dataset folder.

### Structure

```
Training/data/
├── FFHQ/
│   ├── images/          ← Place all FFHQ images here (from all members)
│   └── metadata.jsonl   ← Single merged metadata file (from all members)
├── EasyPortrait/
│   ├── images/          ← Place all EasyPortrait images here
│   └── metadata.jsonl   ← Single merged metadata file
└── LAION_Face/
    ├── images/          ← Place all LAION-Face images here
    └── metadata.jsonl   ← Single merged metadata file
```

### How to Merge Member Data

Each team member generates their own metadata.jsonl file. To create a single metadata.jsonl:

**Option 1: Manual merge (simple)**
```bash
# Concatenate all member metadata files
cat member1_metadata.jsonl member2_metadata.jsonl member3_metadata.jsonl member4_metadata.jsonl member5_metadata.jsonl > Training/data/FFHQ/metadata.jsonl
```

**Option 2: Use the preparation script**
```bash
# Place individual metadata files in the dataset folder
# e.g., Training/data/FFHQ/member1.jsonl, member2.jsonl, etc.
# The script will automatically merge them
python Training/prepare_training_data.py
```

### Required Structure

Each dataset folder should have:

```
{DATASET}/
    images/
        image1.jpg
        image2.jpg
        ...
    metadata.jsonl
```

### Metadata.jsonl Format

Each line should be a JSON object:

```json
{"file_name": "image1.jpg", "text": "A detailed caption describing the person..."}
{"file_name": "image2.jpg", "text": "Another detailed caption..."}
```

### Notes

- All images from all team members should be in the same `images/` folder
- The `metadata.jsonl` file should contain entries from all team members
- File names in metadata.jsonl must match actual image file names
- The training script will automatically verify and prepare the data
