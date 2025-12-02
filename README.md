# DiffyFace: Text-to-Face Generation with LoRA Fine-Tuned Stable Diffusion

![Python version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green)
![CUDA](https://img.shields.io/badge/CUDA-Required-red)

A comprehensive system for generating high-quality face images from text descriptions using Stable Diffusion 2.1 fine-tuned with LoRA (Low-Rank Adaptation). This project includes complete pipelines for dataset preparation, model training, and inference.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Training](#training)
- [Inference](#inference)
- [Model Information](#model-information)
- [Limitations and Ethics](#limitations-and-ethics)
- [Troubleshooting](#troubleshooting)

## Overview

DiffyFace is a LoRA-finetuned version of Stable Diffusion 2.1 specifically optimized for generating realistic face images from detailed text prompts. The model was trained on a curated dataset of 265,000 face images from FFHQ, EasyPortrait, and LAION-Face datasets, each paired with synthetically generated detailed captions.

### What Makes This Project Special

- **Distributed Data Processing**: Designed for team collaboration - process 265k images across multiple team members in parallel
- **Automated Caption Generation**: Uses Groq API to generate detailed facial descriptions automatically
- **Complete Pipeline**: From dataset download to model training to inference - everything is included
- **Multiple Interfaces**: Use via Jupyter notebook, Streamlit web app, or Python API
- **Production Ready**: Pre-trained model available on Hugging Face, ready to use

### Key Components

1. **Data Pipeline**: Automated dataset preparation and caption generation using Groq API, designed for distributed team processing
2. **Training System**: Distributed training support with LoRA fine-tuning for both UNet and text encoder
3. **Inference Tools**: Multiple interfaces including Jupyter notebook, CLI, and Streamlit web app
4. **Model Hosting**: Pre-trained model available on Hugging Face Hub

## Quick Start

**Just want to generate faces?** (Skip to inference)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download pretrained model
python Generation/download.py

# 3. Run inference (choose one):
# Option A: Jupyter notebook (best for Colab)
# Open Generation/inference.ipynb

# Option B: Streamlit web app
bash run_streamlit.sh

# Option C: Command line
python Generation/generate.py
```

**Want to train your own model?** (Full pipeline)
1. **Download datasets** → See [Data Pipeline - Step 1](#step-1-download-and-prepare-datasets)
2. **Generate captions** → See [Data Pipeline - Step 2](#step-2-generate-captions-distributed-processing)
3. **Merge results** → See [Data Pipeline - Step 3](#step-3-merge-results-from-all-team-members)
4. **Train model** → See [Training](#training)

**Working with a team?** 
- Use `Data/data_pipeline.ipynb` for distributed caption generation
- Each member processes a different batch of images
- Merge results when everyone is done

## Features

- ✅ **Pretrained Model**: Pre-trained LoRA weights available on [Hugging Face](https://huggingface.co/rigvedrs/DiffyFace)
- ✅ **Multiple Inference Interfaces**: Jupyter notebook, command-line, and Streamlit web app
- ✅ **Automated Data Pipeline**: Scripts for dataset download, preparation, and caption generation
- ✅ **Distributed Training**: Support for team-based distributed data processing and training
- ✅ **Google Colab Support**: Ready-to-run notebook for Colab environments
- ✅ **Comprehensive Documentation**: Complete guides for all workflows

## Installation

### System Requirements

- **GPU**: NVIDIA GPU with CUDA support (required for inference and training)
- **CUDA**: Version 11.7 or higher
- **VRAM**: At least 8GB recommended
- **Python**: 3.8 or higher

### Quick Setup

```bash
# Create conda environment
conda create -n diffyface python=3.9 -y
conda activate diffyface

# Install dependencies
pip install -r requirements.txt

# Set up Groq API key (for caption generation - optional if only using pretrained model)
export GROQ_API_KEY="your-api-key-here"  # Get one from https://console.groq.com

# Download pretrained LoRA weights
python Generation/download.py
```

**Note**: The Groq API key is only needed if you plan to generate captions for new datasets. For inference with the pretrained model, you don't need it.

### Dependencies

Key dependencies include:
- `diffusers==0.27.2`: Stable Diffusion pipeline
- `transformers==4.40.1`: Model loading and tokenization
- `accelerate==0.29.3`: Distributed training
- `peft==0.10.0`: LoRA implementation
- `huggingface-hub==0.22.2`: Model downloading
- `streamlit>=1.28.0`: Web interface
- `groq`: Groq API client (for caption generation - install with `pip install groq`)

See `requirements.txt` for complete list.

### Getting a Groq API Key (Optional)

If you want to generate captions for new datasets:
1. Go to https://console.groq.com
2. Sign up for a free account
3. Create an API key
4. Set it as an environment variable: `export GROQ_API_KEY="your-key-here"`

## Project Structure

```
DiffyFace/
├── Data/                          # Data preparation pipeline
│   ├── FFHQ/                      # FFHQ dataset scripts
│   ├── EasyPortrait/              # EasyPortrait dataset scripts
│   ├── LAION_Face/                # LAION-Face dataset scripts
│   ├── data_pipeline.ipynb        # ⭐ Notebook for distributed caption generation
│   ├── batch_caption_generator.py # Distributed caption generation (CLI)
│   ├── groq_caption_generator.py  # Groq API integration
│   ├── data_loaders.py            # Dataset loaders
│   ├── dataset_processors.py      # Dataset processors
│   └── prepare_all_datasets.py    # Master script to prepare all datasets
├── Training/                      # Training scripts
│   ├── data/                      # Training data directory
│   │   ├── FFHQ/
│   │   ├── EasyPortrait/
│   │   └── LAION_Face/
│   ├── DATA_STRUCTURE.md          # Data organization guide
│   ├── train.py                   # Main training script
│   ├── finetune_lora.py           # Core LoRA training
│   └── prepare_training_data.py   # Data preparation
├── Generation/                    # Inference and generation
│   ├── model.py                   # Core model class
│   ├── app.py                     # Streamlit web interface
│   ├── generate.py                # CLI generation script
│   ├── download.py                # Download checkpoints
│   ├── inference.ipynb            # ⭐ Jupyter notebook (Colab-ready)
│   └── download_and_upload_model.py # Model upload utility
├── checkpoints/                   # Model checkpoints
│   └── lora30k/                   # LoRA weights
├── models/                        # Base model cache
│   └── stable-diffusion-2-1/      # Local model storage
├── Images/                        # Generated images
│   ├── Saved_images/              # Saved outputs
│   └── colab/                     # Colab outputs
├── run_streamlit.sh               # Helper script to run Streamlit app
└── requirements.txt               # Python dependencies
```

**Key Files:**
- `Data/data_pipeline.ipynb` - **Distribute this to team members** for caption generation
- `Generation/inference.ipynb` - **Use this for inference** in Google Colab
- `run_streamlit.sh` - Helper script to run the web interface

## Data Pipeline

The data pipeline prepares 265,000 face images from three datasets and generates detailed captions for each image. This process was designed to be distributed across multiple team members working in parallel.

### Complete Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE WORKFLOW                        │
└─────────────────────────────────────────────────────────────────┘

Step 1: Download & Prepare
├── Download FFHQ (70k images, ~89GB)
├── Download EasyPortrait (39k images)
├── Download LAION-Face (156k images)
└── Organize into ImageFolder format with metadata.jsonl

Step 2: Generate Captions (Distributed)
├── Team Member 1: Process images 0-14,000
├── Team Member 2: Process images 14,000-28,000
├── Team Member 3: Process images 28,000-42,000
├── Team Member 4: Process images 42,000-56,000
└── Team Member 5: Process images 56,000-70,000
    (Each uses Data/data_pipeline.ipynb with different START_INDEX/END_INDEX)

Step 3: Merge Results
├── Collect all images into single folders
├── Merge all metadata.jsonl files
└── Verify data integrity

Step 4: Ready for Training
└── Organized dataset with 265k image-caption pairs
```

### Overview: What the Pipeline Does

1. **Download** three face datasets (FFHQ, EasyPortrait, LAION-Face)
2. **Prepare** images into a standard format (ImageFolder structure)
3. **Generate captions** using Groq API (distributed across team members)
4. **Merge** results from all team members
5. **Ready for training** with image-caption pairs

### Dataset Overview

| Dataset | Images | Source | Resolution |
|---------|--------|--------|------------|
| FFHQ | 70,000 | [NVlabs](https://github.com/NVlabs/ffhq-dataset) | 768×768 |
| EasyPortrait | 39,000 | [GitHub](https://github.com/hukenovs/easyportrait) | Variable |
| LAION-Face | 156,000 | [Hugging Face](https://huggingface.co/datasets/FacePerceiver/laion-face) | Filtered subset |
| **Total** | **265,000** | | |

### Step-by-Step Pipeline

#### Step 1: Download and Prepare Datasets

Each dataset needs to be downloaded and organized into a standard format. The output structure for each dataset is:

```
dataset_processed/
    images/
        image1.jpg
        image2.jpg
        ...
    metadata.jsonl  (initially empty, captions added in Step 2)
```

**Option A: Prepare All Datasets at Once (Recommended)**

Use the master script to prepare all datasets:
```bash
python Data/prepare_all_datasets.py \
    --ffhq ./downloads/ffhq-dataset/images1024x1024 \
    --easyportrait ./easyportrait_organized \
    --laion-face ./downloads/laion_face/images \
    --output ./data/processed
```

**Option B: Prepare Each Dataset Individually**

**FFHQ (70k images, ~89GB download):**
```bash
# Download using official script
python Data/FFHQ/download_and_prepare.py --clone-and-download --download-dir ./downloads

# Prepare for training (resizes 1024×1024 → 768×768)
python Data/FFHQ/download_and_prepare.py \
    --prepare ./downloads/ffhq-dataset/images1024x1024 \
    --output ./data/processed/ffhq \
    --target-size 768
```

**EasyPortrait (39k images):**
```bash
git clone https://github.com/hukenovs/easyportrait.git
python Data/EasyPortrait/download_and_prepare.py --extract ./easyportrait
python Data/EasyPortrait/download_and_prepare.py \
    --prepare ./easyportrait_organized \
    --output ./data/processed/easyportrait
```

**LAION-Face (156k images):**
```bash
python Data/LAION_Face/download_and_prepare.py \
    --download --download-dir ./downloads --num-samples 156000
python Data/LAION_Face/download_and_prepare.py \
    --prepare ./downloads/images \
    --output ./data/processed/laion_face
```

#### Step 2: Generate Captions (Distributed Processing)

This is where the distributed workflow comes in. With 265,000 images, generating captions takes a long time. The solution: **distribute the work across team members**.

**How It Works:**
- Each team member gets a copy of `Data/data_pipeline.ipynb`
- Each member processes a different range of images (e.g., Member 1: images 0-14,000, Member 2: images 14,000-28,000, etc.)
- All members work in parallel on Google Colab
- Results are saved incrementally to `metadata.jsonl`
- Later, all results are merged into one file

**Setup (One-time):**
```bash
# Set your Groq API key (get one from https://console.groq.com)
export GROQ_API_KEY="your-api-key-here"
```

**For Each Team Member:**

1. **Open `Data/data_pipeline.ipynb` in Google Colab**

2. **Set your batch range** (in the notebook):
   ```python
   DATASET_DIR = "./data/ffhq_processed/images"
   OUTPUT_FILE = "./data/ffhq_processed/metadata.jsonl"
   START_INDEX = 0        # Your starting index
   END_INDEX = 14000      # Your ending index
   ```

3. **Example distribution for FFHQ (70k images) across 5 members:**
   - Member 1: `START_INDEX=0, END_INDEX=14000`
   - Member 2: `START_INDEX=14000, END_INDEX=28000`
   - Member 3: `START_INDEX=28000, END_INDEX=42000`
   - Member 4: `START_INDEX=42000, END_INDEX=56000`
   - Member 5: `START_INDEX=56000, END_INDEX=70000`

4. **Run the notebook cells** - it will:
   - Load images from your range
   - Generate captions using Groq API
   - Save results incrementally to `metadata.jsonl`

**Alternative: Using Command Line Script**

If you prefer not to use the notebook:
```bash
python Data/batch_caption_generator.py \
    ./data/ffhq_processed/images \
    --output ./data/ffhq_processed/metadata.jsonl \
    --start 0 \
    --end 14000
```

**What Captions Look Like:**

Each caption is a detailed description generated by Groq's `llama-4-maverick-17b-128e-instruct` model, including:
- Age, gender, ethnicity
- Facial expression and emotion
- Hair characteristics (color, length, style, hairline)
- Facial features (eyes, nose, mouth, lips, chin, cheekbones)
- Facial hair, accessories (glasses, hats, jewelry)
- Skin tone, makeup, distinctive features

Example caption:
> "A happy 25 year old male with blond hair and a french beard smiles with visible teeth. He has blue eyes, a straight nose, and fair skin tone. The image shows a well-lit portrait with good framing."

**Data Format:**

The final `metadata.jsonl` file contains one JSON object per line:
```json
{"file_name": "00000.png", "text": "A happy 25 year old male with blond hair..."}
{"file_name": "00001.png", "text": "A serious 30 year old female with brown hair..."}
```

#### Step 3: Merge Results from All Team Members

After all team members finish their batches, merge the results:

**Option 1: Simple merge (if all members wrote to same file)**
If all members used the same `metadata.jsonl` file (with append mode), the file is already merged! Just verify all entries are present.

**Option 2: Merge separate files**
If each member created separate files:
```python
from Data.dataset_processors import merge_datasets

# If each member created separate metadata files
merge_datasets(
    dataset_dirs=[
        "./data/ffhq_processed_member1",
        "./data/ffhq_processed_member2",
        "./data/ffhq_processed_member3",
        "./data/ffhq_processed_member4",
        "./data/ffhq_processed_member5"
    ],
    output_dir="./data/ffhq_processed_merged",
    metadata_file="metadata.jsonl"
)
```

**Option 3: Manual merge (using command line)**
```bash
# Combine all metadata.jsonl files
cat member1/metadata.jsonl member2/metadata.jsonl member3/metadata.jsonl \
    member4/metadata.jsonl member5/metadata.jsonl > merged/metadata.jsonl
```

#### Step 4: Verify and Test

Test that the dataset loads correctly:
```python
from Data.data_loaders import ImageFolderWithMetadata
from torch.utils.data import DataLoader

dataset = ImageFolderWithMetadata(
    root_dir="./data/ffhq_processed_merged",
    metadata_file="metadata.jsonl",
    image_size=768
)

print(f"Dataset size: {len(dataset)}")
sample = dataset[0]
print(f"Image shape: {sample['pixel_values'].shape}")
print(f"Caption: {sample['text'][:100]}...")
```

### Key Components Explained

**`Data/data_pipeline.ipynb`** - The notebook distributed to team members
- Contains all the code needed for distributed caption generation
- Each member sets their `START_INDEX` and `END_INDEX`
- Saves results incrementally (so progress isn't lost if interrupted)
- Includes steps to verify and test the dataset

**`Data/groq_caption_generator.py`** - Groq API integration
- Handles API calls to generate captions using Groq's vision model
- Includes retry logic and rate limiting
- Uses the exact caption prompt from the paper
- Supports batch processing with progress tracking

**`Data/batch_caption_generator.py`** - Command-line alternative
- Same functionality as the notebook, but as a script
- Useful for automation or non-Colab environments
- Can be run with command-line arguments for batch ranges

**`Data/dataset_processors.py`** - Dataset-specific processors
- `FFHQProcessor`, `EasyPortraitProcessor`, `LAIONFaceProcessor`
- Handles dataset-specific preparation steps
- `merge_datasets()` function for combining results from multiple sources

**`Data/data_loaders.py`** - PyTorch dataset loaders
- `ImageFolderWithMetadata` - loads images with metadata.jsonl
- `HuggingFaceDatasetWrapper` - wrapper for HuggingFace datasets
- Used during training to load image-caption pairs
- Handles image preprocessing and transforms

**`Data/prepare_all_datasets.py`** - Master script for dataset preparation
- Convenient way to prepare all three datasets at once
- Handles FFHQ, EasyPortrait, and LAION-Face in one command
- Useful when setting up the complete pipeline

### Tips for Distributed Processing

1. **Coordinate with team**: Make sure index ranges don't overlap
2. **Use Google Colab**: Free GPU and easy sharing of notebooks
3. **Save incrementally**: The notebook saves after each image, so progress isn't lost
4. **Monitor progress**: Check the output file periodically to see progress
5. **Handle errors**: If an image fails, the pipeline continues with the next one
6. **Verify counts**: After merging, verify total entries match expected count

## Training

### Data Organization

Before training, organize all team member data into the following structure:

```
Training/data/
├── FFHQ/
│   ├── images/              # All FFHQ images (merged from all members)
│   │   ├── image_00001.jpg
│   │   ├── image_00002.jpg
│   │   └── ...
│   └── metadata.jsonl       # Single merged metadata file
├── EasyPortrait/
│   ├── images/              # All EasyPortrait images
│   └── metadata.jsonl       # Single merged metadata file
└── LAION_Face/
    ├── images/              # All LAION-Face images
    └── metadata.jsonl       # Single merged metadata file
```

**How to Organize Data:**

1. **Collect all images**: Copy all images from all team members into each dataset's `images/` folder
2. **Merge metadata files**: Combine all `metadata.jsonl` files from team members into one file per dataset

**Example for FFHQ (5 team members):**
```bash
# Copy all images to Training/data/FFHQ/images/
# (make sure all 70k images are in one folder)

# Merge all metadata files
cat member1_metadata.jsonl member2_metadata.jsonl member3_metadata.jsonl \
    member4_metadata.jsonl member5_metadata.jsonl > Training/data/FFHQ/metadata.jsonl
```

**Important Notes:**
- Each dataset must have a **single** `metadata.jsonl` file (merged from all members)
- All images must be in a **single** `images/` folder per dataset
- The training script will verify that all images in metadata exist
- Duplicate file names are automatically handled during preparation

### Training Configuration

The model uses LoRA (Low-Rank Adaptation) fine-tuning with:
- **Base Model**: `rigvedrs/Diffy-2-1` (Stable Diffusion 2.1)
- **Resolution**: 768×768
- **LoRA Rank**: 8 (both UNet and text encoder)
- **Learning Rate**: 1e-5 (both components)
- **Training Epochs**: 20
- **Batch Size**: 4 (per device)

### Training Command

**Option 1: Automatic (prepares data + trains)**
```bash
python Training/train.py \
    --prepare-data \
    --merge-datasets \
    --pretrained-model rigvedrs/Diffy-2-1 \
    --output-dir ./Training/output \
    --resolution 768 \
    --train-batch-size 4 \
    --num-train-epochs 20 \
    --learning-rate 1e-5 \
    --learning-rate-text-encoder 1e-5 \
    --rank 8 \
    --text-encoder-rank 8 \
    --gradient-checkpointing
```

**Option 2: Manual (use existing prepared data)**
```bash
# First prepare data
python Training/prepare_training_data.py \
    --data-root ./Training/data \
    --output ./Training/data/prepared \
    --merge

# Then train
python Training/train.py \
    --skip-data-prep \
    --prepared-data-dir ./Training/data/prepared/merged \
    --pretrained-model rigvedrs/Diffy-2-1 \
    --output-dir ./Training/output \
    --resolution 768 \
    --train-batch-size 4 \
    --num-train-epochs 20
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pretrained-model` | `rigvedrs/Diffy-2-1` | Base model path or Hugging Face ID |
| `--resolution` | 768 | Training image resolution |
| `--train-batch-size` | 4 | Batch size per device |
| `--num-train-epochs` | 20 | Number of training epochs |
| `--learning-rate` | 1e-5 | UNet learning rate |
| `--learning-rate-text-encoder` | 1e-5 | Text encoder learning rate |
| `--rank` | 8 | LoRA rank for UNet |
| `--text-encoder-rank` | 8 | LoRA rank for text encoder |
| `--gradient-accumulation-steps` | 1 | Gradient accumulation |
| `--gradient-checkpointing` | False | Enable to save memory |

### Memory Optimization

For limited GPU memory:
```bash
python Training/train.py \
    --train-batch-size 2 \
    --gradient-accumulation-steps 2 \
    --gradient-checkpointing \
    ...
```

### Output

After training, model weights are saved to:
```
Training/output/
    pytorch_lora_weights.safetensors      # LoRA weights
    text_encoder_lora_weights.safetensors # Text encoder LoRA weights
```

## Inference

### Prerequisites

1. **Download LoRA weights:**
```bash
python Generation/download.py
```

This downloads the pretrained weights from [rigvedrs/DiffyFace](https://huggingface.co/rigvedrs/DiffyFace) to `checkpoints/lora30k/`.

2. **Model Loading:**
The inference system automatically loads:
- Base model: `rigvedrs/Diffy-2-1` from Hugging Face Hub
- LoRA weights: `checkpoints/lora30k/pytorch_lora_weights.safetensors`

### Option 1: Jupyter Notebook (Recommended for Colab)

The `Generation/inference.ipynb` notebook is fully automated and works in Google Colab:

1. **Clone repository in Colab:**
```python
!git clone https://github.com/rigvedrs/DiffyFace.git /content/DiffyFace
%cd /content/DiffyFace/Generation
```

2. **Open and run `inference.ipynb`**
   - All dependencies install automatically
   - Checkpoints download automatically
   - Model loads automatically
   - Just edit the prompt and generate!

**Features:**
- Automatic path detection
- Automatic package installation
- Automatic checkpoint downloading
- Batch generation support
- Image saving and display

### Option 2: Streamlit Web Interface

**Run the web app:**

**Method 1: Using the helper script (recommended)**
```bash
# Make sure you're in the project root directory
bash run_streamlit.sh
```

**Method 2: Direct command**
```bash
streamlit run Generation/app.py
```

The app opens at `http://localhost:8501` with:
- Interactive text prompt input
- Negative prompt support
- Adjustable parameters (steps, guidance scale, seed)
- Real-time image generation
- Save and download functionality

**Features:**
- User-friendly web interface
- Real-time generation
- Parameter adjustment
- Image saving to `Images/Saved_images/`
- Direct download capability

**Note**: The `run_streamlit.sh` script automatically activates the conda environment. If using the direct command, make sure your environment is activated first.

### Option 3: Command Line

**Generate a single image:**
```bash
python Generation/generate.py
```

Edit `Generation/generate.py` to customize the prompt and parameters.

**Programmatic usage:**
```python
from Generation.model import DiffyFaceModel

# Initialize model
model = DiffyFaceModel(device="cuda")

# Generate image
image = model.generate(
    prompt="A happy 25 year old male with blond hair and a french beard smiles with visible teeth.",
    negative_prompt="blurry, distorted, low quality",
    num_inference_steps=50,
    seed=42
)

# Save image
image.save("output.png")
```

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | Required | Text description of the face |
| `negative_prompt` | "" | What to avoid in generation |
| `num_inference_steps` | 50 | Denoising steps (20-100) |
| `guidance_scale` | 7.5 | Prompt adherence (1.0-20.0) |
| `seed` | None | Random seed for reproducibility |

### Tips for Better Results

1. **Detailed Prompts**: Be specific about age, gender, hair, facial features, expression
   - Good: "A happy 25 year old male with blond hair and a french beard smiles with visible teeth."
   - Bad: "A person"

2. **Negative Prompts**: Use to avoid unwanted features
   - Example: "blurry, distorted, low quality, deformed, ugly"

3. **Inference Steps**:
   - 20-30: Fast, lower quality
   - 50: Balanced (recommended)
   - 75-100: High quality, slower

4. **Guidance Scale**:
   - 5.0-7.5: More creative, less strict
   - 7.5-10.0: Balanced (recommended)
   - 10.0-15.0: Very strict prompt adherence

## Model Information

### Architecture

- **Base Model**: Stable Diffusion 2.1 (768×768 resolution)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Rank**: 8 (UNet and text encoder)
- **Model Size**: ~5GB base model + ~50MB LoRA weights

### Model Locations

- **Base Model**: `rigvedrs/Diffy-2-1` on Hugging Face Hub
- **LoRA Weights**: `rigvedrs/DiffyFace` on Hugging Face Hub
- **Local Cache**: `~/.cache/huggingface/hub/` (automatic)
- **Local Storage**: `models/stable-diffusion-2-1/` (optional)

### Model Upload

To upload your own trained model to Hugging Face:

```bash
python Generation/download_and_upload_model.py \
    --repo-id your-username/your-model-name \
    --download-only  # First download the base model
```

Then upload:
```bash
python Generation/download_and_upload_model.py \
    --repo-id your-username/your-model-name \
    --upload-only \
    --local-dir ./models/stable-diffusion-2-1
```

## Limitations and Ethics

### Technical Limitations

1. **CUDA Requirement**: The model requires CUDA-enabled GPU. CPU inference is not supported due to architecture differences between SD 2.1 and SD 1.5.

2. **Inherited Biases**: The model inherits biases from Stable Diffusion 2.1 base model, which may manifest in:
   - Skewed representations across ethnicities
   - Gender representation imbalances
   - Stereotypical associations

3. **Dataset Biases**: Training datasets (FFHQ, EasyPortrait, LAION-Face) have their own selection biases that may not represent global diversity.

4. **Caption Generation Bias**: Synthetic captions were generated by automated face analysis models, which themselves may contain biases, potentially leading to:
   - Inaccuracies in facial feature interpretation
   - Underrepresentation of certain demographics
   - Stereotypical descriptions

## Troubleshooting

### Common Issues

**Issue: "CUDA not available"**
- **Solution**: The model requires a CUDA-enabled GPU. Use Google Colab with GPU runtime, or a machine with NVIDIA GPU.

**Issue: "LoRA checkpoint file not found"**
- **Solution**: Run `python Generation/download.py` to download the pretrained weights.

**Issue: "Model not found" or "401 Unauthorized"**
- **Solution**: The model `rigvedrs/Diffy-2-1` should be publicly accessible. If issues persist, check your Hugging Face authentication.

**Issue: "Out of memory" during training**
- **Solution**: 
  - Reduce batch size: `--train-batch-size 2`
  - Enable gradient checkpointing: `--gradient-checkpointing`
  - Increase gradient accumulation: `--gradient-accumulation-steps 2`

**Issue: "Module not found"**
- **Solution**: Install dependencies: `pip install -r requirements.txt`

**Issue: "Path not found" in notebook**
- **Solution**: The notebook auto-detects paths. Ensure the repository is cloned correctly.


