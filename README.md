# DiffyFace: Text-to-Face Generation with LoRA Fine-Tuned Stable Diffusion

![Python version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green)
![CUDA](https://img.shields.io/badge/CUDA-Required-red)

A comprehensive system for generating high-quality face images from text descriptions using Stable Diffusion 2.1 fine-tuned with LoRA (Low-Rank Adaptation). This project includes complete pipelines for dataset preparation, model training, and inference.

## Table of Contents

- [Overview](#overview)
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

### Key Components

1. **Data Pipeline**: Automated dataset preparation and caption generation using Groq API
2. **Training System**: Distributed training support with LoRA fine-tuning for both UNet and text encoder
3. **Inference Tools**: Multiple interfaces including Jupyter notebook, CLI, and Streamlit web app
4. **Model Hosting**: Pre-trained model available on Hugging Face Hub

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

# Download pretrained LoRA weights
python Generation/download.py
```

### Dependencies

Key dependencies include:
- `diffusers==0.27.2`: Stable Diffusion pipeline
- `transformers==4.40.1`: Model loading and tokenization
- `accelerate==0.29.3`: Distributed training
- `peft==0.10.0`: LoRA implementation
- `huggingface-hub==0.22.2`: Model downloading
- `streamlit>=1.28.0`: Web interface

See `requirements.txt` for complete list.

## Project Structure

```
DiffyFace/
├── Data/                          # Data preparation pipeline
│   ├── FFHQ/                      # FFHQ dataset scripts
│   ├── EasyPortrait/              # EasyPortrait dataset scripts
│   ├── LAION_Face/                # LAION-Face dataset scripts
│   ├── batch_caption_generator.py # Distributed caption generation
│   ├── groq_caption_generator.py # Groq API integration
│   ├── data_loaders.py            # Dataset loaders
│   └── dataset_processors.py     # Dataset processors
├── Training/                      # Training scripts
│   ├── data/                      # Training data directory
│   │   ├── FFHQ/
│   │   ├── EasyPortrait/
│   │   └── LAION_Face/
│   ├── train.py                   # Main training script
│   ├── finetune_lora.py           # Core LoRA training
│   └── prepare_training_data.py   # Data preparation
├── Generation/                    # Inference and generation
│   ├── model.py                   # Core model class
│   ├── app.py                     # Streamlit web interface
│   ├── generate.py                # CLI generation script
│   ├── download.py                # Download checkpoints
│   ├── inference.ipynb            # Jupyter notebook
│   └── download_and_upload_model.py # Model upload utility
├── checkpoints/                   # Model checkpoints
│   └── lora30k/                   # LoRA weights
├── models/                        # Base model cache
│   └── stable-diffusion-2-1/      # Local model storage
├── Images/                        # Generated images
│   ├── Saved_images/              # Saved outputs
│   └── colab/                     # Colab outputs
└── requirements.txt               # Python dependencies
```

## Data Pipeline

### Dataset Overview

The model was trained on three face datasets:

| Dataset | Images | Source | Resolution |
|---------|--------|--------|------------|
| FFHQ | 70,000 | [NVlabs](https://github.com/NVlabs/ffhq-dataset) | 768×768 |
| EasyPortrait | 39,000 | [GitHub](https://github.com/hukenovs/easyportrait) | Variable |
| LAION-Face | 156,000 | [Hugging Face](https://huggingface.co/datasets/FacePerceiver/laion-face) | Filtered subset |
| **Total** | **265,000** | | |

### Dataset Preparation

#### 1. Download Datasets

**FFHQ:**
```bash
# Clone repository and download using official script (downloads ~89GB)
python Data/FFHQ/download_and_prepare.py --clone-and-download --download-dir ./downloads

# Prepare dataset for training (resizes to 768×768)
python Data/FFHQ/download_and_prepare.py \
    --prepare ./downloads/ffhq-dataset/images1024x1024 \
    --output ./data/processed/ffhq \
    --target-size 768
```

**Note:** The official download script will download JSON metadata and 1024×1024 images. This requires ~89GB of disk space and may take several hours depending on your internet connection.

**EasyPortrait:**
```bash
git clone https://github.com/hukenovs/easyportrait.git
python Data/EasyPortrait/download_and_prepare.py --extract ./easyportrait
python Data/EasyPortrait/download_and_prepare.py --prepare ./easyportrait_organized --output ./data/processed/easyportrait
```

**LAION-Face:**
```bash
python Data/LAION_Face/download_and_prepare.py --download --download-dir ./downloads --num-samples 156000
python Data/LAION_Face/download_and_prepare.py --prepare ./downloads/images --output ./data/processed/laion_face
```

#### 2. Generate Captions

Captions are generated using Groq API with the `meta-llama/llama-4-maverick-17b-128e-instruct` model. Each caption includes:
- Age, gender, ethnicity
- Facial expression and emotion
- Hair characteristics (color, length, style)
- Facial features (eyes, nose, mouth, etc.)
- Facial hair, accessories, skin tone

**Setup:**
```bash
export GROQ_API_KEY="your-api-key-here"
```

**Generate captions (distributed processing):**
```bash
# Process a batch of images (for distributed team processing)
python Data/batch_caption_generator.py \
    /path/to/images \
    --output metadata.jsonl \
    --start 0 \
    --end 10000 \
    --api-key your-api-key
```

**Data Format:**
Each dataset uses ImageFolder format with `metadata.jsonl`:
```
dataset_root/
    images/
        image1.jpg
        image2.jpg
    metadata.jsonl
```

`metadata.jsonl` format (one JSON per line):
```json
{"file_name": "image1.jpg", "text": "A detailed caption describing the person..."}
{"file_name": "image2.jpg", "text": "Another detailed caption..."}
```

### Distributed Processing

For large datasets, processing was distributed across team members:

**Example for 5 team members processing FFHQ (70k images):**
```bash
# Member 1: indices 0-14,000
python Data/batch_caption_generator.py dataset/images --output metadata_part1.jsonl --start 0 --end 14000

# Member 2: indices 14,000-28,000
python Data/batch_caption_generator.py dataset/images --output metadata_part2.jsonl --start 14000 --end 28000

# ... and so on
```

Then merge the results:
```python
from Data.dataset_processors import merge_datasets
merge_datasets(
    dataset_dirs=["part1", "part2", "part3", "part4", "part5"],
    output_dir="merged_dataset"
)
```

## Training

### Data Organization

Before training, organize all team member data:

```
Training/data/
├── FFHQ/
│   ├── images/              # All FFHQ images (merged from all members)
│   └── metadata.jsonl       # Single merged metadata file
├── EasyPortrait/
│   ├── images/
│   └── metadata.jsonl
└── LAION_Face/
    ├── images/
    └── metadata.jsonl
```

**Merge metadata files:**
```bash
cat member1.jsonl member2.jsonl member3.jsonl member4.jsonl member5.jsonl > Training/data/FFHQ/metadata.jsonl
```

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

### Ethical Considerations

Users are strongly encouraged to:

- **Consider limitations** when deploying in real-world applications
- **Perform validation** especially for diverse human subjects
- **Mitigate biases** through additional training or post-processing
- **Be transparent** about model limitations in applications
- **Respect privacy** and consent when generating images

### Responsible Use

This model should be used responsibly and ethically. Do not:
- Generate images without consent
- Create deepfakes or misleading content
- Use for discriminatory purposes
- Violate privacy or rights of individuals

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

### Getting Help

- Check existing documentation in subdirectories
- Review code comments in source files
- Verify all dependencies are installed
- Ensure CUDA is properly configured

## Citation

If you use this project in your research, please cite:

```bibtex
@software{diffyface2024,
  title={DiffyFace: Text-to-Face Generation with LoRA Fine-Tuned Stable Diffusion},
  author={Your Name},
  year={2024},
  url={https://github.com/rigvedrs/DiffyFace}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- **Stable Diffusion 2.1**: Base model by Stability AI
- **FFHQ Dataset**: Provided by NVIDIA Research
- **EasyPortrait Dataset**: Provided by the EasyPortrait team
- **LAION-Face Dataset**: Provided by LAION
- **Hugging Face**: For model hosting and diffusers library
- **Groq**: For caption generation API

## Contact

For questions, issues, or contributions, please open an issue on the GitHub repository.

---

**Note**: This project requires CUDA/GPU support. For users without GPU access, Google Colab provides free GPU runtime that works perfectly with this project.
