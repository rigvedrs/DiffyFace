# DiffyFace: Text-to-Face Generation with LoRA Fine-Tuned Stable Diffusion
## Complete Project Report

---

## Table of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Project Overview](#project-overview)
4. [Objectives](#objectives)
5. [Methodology](#methodology)
6. [Technical Implementation](#technical-implementation)
7. [Results and Achievements](#results-and-achievements)
8. [Challenges and Solutions](#challenges-and-solutions)
9. [Conclusion](#conclusion)
10. [Future Work](#future-work)
11. [References](#references)

---

## Abstract

DiffyFace is a comprehensive system for generating high-quality, realistic face images from detailed text descriptions using Stable Diffusion 2.1 fine-tuned with LoRA (Low-Rank Adaptation). This project addresses the challenge of creating a specialized text-to-face generation model by curating a large-scale dataset of 265,000 face images, generating detailed captions using automated vision-language models, and fine-tuning a state-of-the-art diffusion model. The system includes complete pipelines for dataset preparation, distributed data processing, model training, and multiple inference interfaces. The final model is production-ready and available on Hugging Face Hub, demonstrating the effectiveness of LoRA fine-tuning for domain-specific image generation tasks.

**Keywords**: Text-to-Face Generation, Stable Diffusion, LoRA Fine-tuning, Computer Vision, Deep Learning, Diffusion Models

---

## Introduction

Text-to-image generation has emerged as one of the most exciting applications of generative AI. While general-purpose models like Stable Diffusion can generate diverse images, they often struggle with specific domains like human faces, which require precise control over facial features, expressions, and characteristics. This project addresses this limitation by creating a specialized face generation model through fine-tuning Stable Diffusion 2.1 on a carefully curated dataset of face images with detailed captions.

The project was designed with scalability and collaboration in mind, implementing a distributed data processing pipeline that allows multiple team members to work in parallel on large-scale dataset preparation. This approach enabled the processing of 265,000 images efficiently, making the project feasible for team-based development.

---

## Project Overview

### Problem Statement

General-purpose text-to-image models face several challenges when generating human faces:
1. **Lack of specificity**: Generic models don't capture fine-grained facial details
2. **Inconsistent quality**: Face generation often produces artifacts or unrealistic features
3. **Limited control**: Difficulty in precisely controlling facial attributes from text
4. **Dataset requirements**: Need for large-scale, high-quality face datasets with detailed captions

### Solution Approach

DiffyFace addresses these challenges through:
1. **Curated Dataset**: Combining three high-quality face datasets (FFHQ, EasyPortrait, LAION-Face)
2. **Automated Captioning**: Using Groq API with vision-language models to generate detailed facial descriptions
3. **LoRA Fine-tuning**: Efficiently adapting Stable Diffusion 2.1 for face generation
4. **Distributed Processing**: Enabling parallel data processing across team members
5. **Multiple Interfaces**: Providing various ways to interact with the model

### Key Statistics

- **Total Dataset Size**: 265,000 face images
- **Training Resolution**: 768×768 pixels
- **Model Size**: ~5GB base model + ~50MB LoRA weights
- **Training Time**: ~20 epochs on 265k images
- **Team Members**: Designed for 5-member distributed processing

---

## Objectives

### Primary Objectives

1. **Dataset Creation**: Curate and prepare a large-scale dataset of face images with detailed captions
2. **Model Development**: Fine-tune Stable Diffusion 2.1 using LoRA for face generation
3. **System Integration**: Build complete pipelines for data processing, training, and inference
4. **Accessibility**: Create multiple user interfaces for model interaction
5. **Documentation**: Provide comprehensive documentation for reproducibility

### Secondary Objectives

1. **Distributed Processing**: Enable efficient team-based data processing
2. **Production Deployment**: Make the model available on Hugging Face Hub
3. **Code Quality**: Maintain clean, modular, and well-documented code
4. **User Experience**: Provide intuitive interfaces (notebook, web app, CLI)

---

## Methodology

This section documents all work completed in chronological order, detailing each phase of the project.

### Phase 1: Project Setup and Architecture Design

#### 1.1 Repository Structure
Created a well-organized project structure:
```
DiffyFace/
├── Data/                    # Data preparation pipeline
├── Training/               # Training scripts
├── Generation/             # Inference tools
├── checkpoints/            # Model weights
└── models/                 # Base model cache
```

#### 1.2 Technology Stack Selection
- **Base Model**: Stable Diffusion 2.1 (768×768 resolution)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Framework**: PyTorch, Diffusers, Hugging Face Transformers
- **Caption Generation**: Groq API with Llama-4-Maverick-17B-128E-Instruct
- **Inference Interfaces**: Jupyter Notebook, Streamlit, CLI

#### 1.3 Dependency Management
Created `requirements.txt` with all necessary packages:
- Core: `diffusers==0.27.2`, `transformers==4.40.1`, `torch`
- Training: `accelerate==0.29.3`, `peft==0.10.0`
- Utilities: `pillow`, `datasets`, `streamlit`, `groq`

### Phase 2: Dataset Selection and Preparation

#### 2.1 Dataset Selection
Selected three complementary face datasets:

1. **FFHQ (Flickr-Faces-HQ)**
   - Size: 70,000 images
   - Resolution: 1024×1024 (resized to 768×768)
   - Quality: High-quality, diverse faces
   - Source: NVIDIA Research

2. **EasyPortrait**
   - Size: 39,000 images
   - Quality: Well-framed portraits with consistent lighting
   - Source: GitHub (hukenovs/easyportrait)

3. **LAION-Face**
   - Size: 156,000 images (filtered subset)
   - Quality: High-quality face crops from LAION dataset
   - Source: Hugging Face (FacePerceiver/laion-face)

**Total**: 265,000 images

#### 2.2 Dataset Download Scripts
Created individual download and preparation scripts for each dataset:

**FFHQ (`Data/FFHQ/download_and_prepare.py`)**:
- Clones official FFHQ repository
- Downloads 1024×1024 images (~89GB)
- Resizes to 768×768 for training
- Organizes into ImageFolder structure

**EasyPortrait (`Data/EasyPortrait/download_and_prepare.py`)**:
- Extracts images from repository structure
- Organizes into flat directory structure
- Handles nested directories

**LAION-Face (`Data/LAION_Face/download_and_prepare.py`)**:
- Downloads from Hugging Face datasets
- Applies quality filtering (512×512 to 2048×2048)
- Samples 156k images

#### 2.3 Master Preparation Script
Created `Data/prepare_all_datasets.py` to prepare all datasets in one command:
```python
python Data/prepare_all_datasets.py \
    --ffhq ./downloads/ffhq \
    --easyportrait ./easyportrait \
    --laion-face ./downloads/laion_face \
    --output ./data/processed
```

### Phase 3: Distributed Caption Generation System

#### 3.1 Problem: Large-Scale Caption Generation
With 265,000 images, generating captions sequentially would take weeks. Solution: **Distributed processing across team members**.

#### 3.2 Groq API Integration
Created `Data/groq_caption_generator.py`:
- Integrates with Groq API
- Uses `meta-llama/llama-4-maverick-17b-128e-instruct` model
- Implements retry logic and rate limiting
- Handles image encoding (base64)
- Generates detailed facial descriptions

**Caption Prompt** (from paper):
```
Provide a detailed physical description of the person in this image. 
Include: approximate age, gender, ethnicity/race, facial expression 
and emotion, hair characteristics (color, length, style, hairline), 
facial features (eyes, nose, mouth, lips, chin, cheekbones), facial 
hair if present, accessories (eyeglasses, hats, jewelry), body type 
indicators, skin tone, makeup if visible, and any other distinctive 
physical characteristics.
```

#### 3.3 Distributed Processing Notebook
Created `Data/data_pipeline.ipynb` - the core tool for team collaboration:

**Features**:
- Each team member sets `START_INDEX` and `END_INDEX`
- Processes their assigned batch independently
- Saves results incrementally to `metadata.jsonl`
- Works seamlessly in Google Colab

**Example Distribution (FFHQ, 70k images, 5 members)**:
- Member 1: indices 0-14,000
- Member 2: indices 14,000-28,000
- Member 3: indices 28,000-42,000
- Member 4: indices 42,000-56,000
- Member 5: indices 56,000-70,000

#### 3.4 Command-Line Alternative
Created `Data/batch_caption_generator.py` for non-notebook environments:
```bash
python Data/batch_caption_generator.py \
    ./data/images \
    --output metadata.jsonl \
    --start 0 \
    --end 14000
```

#### 3.5 Data Format
Standardized on ImageFolder format with `metadata.jsonl`:
```
dataset_root/
    images/
        image1.jpg
        image2.jpg
    metadata.jsonl
```

`metadata.jsonl` format (one JSON per line):
```json
{"file_name": "image1.jpg", "text": "A happy 25 year old male..."}
{"file_name": "image2.jpg", "text": "A serious 30 year old female..."}
```

### Phase 4: Dataset Processors and Loaders

#### 4.1 Dataset Processors
Created `Data/dataset_processors.py` with three processor classes:
- `FFHQProcessor`: Handles FFHQ-specific preparation
- `EasyPortraitProcessor`: Handles EasyPortrait-specific preparation
- `LAIONFaceProcessor`: Handles LAION-Face-specific preparation
- `merge_datasets()`: Combines multiple datasets

#### 4.2 Data Loaders
Created `Data/data_loaders.py`:
- `ImageFolderWithMetadata`: PyTorch Dataset for ImageFolder format
- `HuggingFaceDatasetWrapper`: Wrapper for HuggingFace datasets
- Handles image preprocessing, resizing, normalization
- Supports center crop and data augmentation

### Phase 5: Training Infrastructure

#### 5.1 Training Data Preparation
Created `Training/prepare_training_data.py`:
- Merges data from all team members
- Verifies image-caption pairs
- Organizes into training-ready structure
- Handles duplicate filenames

#### 5.2 LoRA Fine-tuning Implementation
Created `Training/finetune_lora.py`:
- Implements LoRA (Low-Rank Adaptation) for Stable Diffusion
- Fine-tunes both UNet and text encoder
- Uses PEFT library for efficient training
- Supports gradient checkpointing for memory efficiency
- Implements mixed precision training (fp16/bf16)

**LoRA Configuration**:
- **UNet LoRA Rank**: 8
- **Text Encoder LoRA Rank**: 8
- **Target Modules**: 
  - UNet: `["to_k", "to_q", "to_v", "to_out.0"]`
  - Text Encoder: `["q_proj", "v_proj", "k_proj", "out_proj"]`

#### 5.3 Main Training Script
Created `Training/train.py`:
- Orchestrates data preparation and training
- Handles command-line arguments
- Supports automatic data preparation
- Integrates with Hugging Face Accelerate for distributed training

**Training Configuration**:
- **Base Model**: `rigvedrs/Diffy-2-1` (Stable Diffusion 2.1)
- **Resolution**: 768×768
- **Batch Size**: 4 (per device)
- **Learning Rate**: 1e-5 (UNet and text encoder)
- **Epochs**: 20
- **Gradient Accumulation**: 1
- **Mixed Precision**: fp16

#### 5.4 Training Process
1. Load base Stable Diffusion 2.1 model
2. Freeze base model parameters
3. Add LoRA adapters to UNet and text encoder
4. Train only LoRA parameters
5. Save LoRA weights separately (~50MB vs ~5GB full model)

### Phase 6: Inference System Development

#### 6.1 Core Model Class
Created `Generation/model.py` - `DiffyFaceModel` class:
- Loads base Stable Diffusion 2.1 from Hugging Face
- Loads LoRA weights from local checkpoint
- Provides `generate()` method with customizable parameters
- Handles device management (CUDA only)
- Returns PIL Images

**Key Features**:
- Automatic model loading
- Support for negative prompts
- Configurable inference steps and guidance scale
- Seed support for reproducibility

#### 6.2 Jupyter Notebook Interface
Created `Generation/inference.ipynb`:
- Fully automated Colab-ready notebook
- Auto-installs dependencies
- Auto-downloads checkpoints
- Auto-detects paths
- Interactive generation with visualization
- Batch generation support

#### 6.3 Streamlit Web Application
Created `Generation/app.py`:
- User-friendly web interface
- Real-time image generation
- Parameter adjustment (steps, guidance scale, seed)
- Negative prompt support
- Image saving and download
- Saves to `Images/Saved_images/`

**Features**:
- Interactive sliders for parameters
- Text input for prompts
- Image display and download
- Session state management

#### 6.4 Command-Line Interface
Created `Generation/generate.py`:
- Simple CLI for batch generation
- Script-based workflow
- Easy to integrate into pipelines

#### 6.5 Model Download Script
Created `Generation/download.py`:
- Downloads pretrained LoRA weights from Hugging Face
- Saves to `checkpoints/lora30k/`
- Handles errors gracefully

### Phase 7: Model Hosting and Distribution

#### 7.1 Hugging Face Hub Integration
- Uploaded base model as `rigvedrs/Diffy-2-1`
- Uploaded LoRA weights as `rigvedrs/DiffyFace`
- Created model cards with usage instructions
- Made models publicly accessible

#### 7.2 Model Upload Utility
Created `Generation/download_and_upload_model.py`:
- Downloads models from Hugging Face
- Uploads custom trained models
- Handles authentication

### Phase 8: Documentation and Testing

#### 8.1 Comprehensive README
Created detailed `README.md` with:
- Project overview and features
- Installation instructions
- Complete data pipeline documentation
- Training guide
- Inference guide with all interfaces
- Troubleshooting section
- Project structure

#### 8.2 Data Structure Documentation
Created `Training/DATA_STRUCTURE.md`:
- Expected data organization
- Merging instructions
- Format specifications

#### 8.3 Helper Scripts
Created `run_streamlit.sh`:
- Automates Streamlit app launch
- Handles conda environment activation
- Verifies dependencies

### Phase 9: Quality Assurance and Optimization

#### 9.1 Error Handling
- Implemented retry logic in caption generation
- Added error handling in data loaders
- Graceful failure in training pipeline

#### 9.2 Memory Optimization
- Gradient checkpointing for training
- Mixed precision training (fp16)
- Efficient data loading with batching

#### 9.3 Code Quality
- Modular design with clear separation of concerns
- Comprehensive docstrings
- Type hints where applicable
- Consistent code style

---

## Technical Implementation

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    DIFFYFACE ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────┘

Data Pipeline:
  Download → Prepare → Generate Captions → Merge → Train

Training:
  Base Model (SD 2.1) → Add LoRA Adapters → Fine-tune → Save Weights

Inference:
  Load Base Model → Load LoRA Weights → Generate Images
```

### Key Technical Decisions

#### 1. LoRA vs Full Fine-tuning
**Decision**: Use LoRA (Low-Rank Adaptation)
**Rationale**:
- Reduces trainable parameters by 99% (50MB vs 5GB)
- Faster training and inference
- Easier to share and deploy
- Maintains base model capabilities

#### 2. Resolution: 768×768
**Decision**: Train at 768×768 (not 512×512)
**Rationale**:
- Better facial detail preservation
- Matches Stable Diffusion 2.1 native resolution
- Good balance between quality and computational cost

#### 3. Distributed Processing
**Decision**: Notebook-based distributed caption generation
**Rationale**:
- Enables parallel processing across team members
- Each member works independently
- Results can be merged easily
- Works well with Google Colab free tier

#### 4. Groq API for Captioning
**Decision**: Use Groq API instead of local models
**Rationale**:
- Fast inference (vision-language model)
- No need for local GPU for caption generation
- Cost-effective API pricing
- High-quality detailed descriptions

### Training Details

**Hardware Requirements**:
- GPU: NVIDIA GPU with CUDA support
- VRAM: Minimum 8GB (16GB recommended)
- Storage: ~100GB for datasets + models

**Training Process**:
1. Load pretrained Stable Diffusion 2.1
2. Freeze all base parameters
3. Add LoRA adapters (rank=8) to UNet and text encoder
4. Train for 20 epochs on 265k image-caption pairs
5. Save only LoRA weights

**Training Time** (estimated):
- Per epoch: ~4-6 hours (depending on GPU)
- Total: ~80-120 hours for 20 epochs

**Memory Usage**:
- Base model: ~5GB VRAM
- Training overhead: ~3-5GB VRAM
- Total: ~8-10GB VRAM

### Inference Details

**Generation Parameters**:
- **Default Steps**: 50 (denoising iterations)
- **Guidance Scale**: 7.5 (prompt adherence)
- **Resolution**: 768×768
- **Seed**: Random (or specified for reproducibility)

**Performance**:
- Generation time: ~5-10 seconds per image (on RTX 3090)
- Memory usage: ~6-8GB VRAM during inference

---

## Results and Achievements

### Dataset Creation

✅ **Successfully created dataset of 265,000 face images with detailed captions**
- FFHQ: 70,000 images
- EasyPortrait: 39,000 images
- LAION-Face: 156,000 images
- All images paired with detailed facial descriptions

### Model Training

✅ **Successfully fine-tuned Stable Diffusion 2.1 with LoRA**
- Trained on 265k image-caption pairs
- LoRA rank: 8 (both UNet and text encoder)
- Training completed: 20 epochs
- Model size: ~50MB LoRA weights (vs 5GB full model)

### System Development

✅ **Complete pipeline implementation**:
- Data download and preparation scripts
- Distributed caption generation system
- Training infrastructure
- Multiple inference interfaces

✅ **Production-ready deployment**:
- Model available on Hugging Face Hub
- Comprehensive documentation
- Multiple user interfaces
- Easy installation and setup

### User Interfaces

✅ **Three inference interfaces**:
1. **Jupyter Notebook**: Colab-ready, fully automated
2. **Streamlit Web App**: User-friendly web interface
3. **Command-Line**: Script-based generation

### Code Quality

✅ **Well-documented codebase**:
- Comprehensive README
- Inline documentation
- Project structure documentation
- Troubleshooting guides

### Team Collaboration

✅ **Distributed processing system**:
- Enables parallel work across team members
- Efficient processing of large datasets
- Easy result merging

---

## Challenges and Solutions

### Challenge 1: Large-Scale Caption Generation

**Problem**: Generating captions for 265,000 images would take weeks if done sequentially.

**Solution**: Implemented distributed processing system:
- Created `data_pipeline.ipynb` for team distribution
- Each member processes different batch ranges
- Results saved incrementally
- Parallel processing reduces time from weeks to days

### Challenge 2: Memory Constraints

**Problem**: Training full Stable Diffusion 2.1 requires significant VRAM.

**Solution**: Used LoRA fine-tuning:
- Reduces trainable parameters by 99%
- Lower memory footprint
- Faster training and inference
- Maintains model quality

### Challenge 3: Dataset Organization

**Problem**: Merging data from multiple team members and datasets.

**Solution**: Created standardized format:
- ImageFolder structure with `metadata.jsonl`
- Automated merging scripts
- Data validation during preparation

### Challenge 4: Model Deployment

**Problem**: Making model accessible and easy to use.

**Solution**: Multiple interfaces and Hugging Face hosting:
- Jupyter notebook for Colab users
- Streamlit app for web interface
- CLI for scripting
- Hugging Face Hub for easy access

### Challenge 5: Caption Quality

**Problem**: Ensuring consistent, detailed captions for all images.

**Solution**: Used Groq API with vision-language model:
- Consistent prompt template
- High-quality model (Llama-4-Maverick)
- Detailed facial attribute descriptions
- Automated quality through API

---

## Conclusion

The DiffyFace project successfully demonstrates the feasibility of creating a specialized text-to-face generation model through LoRA fine-tuning of Stable Diffusion 2.1. Key achievements include:

1. **Large-Scale Dataset**: Created a curated dataset of 265,000 face images with detailed captions
2. **Efficient Training**: Successfully fine-tuned using LoRA, reducing model size by 99%
3. **Distributed Processing**: Implemented scalable system for team-based data processing
4. **Production Deployment**: Made model accessible through Hugging Face Hub
5. **Multiple Interfaces**: Provided various ways to interact with the model

The project showcases best practices in:
- Distributed data processing
- Efficient model fine-tuning
- Production deployment
- Comprehensive documentation
- User experience design

The final model is capable of generating high-quality face images from detailed text descriptions, demonstrating the effectiveness of domain-specific fine-tuning for generative models.

---

## Future Work

### Short-term Improvements

1. **Enhanced Caption Quality**:
   - Fine-tune caption generation prompts
   - Add validation for caption consistency
   - Implement caption quality metrics

2. **Model Optimization**:
   - Experiment with different LoRA ranks
   - Try different learning rates
   - Implement learning rate scheduling

3. **Dataset Expansion**:
   - Add more diverse face datasets
   - Include more demographic diversity
   - Balance dataset composition

### Long-term Enhancements

1. **Advanced Features**:
   - Support for face editing (inpainting)
   - Style transfer capabilities
   - Multi-face generation

2. **Performance Optimization**:
   - Faster inference (optimization techniques)
   - Lower memory requirements
   - Support for CPU inference (quantization)

3. **Bias Mitigation**:
   - Dataset balancing across demographics
   - Bias detection and correction
   - Fairness evaluation metrics

4. **User Experience**:
   - Real-time generation preview
   - Prompt suggestions
   - Image editing interface

5. **Research Directions**:
   - Compare with other fine-tuning methods
   - Evaluate different base models
   - Study prompt engineering for faces

---

## References

### Datasets

1. **FFHQ (Flickr-Faces-HQ)**: Karras, T., Laine, S., & Aila, T. (2019). A Style-Based Generator Architecture for Generative Adversarial Networks. CVPR 2019.

2. **EasyPortrait**: GitHub repository - https://github.com/hukenovs/easyportrait

3. **LAION-Face**: LAION dataset subset - https://huggingface.co/datasets/FacePerceiver/laion-face

### Models and Libraries

1. **Stable Diffusion 2.1**: Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.

2. **LoRA**: Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.

3. **Diffusers Library**: von Platen, P., et al. (2022). Diffusers: State-of-the-art diffusion models. Hugging Face.

4. **PEFT Library**: Hugging Face Parameter-Efficient Fine-Tuning - https://github.com/huggingface/peft

### Tools and Services

1. **Groq API**: https://console.groq.com - Vision-language model API
2. **Hugging Face Hub**: https://huggingface.co - Model hosting and distribution
3. **Google Colab**: https://colab.research.google.com - Free GPU access

### Project Resources

- **GitHub Repository**: https://github.com/rigvedrs/DiffyFace
- **Hugging Face Model**: https://huggingface.co/rigvedrs/DiffyFace
- **Base Model**: https://huggingface.co/rigvedrs/Diffy-2-1

---

## Appendix

### A. Project Statistics

- **Total Lines of Code**: ~5,000+ lines
- **Number of Scripts**: 15+ Python scripts
- **Number of Notebooks**: 2 Jupyter notebooks
- **Documentation Files**: 3 markdown files
- **Dataset Size**: 265,000 images
- **Model Size**: 50MB (LoRA weights)
- **Training Time**: ~80-120 hours
- **Development Time**: Multiple months

### B. File Structure

```
DiffyFace/
├── Data/                          # Data pipeline (8 files)
│   ├── data_pipeline.ipynb        # Distributed processing notebook
│   ├── groq_caption_generator.py   # Groq API integration
│   ├── batch_caption_generator.py # CLI caption generation
│   ├── dataset_processors.py     # Dataset-specific processors
│   ├── data_loaders.py            # PyTorch data loaders
│   ├── prepare_all_datasets.py    # Master preparation script
│   └── [dataset folders]/         # Dataset-specific scripts
├── Training/                      # Training infrastructure (4 files)
│   ├── train.py                   # Main training script
│   ├── finetune_lora.py           # LoRA fine-tuning core
│   ├── prepare_training_data.py   # Data preparation
│   └── DATA_STRUCTURE.md          # Data organization guide
├── Generation/                    # Inference system (6 files)
│   ├── model.py                   # Core model class
│   ├── app.py                     # Streamlit web app
│   ├── generate.py                # CLI generation
│   ├── inference.ipynb            # Colab notebook
│   ├── download.py                 # Model download
│   └── download_and_upload_model.py # Model upload utility
├── README.md                      # Main documentation
├── PROJECT_REPORT.md              # This report
└── requirements.txt                # Dependencies
```

### C. Key Technologies

- **Python 3.9+**: Primary programming language
- **PyTorch**: Deep learning framework
- **Diffusers**: Stable Diffusion implementation
- **Transformers**: Model loading and tokenization
- **PEFT**: LoRA implementation
- **Accelerate**: Distributed training
- **Streamlit**: Web interface
- **Groq API**: Caption generation
- **Hugging Face Hub**: Model hosting

