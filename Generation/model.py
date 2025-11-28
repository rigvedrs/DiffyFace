"""
DiffyFace Model Module

This module provides the Model class for generating face images using Stable Diffusion
with LoRA fine-tuning. CUDA/GPU only.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DiffyFaceModel:
    """
    A model class for generating face images from text prompts using Stable Diffusion 2.1
    with LoRA fine-tuning. CUDA/GPU only.
    
    Attributes:
        checkpoint: Path to the LoRA checkpoint directory
        weight_name: Name of the LoRA weights file
        pipe: The Stable Diffusion pipeline
    """
    
    def __init__(
        self,
        checkpoint: str = "checkpoints/lora30k",
        weight_name: str = "pytorch_lora_weights.safetensors",
        device: str = "cuda"
    ):
        """
        Initialize the DiffyFace model.
        
        Args:
            checkpoint: Path to LoRA checkpoint directory
            weight_name: Name of the LoRA weights file
            device: Device to use (must be 'cuda')
        """
        if device != "cuda":
            raise ValueError("DiffyFace model only supports CUDA. Please use a GPU-enabled machine.")
        
        # Set up paths
        self.checkpoint = checkpoint
        if not Path(checkpoint).is_absolute():
            self.checkpoint = str(PROJECT_ROOT / checkpoint)
        
        # Check if checkpoint file exists
        checkpoint_file = Path(self.checkpoint) / weight_name
        if not checkpoint_file.exists():
            raise FileNotFoundError(
                f"LoRA checkpoint file not found at {checkpoint_file}\n"
                f"Please download the checkpoint first:\n"
                f"  python Generation/download.py\n"
                f"Or download manually from: https://huggingface.co/rigvedrs/DiffyFace"
            )
        
        # Load LoRA state dict
        print(f"Loading LoRA weights from {self.checkpoint}...")
        state_dict, network_alphas = StableDiffusionPipeline.lora_state_dict(
            self.checkpoint,
            weight_name=weight_name
        )
        
        # Load base Stable Diffusion 2.1 model
        print("Loading Stable Diffusion 2.1 model from rigvedrs/Diffy-2-1...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "rigvedrs/Diffy-2-1",
            torch_dtype=torch.float16
        ).to(device)
        
        # Load LoRA weights into model
        self.pipe.load_lora_into_unet(
            state_dict, network_alphas, self.pipe.unet, adapter_name='diffyface_lora'
        )
        self.pipe.load_lora_into_text_encoder(
            state_dict, network_alphas, self.pipe.text_encoder, adapter_name='diffyface_lora'
        )
        self.pipe.set_adapters(["diffyface_lora"], adapter_weights=[1.0])
        
        print("✓ Model loaded successfully!")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> Image.Image:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text prompt describing the face to generate
            negative_prompt: Negative prompt to guide generation
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility
            save_path: Optional path to save the image
            
        Returns:
            Generated PIL Image
        """
        lora_scale = 1.0
        
        # Set up generator with seed if provided
        generator = torch.manual_seed(seed) if seed is not None else None
        
        # Generate image
        image = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            cross_attention_kwargs={"scale": lora_scale},
            generator=generator
        ).images[0]
        
        # Save if path provided
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            image.save(save_path)
            print(f"✓ Image saved to: {save_path}")
        
        return image
    
    def generate_and_save(
        self,
        prompt: str,
        save_dir: Optional[str] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate an image and save it with a filename based on the prompt.
        
        Args:
            prompt: Text prompt describing the face to generate
            save_dir: Directory to save the image (defaults to checkpoint directory)
            negative_prompt: Negative prompt to guide generation
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility
            
        Returns:
            Generated PIL Image
        """
        # Create filename from prompt
        filename = self._prompt_to_filename(prompt)
        
        # Use checkpoint directory if no save_dir specified
        if save_dir is None:
            save_dir = self.checkpoint
        
        save_path = os.path.join(save_dir, filename)
        
        # Generate and save
        return self.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            seed=seed,
            save_path=save_path
        )
    
    @staticmethod
    def _prompt_to_filename(prompt: str) -> str:
        """
        Convert a prompt to a valid filename.
        
        Args:
            prompt: Text prompt
            
        Returns:
            Filename string with underscores
        """
        # Clean the prompt for filename
        filename = prompt.replace(".", " ").replace(",", " ").replace("!", " ")
        filename = "_".join(filename.split())
        filename = filename.replace("__", "_")  # Remove double underscores
        filename = filename.strip("_")  # Remove leading/trailing underscores
        
        # Add extension
        if not filename.endswith(".png"):
            filename += ".png"
        
        return filename

