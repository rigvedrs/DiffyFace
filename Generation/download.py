"""
Download script for DiffyFace LoRA weights.

Downloads the pretrained LoRA weights from Hugging Face to the checkpoints directory.
"""

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_checkpoints():
    """Download the pretrained LoRA weights from Hugging Face."""
    # Set up paths
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "lora30k"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading DiffyFace LoRA weights from Hugging Face...")
    print(f"Target directory: {checkpoint_dir}")
    
    try:
        hf_hub_download(
            repo_id="rigvedrs/DiffyFace",
            filename="pytorch_lora_weights.safetensors",
            local_dir=str(checkpoint_dir),
            local_dir_use_symlinks=False
        )
        print(f"✓ Successfully downloaded LoRA weights to {checkpoint_dir}")
    except Exception as e:
        print(f"✗ Error downloading weights: {e}")
        raise


if __name__ == "__main__":
    download_checkpoints()