"""
Command-line script for generating face images using DiffyFace model (CUDA version).

Usage:
    python Generation/generate.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Generation.model import DiffyFaceModel


def main():
    """Main function for command-line generation."""
    # Initialize model for CUDA
    print("Initializing DiffyFace model...")
    model = DiffyFaceModel(device="cuda")
    
    # Example prompt
    prompt = 'A happy 55 year old male with blond hair and a goatee smiles with visible teeth.'
    negative_prompt = ''
    
    # Generate and save image
    print(f"\nGenerating image with prompt: '{prompt}'")
    image = model.generate_and_save(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        seed=42
    )


if __name__ == "__main__":
    main()