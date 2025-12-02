"""
Upload LoRA weights to Hugging Face Hub

This script uploads the fine-tuned LoRA weights to your Hugging Face Hub repository.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, upload_file, login
from huggingface_hub.utils import HfHubHTTPError

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def upload_lora_weights(
    lora_path: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload DiffyFace LoRA weights"
):
    """
    Upload LoRA weights to Hugging Face Hub.
    
    Args:
        lora_path: Path to the LoRA weights file or directory
        repo_id: Hugging Face repository ID (e.g., 'username/model-name')
        private: Whether the repository should be private
        commit_message: Commit message for the upload
    """
    print(f"\nUploading LoRA weights to Hugging Face Hub...")
    print(f"Repository: {repo_id}")
    print(f"Local path: {lora_path}")
    
    # Check if user is logged in
    try:
        api = HfApi()
        user = api.whoami()
        print(f"✓ Logged in as: {user['name']}")
    except Exception:
        print("⚠ Not logged in to Hugging Face Hub")
        print("Please log in using: huggingface-cli login")
        response = input("Would you like to log in now? (y/n): ")
        if response.lower() == 'y':
            login()
        else:
            raise Exception("Please log in to Hugging Face Hub first")
    
    lora_file = Path(lora_path)
    if not lora_file.exists():
        raise ValueError(f"LoRA file does not exist: {lora_path}")
    
    # If it's a directory, look for the weights file
    if lora_file.is_dir():
        weight_file = lora_file / "pytorch_lora_weights.safetensors"
        if not weight_file.exists():
            raise ValueError(f"LoRA weights file not found in directory: {weight_file}")
        lora_file = weight_file
    
    try:
        # Create repository if it doesn't exist
        api = HfApi()
        try:
            repo_info = api.repo_info(repo_id, repo_type="model")
            print(f"✓ Repository {repo_id} exists")
            # Check if we need to delete existing files (if it has the base model)
            files = api.list_repo_files(repo_id, repo_type="model")
            if any(f.startswith("unet/") or f.startswith("text_encoder/") or f.startswith("vae/") for f in files):
                print("⚠ Repository contains base model files. These will be replaced with LoRA weights.")
                response = input("Continue? (y/n): ")
                if response.lower() != 'y':
                    print("Upload cancelled.")
                    return
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                print(f"Creating new repository: {repo_id}")
                api.create_repo(
                    repo_id=repo_id,
                    repo_type="model",
                    private=private,
                    exist_ok=False
                )
                print(f"✓ Repository created successfully")
            else:
                raise
        
        # Upload the LoRA weights file
        print(f"\nUploading LoRA weights file...")
        upload_file(
            path_or_fileobj=str(lora_file),
            path_in_repo="pytorch_lora_weights.safetensors",
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message
        )
        
        print(f"\n✓ LoRA weights uploaded successfully!")
        print(f"  View at: https://huggingface.co/{repo_id}")
        print(f"  File: pytorch_lora_weights.safetensors")
    except Exception as e:
        print(f"✗ Error uploading LoRA weights: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Upload LoRA weights to Hugging Face Hub"
    )
    
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Path to LoRA weights file or directory (default: ./checkpoints/lora30k/pytorch_lora_weights.safetensors)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="rigvedrs/DiffyFace",
        help="Hugging Face repository ID (default: rigvedrs/DiffyFace)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload DiffyFace LoRA weights",
        help="Commit message for the upload"
    )
    
    args = parser.parse_args()
    
    # Determine LoRA path
    if args.lora_path:
        lora_path = Path(args.lora_path)
    else:
        lora_path = PROJECT_ROOT / "checkpoints" / "lora30k" / "pytorch_lora_weights.safetensors"
    
    if not lora_path.exists():
        raise ValueError(
            f"LoRA weights not found at: {lora_path}\n"
            "Please specify --lora-path or ensure the weights are at ./checkpoints/lora30k/pytorch_lora_weights.safetensors"
        )
    
    print("="*80)
    print("Uploading LoRA Weights to Hugging Face Hub")
    print("="*80)
    
    upload_lora_weights(
        lora_path=str(lora_path),
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message
    )


if __name__ == "__main__":
    main()

