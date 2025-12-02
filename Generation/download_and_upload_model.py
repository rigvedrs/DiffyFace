"""
Download and Upload Hugging Face Model Script

This script downloads the Stable Diffusion 2.1 model from Hugging Face
and uploads it to your Hugging Face Hub repository.

Usage:
    python Generation/download_and_upload_model.py --repo-id your-username/stable-diffusion-2-1
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download, upload_folder, HfApi, login
from huggingface_hub.utils import HfHubHTTPError

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_model(model_id: str = "stabilityai/stable-diffusion-2-1", cache_dir: str = None):
    """
    Download the Stable Diffusion model from Hugging Face.
    
    Args:
        model_id: Hugging Face model ID (default: stabilityai/stable-diffusion-2-1)
        cache_dir: Directory to cache the downloaded model (default: ~/.cache/huggingface)
    
    Returns:
        Path to the downloaded model directory
    """
    print(f"Downloading model: {model_id}")
    print("This may take a while depending on your internet connection...")
    
    try:
        # Download the entire model repository
        model_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_dir=None,  # Use cache, not local_dir for full download
            local_dir_use_symlinks=False
        )
        
        print(f"✓ Model downloaded successfully!")
        print(f"  Model cached at: {model_path}")
        return model_path
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        raise


def download_model_to_local(model_id: str = "stabilityai/stable-diffusion-2-1", local_dir: str = None):
    """
    Download the Stable Diffusion model to a local directory.
    
    Args:
        model_id: Hugging Face model ID (default: stabilityai/stable-diffusion-2-1)
        local_dir: Local directory to save the model (default: ./models/stable-diffusion-2-1)
    
    Returns:
        Path to the downloaded model directory
    """
    if local_dir is None:
        local_dir = PROJECT_ROOT / "models" / "stable-diffusion-2-1"
    else:
        local_dir = Path(local_dir)
    
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading model: {model_id}")
    print(f"Target directory: {local_dir}")
    print("This may take a while depending on your internet connection...")
    print("(The model is ~5GB, so please be patient)")
    
    try:
        # Download the entire model repository to local directory
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False
        )
        
        print(f"✓ Model downloaded successfully!")
        print(f"  Model saved to: {local_dir}")
        return local_dir
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        raise


def upload_model_to_hub(
    local_model_path: str,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload Stable Diffusion 2.1 model"
):
    """
    Upload a model to Hugging Face Hub.
    
    Args:
        local_model_path: Path to the local model directory
        repo_id: Hugging Face repository ID (e.g., 'username/model-name')
        private: Whether the repository should be private
        commit_message: Commit message for the upload
    """
    print(f"\nUploading model to Hugging Face Hub...")
    print(f"Repository: {repo_id}")
    print(f"Local path: {local_model_path}")
    
    # Check if user is logged in
    try:
        api = HfApi()
        user = api.whoami()
        print(f"✓ Logged in as: {user['name']}")
    except Exception:
        print("⚠ Not logged in to Hugging Face Hub")
        print("Please log in using: huggingface-cli login")
        print("Or run: python -c 'from huggingface_hub import login; login()'")
        response = input("Would you like to log in now? (y/n): ")
        if response.lower() == 'y':
            login()
        else:
            raise Exception("Please log in to Hugging Face Hub first")
    
    local_path = Path(local_model_path)
    if not local_path.exists():
        raise ValueError(f"Model path does not exist: {local_model_path}")
    
    try:
        # Create repository if it doesn't exist
        api = HfApi()
        try:
            api.repo_info(repo_id, repo_type="model")
            print(f"✓ Repository {repo_id} already exists")
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
        
        # Upload the model
        print(f"\nUploading files... (this may take a while)")
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            ignore_patterns=[".git*", "__pycache__*", "*.pyc"]
        )
        
        print(f"\n✓ Model uploaded successfully!")
        print(f"  View at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"✗ Error uploading model: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download Stable Diffusion model and upload to Hugging Face Hub"
    )
    
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Hugging Face model ID to download (default: stabilityai/stable-diffusion-2-1)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Your Hugging Face repository ID (e.g., 'your-username/stable-diffusion-2-1')"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Local directory to save the downloaded model (default: ./models/stable-diffusion-2-1)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for downloads (default: ~/.cache/huggingface)"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the model, don't upload"
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Only upload an existing model, don't download"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the uploaded repository private"
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload Stable Diffusion 2.1 model",
        help="Commit message for the upload"
    )
    
    args = parser.parse_args()
    
    # Step 1: Download model (if not upload-only)
    if not args.upload_only:
        print("="*80)
        print("Step 1: Downloading Model")
        print("="*80)
        local_model_path = download_model_to_local(
            model_id=args.model_id,
            local_dir=args.local_dir
        )
    else:
        # Use provided local directory or default
        if args.local_dir:
            local_model_path = Path(args.local_dir)
        else:
            local_model_path = PROJECT_ROOT / "models" / "stable-diffusion-2-1"
        
        if not local_model_path.exists():
            raise ValueError(
                f"Model path does not exist: {local_model_path}\n"
                "Please provide --local-dir with the path to your model, or run without --upload-only"
            )
        print(f"Using existing model at: {local_model_path}")
    
    # Step 2: Upload to Hub (if not download-only)
    if not args.download_only:
        print("\n" + "="*80)
        print("Step 2: Uploading to Hugging Face Hub")
        print("="*80)
        upload_model_to_hub(
            local_model_path=str(local_model_path),
            repo_id=args.repo_id,
            private=args.private,
            commit_message=args.commit_message
        )
    else:
        print(f"\n✓ Download complete! Model saved to: {local_model_path}")
        print("To upload later, run with --upload-only and --repo-id")


if __name__ == "__main__":
    main()

