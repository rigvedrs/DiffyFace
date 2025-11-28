"""
Main Training Script for DiffyFace

This script handles data preparation and training in one pipeline.
It merges distributed data from team members and trains the LoRA model.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

from prepare_training_data import prepare_all_datasets, merge_all_datasets


def main():
    parser = argparse.ArgumentParser(
        description="Train DiffyFace model with distributed data from team members"
    )
    
    # Data preparation arguments
    parser.add_argument(
        "--data-root",
        type=str,
        default="./Training/data",
        help="Root directory containing dataset folders with member subdirectories"
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Prepare data from distributed team member outputs"
    )
    parser.add_argument(
        "--prepared-data-dir",
        type=str,
        default="./Training/data/prepared",
        help="Directory containing prepared datasets"
    )
    parser.add_argument(
        "--merge-datasets",
        action="store_true",
        help="Merge all datasets into one for training"
    )
    
    # Training arguments (passed to finetune_lora.py)
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./Training/output",
        help="Output directory for trained model"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=768,
        help="Training resolution (default: 768)"
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--learning-rate-text-encoder",
        type=float,
        default=1e-5,
        help="Text encoder learning rate"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="LoRA rank"
    )
    parser.add_argument(
        "--text-encoder-rank",
        type=int,
        default=8,
        help="Text encoder LoRA rank"
    )
    parser.add_argument(
        "--checkpointing-steps",
        type=int,
        default=5000,
        help="Checkpointing steps"
    )
    parser.add_argument(
        "--validation-prompts",
        type=str,
        nargs="+",
        default=None,
        help="Validation prompts"
    )
    parser.add_argument(
        "--num-validation-images",
        type=int,
        default=5,
        help="Number of validation images"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip data preparation (use existing prepared data)"
    )
    
    args = parser.parse_args()
    
    # Step 1: Prepare data if requested
    if args.prepare_data and not args.skip_data_prep:
        print("="*80)
        print("Step 1: Preparing Training Data")
        print("="*80)
        
        prepared = prepare_all_datasets(
            data_root=args.data_root,
            output_base=args.prepared_data_dir,
            datasets=None  # All datasets
        )
        
        if args.merge_datasets:
            print("\n" + "="*80)
            print("Merging All Datasets")
            print("="*80)
            merged_dir = merge_all_datasets(
                dataset_dirs=list(prepared.values()),
                output_dir=Path(args.prepared_data_dir) / "merged"
            )
            train_data_dir = merged_dir
        else:
            # Use first dataset or let user specify
            train_data_dir = list(prepared.values())[0] if prepared else None
            if not train_data_dir:
                print("Error: No datasets prepared")
                return
    else:
        # Use existing prepared data
        prepared_data_path = Path(args.prepared_data_dir)
        
        if args.merge_datasets:
            merged_dir = prepared_data_path / "merged"
            if merged_dir.exists():
                train_data_dir = merged_dir
            else:
                # Merge existing prepared datasets
                dataset_dirs = [
                    prepared_data_path / "FFHQ",
                    prepared_data_path / "EasyPortrait",
                    prepared_data_path / "LAION_Face"
                ]
                train_data_dir = merge_all_datasets(
                    dataset_dirs=[d for d in dataset_dirs if d.exists()],
                    output_dir=merged_dir
                )
        else:
            # Use a specific dataset (default to first available)
            for dataset_name in ["FFHQ", "EasyPortrait", "LAION_Face"]:
                dataset_dir = prepared_data_path / dataset_name
                if dataset_dir.exists():
                    train_data_dir = dataset_dir
                    break
            else:
                print(f"Error: No prepared datasets found in {prepared_data_path}")
                return
    
    print("\n" + "="*80)
    print("Step 2: Starting Training")
    print("="*80)
    print(f"Training data: {train_data_dir}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Step 2: Build training command
    training_script = Path(__file__).parent / "finetune_lora.py"
    
    cmd = [
        sys.executable,
        str(training_script),
        "--pretrained_model_name_or_path", args.pretrained_model,
        "--train_data_dir", str(train_data_dir),
        "--output_dir", args.output_dir,
        "--resolution", str(args.resolution),
        "--train_batch_size", str(args.train_batch_size),
        "--num_train_epochs", str(args.num_train_epochs),
        "--learning_rate", str(args.learning_rate),
        "--learning_rate_text_encoder", str(args.learning_rate_text_encoder),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--rank", str(args.rank),
        "--text_encoder_rank", str(args.text_encoder_rank),
        "--checkpointing_steps", str(args.checkpointing_steps),
        "--num_validation_images", str(args.num_validation_images),
        "--center_crop",
        "--train_text_encoder",
    ]
    
    if args.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    
    if args.validation_prompts:
        cmd.extend(["--validation_prompts"] + args.validation_prompts)
    
    # Run training
    print("\nRunning training command:")
    print(" ".join(cmd))
    print("\n" + "="*80)
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "="*80)
        print("✓ Training completed successfully!")
        print(f"Model saved to: {args.output_dir}")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("✗ Training failed with exit code:", result.returncode)
        print("="*80)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()

