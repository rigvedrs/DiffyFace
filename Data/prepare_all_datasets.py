"""
Master Script to Prepare All Datasets

This script provides a unified interface to prepare all three datasets
(FFHQ, EasyPortrait, LAION-Face) for caption generation and finetuning.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Data.FFHQ.download_and_prepare import prepare_ffhq_for_training
from Data.EasyPortrait.download_and_prepare import prepare_easyportrait_for_training
from Data.LAION_Face.download_and_prepare import prepare_laion_face_for_training


def prepare_all_datasets(
    ffhq_source: str = None,
    easyportrait_source: str = None,
    laion_face_source: str = None,
    output_base: str = "./data/processed"
):
    """
    Prepare all datasets for training.
    
    Args:
        ffhq_source: Path to FFHQ images directory
        easyportrait_source: Path to EasyPortrait images directory
        laion_face_source: Path to LAION-Face images directory
        output_base: Base output directory
    """
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Preparing All Datasets for DiffyFace Training")
    print("="*80)
    
    # Prepare FFHQ
    if ffhq_source:
        print("\n" + "="*80)
        print("Preparing FFHQ Dataset")
        print("="*80)
        prepare_ffhq_for_training(
            source_dir=ffhq_source,
            output_dir=str(output_base / "ffhq"),
            resize=True,
            target_size=768
        )
    
    # Prepare EasyPortrait
    if easyportrait_source:
        print("\n" + "="*80)
        print("Preparing EasyPortrait Dataset")
        print("="*80)
        prepare_easyportrait_for_training(
            source_dir=easyportrait_source,
            output_dir=str(output_base / "easyportrait")
        )
    
    # Prepare LAION-Face
    if laion_face_source:
        print("\n" + "="*80)
        print("Preparing LAION-Face Dataset")
        print("="*80)
        prepare_laion_face_for_training(
            source_dir=laion_face_source,
            output_dir=str(output_base / "laion_face"),
            filter_quality=True,
            num_samples=156000
        )
    
    print("\n" + "="*80)
    print("All Datasets Prepared Successfully!")
    print("="*80)
    print(f"Output directory: {output_base}")
    print("\nNext steps:")
    print("1. Generate captions for each dataset")
    print("2. Merge datasets (optional)")
    print("3. Ready for finetuning!")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare all datasets for DiffyFace training"
    )
    parser.add_argument(
        "--ffhq",
        type=str,
        default=None,
        help="Path to FFHQ images directory"
    )
    parser.add_argument(
        "--easyportrait",
        type=str,
        default=None,
        help="Path to EasyPortrait images directory"
    )
    parser.add_argument(
        "--laion-face",
        type=str,
        default=None,
        help="Path to LAION-Face images directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/processed",
        help="Base output directory for all processed datasets"
    )
    
    args = parser.parse_args()
    
    if not any([args.ffhq, args.easyportrait, args.laion_face]):
        print("Error: At least one dataset source must be provided")
        print("Use --ffhq, --easyportrait, or --laion-face")
        return
    
    prepare_all_datasets(
        ffhq_source=args.ffhq,
        easyportrait_source=args.easyportrait,
        laion_face_source=args.laion_face,
        output_base=args.output
    )


if __name__ == "__main__":
    main()

