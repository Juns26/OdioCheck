import subprocess
import sys
import os
import argparse
from dataset import AudioDataset


def run_command(cmd):
    try:
        subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError:
        sys.exit(1)


def download_dataset():
    run_command(["git", "lfs", "install"])
    dataset_dir = "MLAAD-tiny"
    if not os.path.exists(dataset_dir):
        print("=== Cloning MLAAD-tiny dataset ===")
        run_command(["git", "clone", "https://huggingface.co/datasets/mueller91/MLAAD-tiny"])
    else:
        print(f"Dataset directory '{dataset_dir}' already exists. Skipping clone.")


def precompute_cqcc(data_dir, cqcc_cache_dir, force=False):
    for lang in ["en", "de"]:
        print(f"\n--- Precomputing CQCC for language: {lang} ---")
        dataset = AudioDataset(
            data_dir=data_dir,
            augment=False,
            cqcc_cache_dir=cqcc_cache_dir,
            target_lang=lang
        )
        dataset.precompute_cqcc_cache(force=force)
    print("\nFinished all CQCC preprocessing.")


def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset and precompute CQCC features.")
    parser.add_argument("--data-dir", default="MLAAD-tiny")
    parser.add_argument(
        "--cqcc-cache-dir",
        default=os.path.join(os.path.dirname(__file__), "precomputed_features", "cqcc")
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-cqcc", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.skip_download:
        download_dataset()

    if not args.skip_cqcc:
        precompute_cqcc(args.data_dir, args.cqcc_cache_dir, args.force)