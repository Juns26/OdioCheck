import argparse
import os

from dataset import AudioDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute CQCC tensors for faster training.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset root containing real/original and fake folders."
    )
    parser.add_argument(
        "--cqcc-cache-dir",
        default=os.path.join(os.path.dirname(__file__), "precomputed_features", "cqcc"),
        help="Directory where precomputed CQCC tensors will be stored."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute files even if cached tensors already exist."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = AudioDataset(
        data_dir=args.data_dir,
        augment=False, # CQCC is computed on clean audio only
        cqcc_cache_dir=args.cqcc_cache_dir
    )
    print(f"Precomputing CQCC into: {args.cqcc_cache_dir}")
    dataset.precompute_cqcc_cache(force=args.force)
    print("Finished CQCC preprocessing.")


if __name__ == "__main__":
    main()
