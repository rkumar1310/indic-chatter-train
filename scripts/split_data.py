"""
split_data.py — Split preprocessed data into train/val/test sets.

Stratified by speaker (gender) so each split has balanced male/female representation.

Usage:
  python scripts/split_data.py --input data/processed/hi_train.json
  python scripts/split_data.py --input data/processed/hi_train.json --val_size 500 --test_size 200
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def split_data(input_path, val_size=500, test_size=100, seed=42):
    """
    Split data stratified by speaker_id.
    val and test get exactly val_size and test_size samples (split evenly across speakers).
    Everything else goes to train.
    """
    random.seed(seed)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group by speaker_id
    by_speaker = defaultdict(list)
    for sample in data:
        by_speaker[sample["speaker_id"]].append(sample)

    n_speakers = len(by_speaker)
    val_per_speaker = val_size // n_speakers
    test_per_speaker = test_size // n_speakers

    train, val, test = [], [], []

    for speaker_id, samples in by_speaker.items():
        random.shuffle(samples)

        test_split = samples[:test_per_speaker]
        val_split = samples[test_per_speaker:test_per_speaker + val_per_speaker]
        train_split = samples[test_per_speaker + val_per_speaker:]

        test.extend(test_split)
        val.extend(val_split)
        train.extend(train_split)

    # Shuffle each split
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    # Save
    base = Path(input_path).parent
    lang = Path(input_path).stem.split("_")[0]  # "hi" from "hi_train.json"

    paths = {}
    for name, split in [("train", train), ("val", val), ("test", test)]:
        out_path = base / f"{lang}_{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        paths[name] = str(out_path)

    # Summary
    def split_stats(split):
        hours = sum(s["duration_sec"] for s in split) / 3600
        speakers = defaultdict(int)
        for s in split:
            speakers[f"{s['gender']}(id={s['speaker_id']})"] += 1
        return len(split), round(hours, 2), dict(speakers)

    print(f"\nSplit results for {lang}:")
    print(f"{'Split':<8} {'Samples':>8} {'Hours':>8}  Speakers")
    print("-" * 55)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        n, hrs, spkrs = split_stats(split)
        print(f"{name:<8} {n:>8} {hrs:>8}  {spkrs}")
    print(f"\nSaved to:")
    for name, path in paths.items():
        print(f"  {path}")


def main():
    parser = argparse.ArgumentParser(description="Split preprocessed data into train/val/test")
    parser.add_argument("--input", type=str, required=True, help="Path to preprocessed JSON")
    parser.add_argument("--val_size", type=int, default=500, help="Validation set size (default: 500)")
    parser.add_argument("--test_size", type=int, default=100, help="Test set size (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    split_data(args.input, args.val_size, args.test_size, args.seed)


if __name__ == "__main__":
    main()
