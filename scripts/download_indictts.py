"""
download_indictts.py — Download IndicTTS datasets from HuggingFace and save as WAV + metadata.

Datasets:
  SPRINGLab/IndicTTS-Hindi     11,825 samples  ~10.33 hrs  48kHz
  SPRINGLab/IndicTTS_Telugu     8,576 samples   ~8.74 hrs  48kHz
  SPRINGLab/IndicTTS_Kannada    9,694 samples   ~7.35 hrs  48kHz
  SPRINGLab/IndicTTS_Bengali   12,852 samples  ~15.06 hrs  48kHz

Usage:
  pip install datasets soundfile tqdm
  python scripts/download_indictts.py --output_dir data/indictts --languages hi te kn bn
  python scripts/download_indictts.py --output_dir data/indictts --languages hi     # just Hindi
"""

import argparse
import csv
import io
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm

# NOTE: Hindi uses hyphen, others use underscore
DATASETS = {
    "hi": {
        "hf_id": "SPRINGLab/IndicTTS-Hindi",
        "name": "Hindi",
        "gender_type": "classlabel",  # 0=female, 1=male
    },
    "te": {
        "hf_id": "SPRINGLab/IndicTTS_Telugu",
        "name": "Telugu",
        "gender_type": "classlabel",
    },
    "kn": {
        "hf_id": "SPRINGLab/IndicTTS_Kannada",
        "name": "Kannada",
        "gender_type": "classlabel",
    },
    "bn": {
        "hf_id": "SPRINGLab/IndicTTS_Bengali",
        "name": "Bengali",
        "gender_type": "string",  # Bengali uses "male"/"female" strings
    },
    "ta": {
        "hf_id": "SPRINGLab/IndicTTS_Tamil",
        "name": "Tamil",
        "gender_type": "classlabel",
    },
    "ml": {
        "hf_id": "SPRINGLab/IndicTTS_Malayalam",
        "name": "Malayalam",
        "gender_type": "classlabel",
    },
    "mr": {
        "hf_id": "SPRINGLab/IndicTTS_Marathi",
        "name": "Marathi",
        "gender_type": "classlabel",
    },
    "gu": {
        "hf_id": "SPRINGLab/IndicTTS_Gujarati",
        "name": "Gujarati",
        "gender_type": "classlabel",
    },
}


def normalize_gender(sample, gender_type):
    """Normalize gender to 'male'/'female' string across datasets."""
    if gender_type == "classlabel":
        return "female" if sample["gender"] == 0 else "male"
    else:
        return sample["gender"].lower().strip()


def download_language(lang_code, output_dir, num_proc=4):
    """Download one language dataset and save as WAV files + metadata CSV."""
    info = DATASETS[lang_code]
    lang_dir = Path(output_dir) / lang_code
    wav_dir = lang_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading {info['name']} ({info['hf_id']})")
    print(f"Output: {lang_dir}")
    print(f"{'='*60}")

    # Load dataset with audio decoding disabled to avoid torchcodec dependency.
    # We decode audio manually with soundfile instead.
    ds = load_dataset(info["hf_id"], split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    print(f"Loaded {len(ds)} samples")

    # Prepare metadata CSV
    metadata_path = lang_dir / "metadata.csv"
    metadata_rows = []

    counts = {"male": 0, "female": 0}
    total_duration = 0.0

    for idx, sample in enumerate(tqdm(ds, desc=f"Processing {info['name']}")):
        gender = normalize_gender(sample, info["gender_type"])
        counts[gender] += 1

        # Decode audio manually via soundfile (avoids torchcodec)
        audio_bytes = sample["audio"]["bytes"]
        audio_path = sample["audio"].get("path")

        if audio_bytes is not None:
            wav_array, sr = sf.read(io.BytesIO(audio_bytes))
        elif audio_path is not None:
            wav_array, sr = sf.read(audio_path)
        else:
            print(f"WARNING: skipping sample {idx} — no audio data")
            continue

        duration = len(wav_array) / sr
        total_duration += duration

        # Save WAV file
        filename = f"{lang_code}_{gender}_{idx:06d}"
        wav_path = wav_dir / f"{filename}.wav"
        sf.write(str(wav_path), wav_array, sr)

        # Collect metadata
        text = sample["text"].strip()
        metadata_rows.append({
            "filename": filename,
            "text": text,
            "gender": gender,
            "duration_sec": round(duration, 2),
            "sample_rate": sr,
        })

    # Write metadata CSV
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text", "gender", "duration_sec", "sample_rate"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Summary
    print(f"\n{info['name']} Summary:")
    print(f"  Total samples: {len(metadata_rows)}")
    print(f"  Female: {counts['female']}, Male: {counts['male']}")
    print(f"  Total duration: {total_duration/3600:.2f} hours")
    print(f"  WAVs saved to: {wav_dir}")
    print(f"  Metadata saved to: {metadata_path}")

    return {
        "lang": lang_code,
        "name": info["name"],
        "total_samples": len(metadata_rows),
        "female_count": counts["female"],
        "male_count": counts["male"],
        "total_hours": round(total_duration / 3600, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Download IndicTTS datasets from HuggingFace")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/indictts",
        help="Root output directory (default: data/indictts)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Languages to download (default: all). Options: hi te kn bn",
    )
    args = parser.parse_args()

    print(f"Output directory: {args.output_dir}")
    print(f"Languages: {', '.join(args.languages)}")

    summaries = []
    for lang in args.languages:
        summary = download_language(lang, args.output_dir)
        summaries.append(summary)

    # Print final summary
    print(f"\n{'='*60}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"{'Language':<12} {'Samples':>8} {'Female':>8} {'Male':>8} {'Hours':>8}")
    print("-" * 52)
    total_hours = 0
    total_samples = 0
    for s in summaries:
        print(f"{s['name']:<12} {s['total_samples']:>8} {s['female_count']:>8} {s['male_count']:>8} {s['total_hours']:>8}")
        total_hours += s["total_hours"]
        total_samples += s["total_samples"]
    print("-" * 52)
    print(f"{'TOTAL':<12} {total_samples:>8} {'':>8} {'':>8} {total_hours:>8}")

    print(f"\nData structure:")
    print(f"  {args.output_dir}/")
    for lang in args.languages:
        print(f"    {lang}/")
        print(f"      wavs/        ← {DATASETS[lang]['name']} WAV files (48kHz)")
        print(f"      metadata.csv ← filename|text|gender|duration|sr")


if __name__ == "__main__":
    main()
