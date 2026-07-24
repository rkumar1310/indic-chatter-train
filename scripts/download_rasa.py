"""
download_rasa.py — Download ai4bharat/Rasa dataset and save as WAV + metadata.

Rasa dataset: Multi-language expressive speech from ai4bharat.
  Telugu:  ~52 hours (27h female + 25h male), 30K+ utterances
  Kannada: ~48 hours, similar structure
  Hindi:   ~70 hours
  ...and more languages

Requires HuggingFace login: `huggingface-cli login`

Usage:
  python scripts/download_rasa.py --language Telugu --output_dir data/rasa
  python scripts/download_rasa.py --language Kannada --output_dir data/rasa
  python scripts/download_rasa.py --language Telugu --output_dir data/rasa --max_samples 5000
"""

import argparse
import csv
import io
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


# Map short codes to Rasa config names
LANG_CODE_MAP = {
    "te": "Telugu",
    "kn": "Kannada",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
}

# Reverse map: config name → short code
LANG_NAME_TO_CODE = {v: k for k, v in LANG_CODE_MAP.items()}


def download_rasa_language(language, output_dir, max_samples=None):
    """Download one language from Rasa dataset and save as WAV + metadata CSV."""
    from datasets import load_dataset, Audio

    # Resolve language name
    if language in LANG_CODE_MAP:
        lang_code = language
        config_name = LANG_CODE_MAP[language]
    elif language in LANG_NAME_TO_CODE:
        config_name = language
        lang_code = LANG_NAME_TO_CODE[language]
    else:
        # Try as-is (user might pass exact config name like "Telugu")
        config_name = language
        lang_code = language[:2].lower()

    lang_dir = Path(output_dir) / lang_code
    wav_dir = lang_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"Downloading ai4bharat/Rasa {config_name} (code: {lang_code})")
    print(f"Output: {lang_dir}")
    print(f"{'='*60}")

    # Load with audio decoding disabled to avoid torchcodec issues
    print("Loading dataset (this may take a while for first download)...")
    ds = load_dataset("ai4bharat/Rasa", config_name, split="train")

    # Disable audio decoding FIRST — before any sample access
    ds = ds.cast_column("audio", Audio(decode=False))

    # Check available columns
    print(f"Loaded {len(ds)} samples")
    print(f"Columns: {ds.column_names}")

    # Show first sample structure (safe now — audio won't be decoded)
    sample = ds[0]
    print(f"\nSample keys: {list(sample.keys())}")
    for k, v in sample.items():
        if k != "audio":
            print(f"  {k}: {repr(v)[:100]}")
        else:
            print(f"  audio: dict with keys={list(v.keys()) if isinstance(v, dict) else 'N/A'}")

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
        print(f"Using first {len(ds)} samples (--max_samples={max_samples})")

    # Process samples
    metadata_rows = []
    counts = {"male": 0, "female": 0, "unknown": 0}
    total_duration = 0.0
    skipped = 0

    for idx, sample in enumerate(tqdm(ds, desc=f"Processing Rasa {config_name}")):
        # Extract gender — Rasa uses "gender" field
        gender_raw = sample.get("gender", "")
        if isinstance(gender_raw, int):
            gender = "female" if gender_raw == 0 else "male"
        elif isinstance(gender_raw, str):
            gender = gender_raw.lower().strip()
            if gender not in ("male", "female"):
                gender = "unknown"
        else:
            gender = "unknown"
        counts[gender] += 1

        # Extract text — Rasa uses "text" column
        text = str(sample.get("text", "")).strip()
        if not text:
            skipped += 1
            continue

        # Decode audio manually via soundfile (avoids torchcodec)
        audio_data = sample.get("audio", {})
        audio_bytes = audio_data.get("bytes")
        audio_path = audio_data.get("path")

        try:
            if audio_bytes is not None:
                wav_array, sr = sf.read(io.BytesIO(audio_bytes))
            elif audio_path is not None and os.path.exists(audio_path):
                wav_array, sr = sf.read(audio_path)
            else:
                skipped += 1
                continue
        except Exception as e:
            if idx < 5:
                print(f"WARNING: failed to read audio for sample {idx}: {e}")
            skipped += 1
            continue

        # Convert to mono if stereo
        if wav_array.ndim > 1:
            wav_array = wav_array.mean(axis=1)

        duration = len(wav_array) / sr
        total_duration += duration

        # Save WAV
        filename = f"{lang_code}_{gender}_{idx:06d}"
        wav_path = wav_dir / f"{filename}.wav"
        sf.write(str(wav_path), wav_array, sr)

        metadata_rows.append({
            "filename": filename,
            "text": text,
            "gender": gender,
            "duration_sec": round(duration, 2),
            "sample_rate": sr,
        })

    # Write metadata CSV
    metadata_path = lang_dir / "metadata.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text", "gender", "duration_sec", "sample_rate"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Summary
    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE — {config_name}")
    print(f"{'='*60}")
    print(f"  Total samples: {len(metadata_rows)}")
    print(f"  Skipped: {skipped}")
    print(f"  Female: {counts['female']}, Male: {counts['male']}, Unknown: {counts['unknown']}")
    print(f"  Total duration: {total_duration/3600:.2f} hours")
    print(f"  WAVs saved to: {wav_dir}")
    print(f"  Metadata saved to: {metadata_path}")
    print(f"\nNext steps:")
    print(f"  python scripts/preprocess_indictts.py \\")
    print(f"    --data_dir {output_dir} --languages {lang_code} \\")
    print(f"    --tokenizer data/tokenizer/extended_tokenizer.json \\")
    print(f"    --output_dir data/processed_rasa --device cuda")


def main():
    parser = argparse.ArgumentParser(description="Download ai4bharat/Rasa dataset")
    parser.add_argument("--language", type=str, required=True,
                        help="Language to download. Use short code (te, kn, hi, bn) or "
                             "full name (Telugu, Kannada, Hindi, Bengali)")
    parser.add_argument("--output_dir", type=str, default="data/rasa",
                        help="Output directory (default: data/rasa)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    args = parser.parse_args()

    download_rasa_language(args.language, args.output_dir, args.max_samples)


if __name__ == "__main__":
    main()
