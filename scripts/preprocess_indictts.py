"""
preprocess_indictts.py — Convert downloaded IndicTTS data to T3 training format.

Reads WAV files + metadata.csv from download_indictts.py output,
produces train.json with (text_tokens, speech_tokens, speaker_id, ...) pairs.

Prerequisites:
  - Run download_indictts.py first
  - Chatterbox model weights downloaded (auto-downloads from HuggingFace on first run)

Usage:
  python scripts/preprocess_indictts.py --data_dir data/indictts --languages hi
  python scripts/preprocess_indictts.py --data_dir data/indictts --languages hi te kn bn
  python scripts/preprocess_indictts.py --data_dir data/indictts --languages hi --device cuda
  python scripts/preprocess_indictts.py --data_dir data/indictts --languages hi --batch_size 16
"""

import argparse
import csv
import json
import os
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm

from chatterbox.models.s3tokenizer import S3Tokenizer, S3_SR
from chatterbox.models.tokenizers import MTLTokenizer
from chatterbox.models.s3gen import S3Gen
from huggingface_hub import snapshot_download


# Speaker ID mapping: (lang, gender) → integer
# Extend this as you add languages
SPEAKER_ID_MAP = {
    ("hi", "female"): 0,  ("hi", "male"):  1,
    ("te", "female"): 2,  ("te", "male"):  3,
    ("kn", "female"): 4,  ("kn", "male"):  5,
    ("bn", "female"): 6,  ("bn", "male"):  7,
    ("ta", "female"): 8,  ("ta", "male"):  9,
    ("ml", "female"): 10, ("ml", "male"):  11,
    ("mr", "female"): 12, ("mr", "male"):  13,
    ("gu", "female"): 14, ("gu", "male"):  15,
}


def load_s3_tokenizer(device="cuda"):
    """Load S3Tokenizer from Chatterbox model weights."""
    print("Loading S3Tokenizer from Chatterbox weights...")

    ckpt_dir = Path(
        snapshot_download(
            repo_id="ResembleAI/chatterbox",
            repo_type="model",
            revision="main",
            allow_patterns=["s3gen.pt"],
            token=os.getenv("HF_TOKEN"),
        )
    )

    # S3Gen contains the S3Tokenizer as self.tokenizer
    s3gen = S3Gen()
    s3gen.load_state_dict(
        torch.load(ckpt_dir / "s3gen.pt", map_location="cpu", weights_only=True)
    )

    # Extract just the tokenizer
    s3_tokenizer = s3gen.tokenizer
    s3_tokenizer.to(device).eval()

    print(f"S3Tokenizer loaded on {device}")
    return s3_tokenizer


def load_text_tokenizer(tokenizer_path=None):
    """Load MTLTokenizer from custom path or Chatterbox model weights."""
    if tokenizer_path:
        print(f"Loading MTLTokenizer from {tokenizer_path}")
        tokenizer = MTLTokenizer(tokenizer_path)
        vocab_size = len(tokenizer.tokenizer.get_vocab())
        print(f"MTLTokenizer loaded (vocab={vocab_size})")
        return tokenizer

    print("Loading MTLTokenizer from Chatterbox weights...")

    ckpt_dir = Path(
        snapshot_download(
            repo_id="ResembleAI/chatterbox",
            repo_type="model",
            revision="main",
            allow_patterns=["grapheme_mtl_merged_expanded_v1.json"],
            token=os.getenv("HF_TOKEN"),
        )
    )

    tokenizer = MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))
    print("MTLTokenizer loaded")
    return tokenizer


def process_audio_batch(wav_paths, s3_tokenizer, device, target_sr=S3_SR):
    """
    Load a batch of WAV files, resample to 16kHz, and extract speech tokens.
    Processes one at a time (S3Tokenizer handles list input but processes sequentially).
    """
    results = []

    for wav_path in wav_paths:
        try:
            # Load and resample to 16kHz
            wav, sr = librosa.load(wav_path, sr=target_sr, mono=True)
            duration = len(wav) / target_sr

            # S3Tokenizer expects list of numpy arrays
            with torch.no_grad():
                speech_tokens, token_lens = s3_tokenizer([wav])

            speech_tokens = speech_tokens[0].cpu().tolist()
            results.append({
                "speech_tokens": speech_tokens,
                "duration_sec": round(duration, 2),
                "n_tokens": len(speech_tokens),
                "success": True,
            })
        except Exception as e:
            print(f"WARNING: failed to process {wav_path}: {e}")
            results.append({"success": False, "error": str(e)})

    return results


def process_language(
    lang_code,
    data_dir,
    output_dir,
    s3_tokenizer,
    text_tokenizer,
    device,
    min_duration=1.0,
    max_duration=30.0,
):
    """Process one language: read WAVs + metadata, produce train.json."""
    lang_dir = Path(data_dir) / lang_code
    metadata_path = lang_dir / "metadata.csv"
    wav_dir = lang_dir / "wavs"

    if not metadata_path.exists():
        print(f"ERROR: {metadata_path} not found. Run download_indictts.py first.")
        return None

    # Read metadata
    samples = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)

    print(f"\n{'='*60}")
    print(f"Processing {lang_code.upper()}: {len(samples)} samples")
    print(f"{'='*60}")

    # Process each sample
    output_samples = []
    skipped = {"too_short": 0, "too_long": 0, "audio_error": 0, "tokenizer_error": 0}

    for sample in tqdm(samples, desc=f"Processing {lang_code}"):
        filename = sample["filename"]
        transcript = sample["text"]
        gender = sample["gender"]
        duration = float(sample["duration_sec"])

        # Duration filter
        if duration < min_duration:
            skipped["too_short"] += 1
            continue
        if duration > max_duration:
            skipped["too_long"] += 1
            continue

        wav_path = wav_dir / f"{filename}.wav"
        if not wav_path.exists():
            skipped["audio_error"] += 1
            continue

        # --- Step 1: Extract speech tokens (S3Tokenizer) ---
        try:
            wav_16k, _ = librosa.load(str(wav_path), sr=S3_SR, mono=True)

            with torch.no_grad():
                speech_tokens, token_lens = s3_tokenizer([wav_16k])

            speech_tokens_list = speech_tokens[0].cpu().tolist()
        except Exception as e:
            print(f"WARNING: S3Tokenizer failed on {filename}: {e}")
            skipped["audio_error"] += 1
            continue

        # --- Step 2: Tokenize text (MTLTokenizer) ---
        try:
            text_tokens_list = text_tokenizer.encode(
                transcript,
                language_id=lang_code,
            )
        except Exception as e:
            print(f"WARNING: text tokenizer failed on {filename}: {e}")
            skipped["tokenizer_error"] += 1
            continue

        # --- Step 3: Speaker ID ---
        speaker_key = (lang_code, gender)
        speaker_id = SPEAKER_ID_MAP.get(speaker_key, -1)
        if speaker_id == -1:
            print(f"WARNING: unknown speaker key {speaker_key}, assigning -1")

        # --- Step 4: Assemble training sample ---
        output_samples.append({
            "text_tokens": text_tokens_list,
            "speech_tokens": speech_tokens_list,
            "speaker_id": speaker_id,
            "transcript": transcript,
            "lang": lang_code,
            "gender": gender,
            "audio_file": str(wav_path),
            "duration_sec": round(len(wav_16k) / S3_SR, 2),
            "n_speech_tokens": len(speech_tokens_list),
            "n_text_tokens": len(text_tokens_list),
        })

    # --- Save output ---
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{lang_code}_train.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_samples, f, ensure_ascii=False, indent=2)

    # --- Summary ---
    total_duration = sum(s["duration_sec"] for s in output_samples)
    total_speech_tokens = sum(s["n_speech_tokens"] for s in output_samples)
    total_text_tokens = sum(s["n_text_tokens"] for s in output_samples)

    female_count = sum(1 for s in output_samples if s["gender"] == "female")
    male_count = sum(1 for s in output_samples if s["gender"] == "male")

    summary = {
        "lang": lang_code,
        "total_samples": len(output_samples),
        "female": female_count,
        "male": male_count,
        "total_hours": round(total_duration / 3600, 2),
        "total_speech_tokens": total_speech_tokens,
        "total_text_tokens": total_text_tokens,
        "avg_speech_tokens_per_sample": round(total_speech_tokens / max(len(output_samples), 1), 1),
        "avg_text_tokens_per_sample": round(total_text_tokens / max(len(output_samples), 1), 1),
        "skipped": skipped,
        "output_file": str(out_path),
    }

    print(f"\n{lang_code.upper()} Summary:")
    print(f"  Samples:       {summary['total_samples']} ({female_count}F + {male_count}M)")
    print(f"  Duration:      {summary['total_hours']} hours")
    print(f"  Speech tokens: {total_speech_tokens:,} (avg {summary['avg_speech_tokens_per_sample']}/sample)")
    print(f"  Text tokens:   {total_text_tokens:,} (avg {summary['avg_text_tokens_per_sample']}/sample)")
    print(f"  Skipped:       {skipped}")
    print(f"  Saved to:      {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Preprocess IndicTTS data for T3 training")
    parser.add_argument(
        "--data_dir", type=str, default="data/indictts",
        help="Root directory of downloaded IndicTTS data (default: data/indictts)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/processed",
        help="Output directory for training JSON files (default: data/processed)",
    )
    parser.add_argument(
        "--languages", nargs="+", default=["hi"],
        choices=["hi", "te", "kn", "bn", "ta", "ml", "mr", "gu"],
        help="Languages to process (default: hi)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for S3Tokenizer (default: cpu)",
    )
    parser.add_argument(
        "--min_duration", type=float, default=1.0,
        help="Minimum audio duration in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--max_duration", type=float, default=30.0,
        help="Maximum audio duration in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="Path to custom tokenizer JSON (e.g. extended_tokenizer.json for Telugu/Kannada/Bengali). "
             "Default: downloads original from HuggingFace.",
    )
    args = parser.parse_args()

    print(f"Data directory:   {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Languages:        {', '.join(args.languages)}")
    print(f"Device:           {args.device}")
    print(f"Duration filter:  {args.min_duration}s - {args.max_duration}s")

    # Load models (once, shared across languages)
    s3_tokenizer = load_s3_tokenizer(device=args.device)
    text_tokenizer = load_text_tokenizer(args.tokenizer)

    # Process each language
    summaries = []
    for lang in args.languages:
        summary = process_language(
            lang_code=lang,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            s3_tokenizer=s3_tokenizer,
            text_tokenizer=text_tokenizer,
            device=args.device,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
        if summary:
            summaries.append(summary)

    # Final summary
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"{'Lang':<6} {'Samples':>8} {'Hours':>8} {'Speech Toks':>12} {'Text Toks':>10} {'Skipped':>8}")
    print("-" * 58)
    for s in summaries:
        total_skipped = sum(s["skipped"].values())
        print(f"{s['lang']:<6} {s['total_samples']:>8} {s['total_hours']:>8} {s['total_speech_tokens']:>12,} {s['total_text_tokens']:>10,} {total_skipped:>8}")

    print(f"\nOutput files:")
    for s in summaries:
        print(f"  {s['output_file']}")

    # Save combined summary
    summary_path = Path(args.output_dir) / "preprocessing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
