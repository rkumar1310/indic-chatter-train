"""
extract_conds.py — Extract per-speaker conds.pt files for T3 training.

For each speaker in the dataset, selects the best reference audio clip
(closest to 10 seconds, clean single-speaker) and runs prepare_conditionals()
to produce T3Cond + S3Gen ref_dict. Saves as conds/{speaker_name}.pt.

These conds.pt files are used during training so the model can condition on
speaker identity without re-running the encoder pipeline each step.

Usage:
  python scripts/extract_conds.py --data_dir data/indictts --languages hi --device cuda
  python scripts/extract_conds.py --data_dir data/indictts --languages hi te kn bn --device cpu
"""

import argparse
import csv
import json
from pathlib import Path

import torch


TARGET_DURATION = 10.0  # Ideal reference clip length (seconds)
MIN_DURATION = 5.0      # Skip clips shorter than this
MAX_DURATION = 15.0     # Skip clips longer than this


def find_best_reference(metadata_path, wav_dir, gender, target_dur=TARGET_DURATION):
    """
    Find the audio clip closest to target_dur seconds for a given gender.
    Prefers clips in the 5-15s range, picks the one closest to 10s.
    """
    candidates = []

    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["gender"] != gender:
                continue
            dur = float(row["duration_sec"])
            if dur < MIN_DURATION or dur > MAX_DURATION:
                continue
            wav_path = wav_dir / f"{row['filename']}.wav"
            if wav_path.exists():
                candidates.append((wav_path, dur, row["text"]))

    if not candidates:
        # Fallback: relax duration constraints
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["gender"] != gender:
                    continue
                dur = float(row["duration_sec"])
                if dur < 3.0:
                    continue
                wav_path = wav_dir / f"{row['filename']}.wav"
                if wav_path.exists():
                    candidates.append((wav_path, dur, row["text"]))

    if not candidates:
        return None, None, None

    # Pick closest to target duration
    candidates.sort(key=lambda x: abs(x[1] - target_dur))
    return candidates[0]


def extract_conds_for_language(model, lang_code, data_dir, output_dir):
    """Extract conds.pt for each speaker (male/female) in a language."""
    lang_dir = Path(data_dir) / lang_code
    metadata_path = lang_dir / "metadata.csv"
    wav_dir = lang_dir / "wavs"

    if not metadata_path.exists():
        print(f"ERROR: {metadata_path} not found. Run download_indictts.py first.")
        return []

    results = []

    for gender in ["female", "male"]:
        speaker_name = f"{lang_code}_{gender}"
        wav_path, duration, text = find_best_reference(metadata_path, wav_dir, gender)

        if wav_path is None:
            print(f"WARNING: no suitable reference audio for {speaker_name}")
            continue

        print(f"\n  {speaker_name}:")
        print(f"    Reference: {wav_path.name} ({duration:.1f}s)")
        print(f"    Text: {text[:80]}{'...' if len(text) > 80 else ''}")

        # Extract conditionals
        with torch.no_grad():
            model.prepare_conditionals(str(wav_path), exaggeration=0.5)

        # Save
        out_path = Path(output_dir) / f"{speaker_name}.pt"
        model.conds.save(out_path)

        results.append({
            "speaker": speaker_name,
            "lang": lang_code,
            "gender": gender,
            "reference_wav": str(wav_path),
            "reference_duration_sec": duration,
            "reference_text": text,
            "conds_path": str(out_path),
        })

        print(f"    Saved: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract per-speaker conds.pt files")
    parser.add_argument("--data_dir", type=str, default="data/indictts",
                        help="Root directory of downloaded IndicTTS data")
    parser.add_argument("--output_dir", type=str, default="conds",
                        help="Output directory for conds.pt files (default: conds)")
    parser.add_argument("--languages", nargs="+", default=["hi"],
                        choices=["hi", "te", "kn", "bn", "ta", "ml", "mr", "gu"],
                        help="Languages to process (default: hi)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (default: cpu)")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading Chatterbox-Multilingual model...")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_pretrained(args.device)
    print(f"Model loaded on {args.device}")

    # Extract conds for each language
    all_results = []
    for lang in args.languages:
        print(f"\n{'='*50}")
        print(f"Extracting conds for {lang.upper()}")
        print(f"{'='*50}")
        results = extract_conds_for_language(model, lang, args.data_dir, args.output_dir)
        all_results.extend(results)

    # Save manifest
    manifest_path = Path(args.output_dir) / "conds_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # Summary
    print(f"\n{'='*50}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*50}")
    for r in all_results:
        print(f"  {r['speaker']}: {r['conds_path']} (from {r['reference_duration_sec']:.1f}s clip)")
    print(f"\nManifest: {manifest_path}")
    print(f"\nUsage in training:")
    print(f"  conds = Conditionals.load('conds/hi_female.pt')")


if __name__ == "__main__":
    main()
