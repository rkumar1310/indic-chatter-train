#!/usr/bin/env bash
#
# speedrun.sh — download data, extend the tokenizer, warm-start the embeddings,
# and LoRA fine-tune Chatterbox-Multilingual on a new Indian language. One
# command, run it, come back to a checkpoint.
#
# Usage:
#   ./speedrun.sh te                      # Telugu, the language this pipeline was built for
#   ./speedrun.sh kn                      # Kannada
#   DEVICE=cpu EPOCHS=1 ./speedrun.sh te  # smoke test on CPU, no GPU required
#
# Supported language codes: te kn bn ta ml mr gu
# (Marathi ("mr") reuses the Devanagari script Chatterbox already ships with —
#  the tokenizer step still runs, it just won't add any new grapheme rows.)
#
# What this actually runs, in order:
#   1. download_indictts.py  — studio-quality reference audio (Hindi + target language)
#   2. download_rasa.py      — larger, community-recorded corpus for the target language
#   3. extract_conds.py      — per-speaker voice conditioning from the IndicTTS clips
#   4. extend_tokenizer.py   — add the new script's graphemes + Brahmic init map
#   5. extend_t3_embeddings.py — resize text_emb/text_head, warm-start from Devanagari
#   6. preprocess_indictts.py  — audio+text -> (speech_tokens, text_tokens) pairs, x2 (both sources)
#   7. split_data.py         — speaker-stratified train/val/test
#   8. train_t3_lora.py      — the actual LoRA fine-tune, Hindi replay + target language
#
# See README.md for what each step does and why, and for the incremental
# round-by-round recipe this project was actually developed with.

set -euo pipefail

LANG="${1:-te}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-12}"
LR="${LR:-5e-5}"
LANG_WEIGHT="${LANG_WEIGHT:-0.7}"
HI_WEIGHT="${HI_WEIGHT:-0.3}"

case "$LANG" in
  te|kn|bn|ta|ml|mr|gu) ;;
  *) echo "Unsupported language: $LANG (expected one of: te kn bn ta ml mr gu)"; exit 1 ;;
esac

echo "=============================================================="
echo " chatterbox-indic-train speedrun: $LANG"
echo " device=$DEVICE epochs=$EPOCHS batch_size=$BATCH_SIZE lr=$LR"
echo "=============================================================="

step() { echo; echo "── $1 ──────────────────────────────────────────"; }

step "1/8  Downloading IndicTTS (Hindi + $LANG)"
python scripts/download_indictts.py --output_dir data/indictts --languages hi "$LANG"

step "2/8  Downloading Rasa ($LANG) — larger, expressive, community-recorded"
python scripts/download_rasa.py --language "$LANG" --output_dir data/rasa

step "3/8  Extracting speaker conditioning (voice timbre) from IndicTTS clips"
python scripts/extract_conds.py --data_dir data/indictts --languages hi "$LANG" \
  --output_dir conds --device "$DEVICE"

step "4/8  Extending the tokenizer with $LANG graphemes + Brahmic init map"
python scripts/extend_tokenizer.py --languages "$LANG" \
  --output data/tokenizer/extended_tokenizer.json \
  --init_map_output data/tokenizer/brahmic_init_map.json

step "5/8  Extending T3 embeddings (warm-started from Devanagari where possible)"
python scripts/extend_t3_embeddings.py \
  --tokenizer data/tokenizer/extended_tokenizer.json \
  --init_map data/tokenizer/brahmic_init_map.json \
  --output data/models/t3_extended.pt \
  --device "$DEVICE"

step "6/8  Preprocessing audio -> training pairs (IndicTTS, then Rasa)"
python scripts/preprocess_indictts.py --data_dir data/indictts --languages hi "$LANG" \
  --tokenizer data/tokenizer/extended_tokenizer.json \
  --output_dir data/processed --device "$DEVICE"

python scripts/preprocess_indictts.py --data_dir data/rasa --languages "$LANG" \
  --tokenizer data/tokenizer/extended_tokenizer.json \
  --output_dir data/processed_rasa --device "$DEVICE"

step "7/8  Splitting into train/val/test (speaker-stratified)"
python scripts/split_data.py --input data/processed/hi_train.json
python scripts/split_data.py --input "data/processed/${LANG}_train.json"
python scripts/split_data.py --input "data/processed_rasa/${LANG}_train.json"

step "8/8  LoRA fine-tuning T3 (Hindi replay + $LANG, both data sources)"
python scripts/train_t3_lora.py \
  --train_data "data/processed/${LANG}_train.json" "data/processed_rasa/${LANG}_train.json" data/processed/hi_train.json \
  --val_data "data/processed/${LANG}_val.json" data/processed/hi_val.json \
  --extended_model data/models/t3_extended.pt \
  --conds_dir conds/ \
  --output_dir "checkpoints/${LANG}_speedrun" \
  --lang_weights "${LANG}:${LANG_WEIGHT}" "hi:${HI_WEIGHT}" \
  --device "$DEVICE" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LR" --bf16

echo
echo "=============================================================="
echo " Done. Checkpoint: checkpoints/${LANG}_speedrun/best.pt"
echo " Tokenizer:         data/tokenizer/extended_tokenizer.json"
echo
echo " Try it:"
echo "   see example_inference.py for how to load this checkpoint"
echo "=============================================================="
