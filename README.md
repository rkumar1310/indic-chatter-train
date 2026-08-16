# chatterbox-indic-train

![Atoms of AI](./atoms2.png)

Training scripts for adding Indian languages to [Chatterbox-Multilingual](https://github.com/resemble-ai/chatterbox) via tokenizer extension + LoRA — no phoneme engineering, no G2P, no retraining from scratch.

This is the training-side companion to [chatterbox-indic](https://github.com/reenigne314/chatterbox-indic) (the inference fork) and [chatterbox-indic-lora](https://huggingface.co/reenigne314/chatterbox-indic-lora) (the trained weights on HuggingFace). Those two let you *use* the result. This repo is how it was *made* — download the data, extend the vocabulary, warm-start the embeddings, fine-tune with LoRA, all from source.

Written up in detail on [Atoms of AI](https://theatomsofai.substack.com/p/teaching-an-ai-to-speak-indian-languages) — Part 1 covers the architecture, Part 2 walks through this exact pipeline end to end for Telugu.

## Quick start

```bash
git clone https://github.com/reenigne314/chatterbox-indic-train.git
cd chatterbox-indic-train

# 1. PyTorch first, matched to your GPU (see requirements.txt for the exact commands)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. The chatterbox package these scripts import (S3Tokenizer, MTLTokenizer, T3, S3Gen...)
pip install git+https://github.com/reenigne314/chatterbox-indic.git

# 3. Everything else
pip install -r requirements.txt

# 4. Download, preprocess, and train — one command
./speedrun.sh te
```

That downloads ~60 hours of raw Telugu audio (studio + community-recorded), extends the tokenizer with Telugu's 74 graphemes, warm-starts their embeddings from Devanagari, and LoRA fine-tunes T3 with Hindi replay data mixed in so English/Hindi/the other 21 original languages don't degrade. It produces `checkpoints/te_speedrun/best.pt` — load it with `example_inference.py`.

Swap `te` for `kn`, `bn`, `ta`, `ml`, `mr`, or `gu` for any of the other 6 languages this pipeline has been run on.

```bash
# Smoke test without a GPU or the full dataset — sanity-checks the whole chain
DEVICE=cpu EPOCHS=1 ./speedrun.sh te
```

## What speedrun.sh actually does

| Step | Script | What it does |
|---|---|---|
| 1 | `download_indictts.py` | Pulls [SPRINGLab/IndicTTS](https://huggingface.co/SPRINGLab) — studio-quality, 1 male + 1 female speaker per language |
| 2 | `download_rasa.py` | Pulls [ai4bharat/Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) — larger, expressive, community-recorded (~52h for Telugu vs IndicTTS's ~8.7h) |
| 3 | `extract_conds.py` | Picks a ~10s clean reference clip per speaker, runs it through the frozen voice encoder to get the speaker conditioning used at both train and inference time |
| 4 | `extend_tokenizer.py` | Adds the target script's graphemes to the 2,454-token vocabulary, plus a Brahmic cross-script map (e.g. Telugu "క" → Devanagari "क") for warm-starting |
| 5 | `extend_t3_embeddings.py` | Resizes `text_emb`/`text_head`, copies the new rows from their Devanagari phonetic equivalent where one exists, random-inits the rest |
| 6 | `preprocess_indictts.py` | Runs the frozen S3Tokenizer (audio → speech tokens) and MTLTokenizer (text → text tokens) over every utterance, filtering by duration |
| 7 | `split_data.py` | Speaker-stratified train/val/test split, so no speaker's voice leaks from train into test |
| 8 | `train_t3_lora.py` | LoRA (rank 32, q/k/v/o projections) on T3's Llama backbone, with a gradient-masking hook so the 2,454 original vocabulary rows never update |

Only Step 8 touches model weights, and only ~1.5% of them: the LoRA adapters plus the newly-added embedding rows. Everything else — S3Gen, the voice encoder, the speech token vocabulary, the other 23 languages' text embeddings — stays exactly as Resemble AI shipped it.

## The incremental recipe (what actually happened)

`speedrun.sh` is a one-shot version for convenience. The real Telugu results were produced incrementally, and if you want to reproduce that exact path (or you're debugging why your run's loss looks different from the numbers in Part 2), here's the actual round history:

```bash
# Round 1 — sanity check: does the LoRA + training loop even work?
# (Original vocab, no new languages. 4 GPUs, since nothing about this needs 1.)
torchrun --nproc_per_node=4 scripts/train_t3_lora.py \
  --train_data data/processed/hi_train.json --val_data data/processed/hi_val.json \
  --conds_dir conds/ --output_dir checkpoints/round1_hindi \
  --epochs 10 --batch_size 32 --lr 1e-4 --bf16

# Round 2 — add Telugu (IndicTTS only, ~6.9k utterances)
python scripts/train_t3_lora.py \
  --train_data data/processed/te_train.json data/processed/hi_train.json \
  --val_data data/processed/te_val.json data/processed/hi_val.json \
  --extended_model data/models/t3_extended.pt --conds_dir conds/ \
  --output_dir checkpoints/round2_telugu --lang_weights te:0.5 hi:0.3 \
  --epochs 15 --batch_size 12 --lr 5e-5 --bf16

# Round 3 — more epochs, Telugu-heavier mix, warm-started from Round 2.
# (Spoiler: barely moves the loss. Same ~6.9k utterances, no new signal.)
python scripts/train_t3_lora.py \
  --train_data data/processed/te_train.json data/processed/hi_train.json \
  --val_data data/processed/te_val.json data/processed/hi_val.json \
  --extended_model data/models/t3_extended.pt \
  --warm_start checkpoints/round2_telugu/best.pt --conds_dir conds/ \
  --output_dir checkpoints/round3_telugu --lang_weights te:0.7 hi:0.3 \
  --epochs 20 --batch_size 12 --lr 3e-5 --bf16

# Round 4 — swap in Rasa (26k utterances instead of 6.9k). This is the round
# that actually moved the val loss.
python scripts/train_t3_lora.py \
  --train_data data/processed_rasa/te_train.json data/processed/hi_train.json \
  --val_data data/processed_rasa/te_val.json data/processed/hi_val.json \
  --extended_model data/models/t3_extended.pt \
  --warm_start checkpoints/round3_telugu/best.pt --conds_dir conds/ \
  --output_dir checkpoints/round4_telugu --lang_weights te:0.7 hi:0.3 \
  --epochs 20 --batch_size 12 --lr 3e-5 --bf16
```

Adding a further language on top of an existing one is `extend_t3_embeddings.py --base_checkpoint checkpoints/round4_telugu/best.pt` (carries forward the trained Telugu embeddings + LoRA weights, then extends again for the new script) followed by another `train_t3_lora.py` run with `--warm_start` pointed at that round's checkpoint.

## Hardware

Developed on 4x RTX PRO 6000 Blackwell (96GB each). In practice, a single-language LoRA round doesn't need more than one GPU — `train_t3_lora.py` auto-detects `torchrun` and runs DDP across however many GPUs you launch it with, but every Telugu round above ran on exactly one. Multi-GPU only mattered once all 8 languages were being trained together in a single run.

`--bf16` is recommended on Blackwell/Ampere+; `--fp16` (with grad scaling) is there for older cards.

## A note on `T3.loss()`

If you're fine-tuning against `T3.loss()` (the model's own built-in training method, in the upstream `chatterbox` package) instead of the `compute_loss()` in `train_t3_lora.py`, know that the upstream method does not shift logits against targets before computing cross-entropy — it compares `speech_logits[i]` to `speech_tokens[i]` directly, instead of `speech_tokens[i+1]`. For a causal, autoregressive model that's the wrong target: it turns training into predicting the token you were just shown rather than the next one. `train_t3_lora.py`'s `compute_loss()` shifts explicitly; if you write your own training loop, make sure yours does too. Details and code comparison in [Part 2](https://theatomsofai.substack.com/p/teaching-an-ai-to-speak-indian-languages).

## License

MIT — see [LICENSE](LICENSE). Depends on, but does not vendor, Resemble AI's [Chatterbox](https://github.com/resemble-ai/chatterbox) (also MIT).
