#!/usr/bin/env python3
"""
extend_t3_embeddings.py — Extend T3's text_emb and text_head for new vocab tokens.

Resizes the embedding and projection layers to accommodate new grapheme tokens,
using Brahmic cross-script initialization to warm-start new embeddings from Devanagari.

Supports two modes:
  1. Fresh extension from HuggingFace (original 2454 vocab → extended)
  2. Incremental extension from training checkpoint (e.g., 2529 → 2529+N)
     Uses --base_checkpoint to load trained embeddings + LoRA from a previous round.

Usage:
  # Fresh extension (Telugu from scratch):
  python scripts/extend_t3_embeddings.py \
    --tokenizer data/tokenizer/extended_tokenizer.json \
    --init_map data/tokenizer/brahmic_init_map.json \
    --output data/models/t3_extended.pt

  # Incremental extension (Kannada on top of trained Telugu):
  python scripts/extend_t3_embeddings.py \
    --tokenizer data/tokenizer/extended_tokenizer.json \
    --init_map data/tokenizer/brahmic_init_map.json \
    --base_checkpoint checkpoints/round4_telugu/best.pt \
    --output data/models/t3_extended_kn.pt
"""

import argparse
import json
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Extend T3 embeddings for new vocab tokens")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to extended tokenizer JSON")
    parser.add_argument("--init_map", type=str, required=True,
                        help="Path to brahmic_init_map.json")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="Optional: LoRA checkpoint to carry forward (for fresh extension)")
    parser.add_argument("--base_checkpoint", type=str, default=None,
                        help="Training checkpoint with trained embeddings + LoRA to build on "
                             "(for incremental extension, e.g. adding Kannada after Telugu training)")
    parser.add_argument("--output", type=str, default="data/models/t3_extended.pt",
                        help="Output path for extended T3 state dict")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for loading (default: cpu)")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ─── Load init map ───
    with open(args.init_map, "r") as f:
        init_data = json.load(f)

    original_vocab_size = init_data["original_vocab_size"]
    new_vocab_size = init_data["extended_vocab_size"]
    init_map = {int(k): v for k, v in init_data["map"].items()}

    print(f"Vocab extension: {original_vocab_size} → {new_vocab_size} (+{new_vocab_size - original_vocab_size})")
    print(f"Brahmic init mappings: {len(init_map)}")

    # ─── Load T3 model ───
    from safetensors.torch import load_file as load_safetensors
    from huggingface_hub import snapshot_download
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config

    # Always start from the HuggingFace base model
    print("\nDownloading T3 base weights...")
    ckpt_dir = Path(
        snapshot_download(
            repo_id="ResembleAI/chatterbox",
            repo_type="model",
            revision="main",
            allow_patterns=["t3_mtl23ls_v2.safetensors"],
            token=os.getenv("HF_TOKEN"),
        )
    )

    config = T3Config.multilingual()
    t3 = T3(hp=config)

    print("Loading T3 base state dict...")
    state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
    if "model" in state:
        state = state["model"][0]
    t3.load_state_dict(state)

    base_hf_vocab = t3.text_emb.weight.shape[0]  # 2454 from HuggingFace
    hidden_dim = t3.text_emb.weight.shape[1]

    # ─── If base_checkpoint provided: apply trained embeddings from previous round ───
    lora_state = None
    if args.base_checkpoint:
        print(f"\nLoading trained checkpoint: {args.base_checkpoint}")
        ckpt = torch.load(args.base_checkpoint, map_location=args.device, weights_only=False)

        ckpt_vocab_size = ckpt.get("new_vocab_size", None)
        ckpt_emb_weight = ckpt.get("text_emb_weight", None)
        ckpt_head_weight = ckpt.get("text_head_weight", None)

        if ckpt_emb_weight is not None and ckpt_head_weight is not None:
            ckpt_old_vocab = ckpt_emb_weight.shape[0]
            print(f"  Checkpoint has trained embeddings: ({ckpt_old_vocab}, {hidden_dim})")

            # Resize text_emb to match checkpoint
            if ckpt_old_vocab != base_hf_vocab:
                new_emb = torch.nn.Embedding(ckpt_old_vocab, hidden_dim)
                new_emb.weight.data[:base_hf_vocab] = t3.text_emb.weight.data
                t3.text_emb = new_emb

                new_head = torch.nn.Linear(hidden_dim, ckpt_old_vocab, bias=False)
                new_head.weight.data[:base_hf_vocab] = t3.text_head.weight.data
                t3.text_head = new_head

            # Copy trained embeddings (these include trained Telugu rows etc.)
            t3.text_emb.weight.data.copy_(ckpt_emb_weight)
            t3.text_head.weight.data.copy_(ckpt_head_weight)
            print(f"  Applied trained embeddings ({ckpt_old_vocab} tokens)")
        else:
            print(f"  WARNING: checkpoint has no extended embeddings, using HF base only")

        # Extract LoRA to carry forward
        if "lora_state_dict" in ckpt and ckpt["lora_state_dict"]:
            lora_state = ckpt["lora_state_dict"]
            print(f"  LoRA keys from checkpoint: {len(lora_state)}")
            ckpt_loss = ckpt.get('val_loss', ckpt.get('loss', '?'))
            print(f"  From epoch={ckpt.get('epoch', '?')}, loss={ckpt_loss}")
        else:
            print(f"  No LoRA state in checkpoint")

    # Current model state (after optional checkpoint application)
    old_emb_weight = t3.text_emb.weight.data.clone()
    old_head_weight = t3.text_head.weight.data.clone()
    old_vocab = old_emb_weight.shape[0]

    assert old_vocab == original_vocab_size, \
        f"Vocab mismatch: model has {old_vocab}, init_map says {original_vocab_size}. " \
        f"Make sure --base_tokenizer was used in extend_tokenizer.py if extending incrementally."
    print(f"\nCurrent text_emb: ({old_vocab}, {hidden_dim})")
    print(f"Current text_head: {old_head_weight.shape}")

    # ─── Extend text_emb ───
    print(f"\nExtending text_emb: ({old_vocab}, {hidden_dim}) → ({new_vocab_size}, {hidden_dim})")

    new_emb = torch.nn.Embedding(new_vocab_size, hidden_dim)

    # Copy existing embeddings
    new_emb.weight.data[:old_vocab] = old_emb_weight

    # Initialize new rows with small random noise (fallback)
    torch.nn.init.normal_(new_emb.weight.data[old_vocab:], mean=0.0, std=0.02)

    # Warm-start from Brahmic init map
    brahmic_init_count = 0
    for new_id, deva_id in init_map.items():
        if new_id < new_vocab_size and deva_id < old_vocab:
            new_emb.weight.data[new_id] = old_emb_weight[deva_id].clone()
            brahmic_init_count += 1

    print(f"  Copied {old_vocab} existing embeddings")
    print(f"  Brahmic warm-start: {brahmic_init_count} embeddings (from Devanagari)")
    print(f"  Random init: {new_vocab_size - old_vocab - brahmic_init_count} embeddings")

    t3.text_emb = new_emb

    # ─── Extend text_head ───
    print(f"\nExtending text_head: ({old_head_weight.shape[0]}, {old_head_weight.shape[1]}) → ({new_vocab_size}, {hidden_dim})")

    new_head = torch.nn.Linear(hidden_dim, new_vocab_size, bias=False)

    # Copy existing head weights
    new_head.weight.data[:old_vocab] = old_head_weight

    # Initialize new rows
    torch.nn.init.normal_(new_head.weight.data[old_vocab:], mean=0.0, std=0.02)

    # Warm-start head from Brahmic map too
    head_init_count = 0
    for new_id, deva_id in init_map.items():
        if new_id < new_vocab_size and deva_id < old_vocab:
            new_head.weight.data[new_id] = old_head_weight[deva_id].clone()
            head_init_count += 1

    print(f"  Brahmic warm-start: {head_init_count} head rows (from Devanagari)")

    t3.text_head = new_head

    # ─── Update config ───
    config.text_tokens_dict_size = new_vocab_size
    t3.hp = config
    print(f"\nUpdated T3Config.text_tokens_dict_size: {original_vocab_size} → {new_vocab_size}")

    # ─── Optionally load LoRA from separate checkpoint (fresh extension mode) ───
    if args.lora_checkpoint and lora_state is None:
        print(f"\nLoading LoRA checkpoint: {args.lora_checkpoint}")
        ckpt = torch.load(args.lora_checkpoint, map_location=args.device, weights_only=False)
        lora_state = ckpt["lora_state_dict"]
        print(f"  LoRA keys: {len(lora_state)}")
        print(f"  From epoch={ckpt.get('epoch', '?')}, step={ckpt.get('step', '?')}")
    elif lora_state is not None:
        print(f"\nLoRA state already loaded from --base_checkpoint ({len(lora_state)} keys)")

    # ─── Save ───
    # hf_base_vocab_size = the original HuggingFace vocab (2454) — rows to FREEZE during training
    # original_vocab_size = what we extended FROM (may be 2454 or 2529 for incremental)
    # text_tokens_dict_size = the final extended size
    save_data = {
        "model_state_dict": t3.state_dict(),
        "config": {
            "text_tokens_dict_size": new_vocab_size,
            "original_vocab_size": original_vocab_size,
            "hf_base_vocab_size": base_hf_vocab,  # Always 2454 — used for gradient masking
            "hidden_dim": hidden_dim,
            "languages_added": init_data["languages_added"],
        },
        "lora_state_dict": lora_state,
    }

    torch.save(save_data, args.output)
    print(f"\nSaved extended T3 → {args.output}")

    # ─── Verify ───
    print(f"\n{'='*50}")
    print("VERIFICATION")
    print(f"{'='*50}")

    # Reload and check
    t3_check = T3(hp=config)
    t3_check.load_state_dict(save_data["model_state_dict"])

    print(f"  text_emb shape: {t3_check.text_emb.weight.shape}")
    print(f"  text_head shape: {t3_check.text_head.weight.shape}")
    print(f"  speech_emb shape: {t3_check.speech_emb.weight.shape} (unchanged)")
    print(f"  speech_head shape: {t3_check.speech_head.weight.shape} (unchanged)")

    # Verify Brahmic init worked — new Telugu embedding should be similar to Devanagari
    if init_map:
        sample_new_id = list(init_map.keys())[0]
        sample_deva_id = init_map[sample_new_id]
        cos_sim = torch.nn.functional.cosine_similarity(
            t3_check.text_emb.weight.data[sample_new_id].unsqueeze(0),
            t3_check.text_emb.weight.data[sample_deva_id].unsqueeze(0),
        )
        print(f"  Brahmic init check: new[{sample_new_id}] vs deva[{sample_deva_id}] cosine={cos_sim.item():.4f} (should be ~1.0)")

    # ─── Summary ───
    print(f"\n{'='*50}")
    print("PHASE 2 COMPLETE")
    print(f"{'='*50}")
    print(f"  Extended T3: {args.output}")
    print(f"  Vocab: {original_vocab_size} → {new_vocab_size}")
    print(f"  Brahmic warm-start: {brahmic_init_count} embeddings")
    print(f"  LoRA carried forward: {'Yes' if lora_state else 'No'}")
    print(f"\nNext steps:")
    print(f"  1. Download + preprocess Telugu data")
    print(f"  2. Run Round 2 training with extended model")
    print(f"  3. Update train_t3_lora.py to load extended model + tokenizer")


if __name__ == "__main__":
    main()
