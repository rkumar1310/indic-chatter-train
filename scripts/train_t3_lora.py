#!/usr/bin/env python3
"""
train_t3_lora.py — LoRA fine-tuning for T3 multilingual TTS model.

Supports single-GPU and multi-GPU (DDP) training. Auto-detects DDP when
launched with torchrun.

=== Round 1: Hindi (original vocab) ===
  python scripts/train_t3_lora.py \
    --train_data data/processed/hi_train.json \
    --val_data data/processed/hi_val.json \
    --conds_dir conds/ \
    --output_dir checkpoints/round1_hindi \
    --device cuda --epochs 10 --batch_size 16 --lr 1e-4 --bf16

=== Round 2: Telugu + Hindi + English (extended vocab) ===
  python scripts/train_t3_lora.py \
    --train_data data/processed/te_train.json data/processed/hi_train.json \
    --val_data data/processed/te_val.json data/processed/hi_val.json \
    --extended_model data/models/t3_extended.pt \
    --conds_dir conds/ \
    --output_dir checkpoints/round2_telugu \
    --lang_weights te:0.5 hi:0.3 en:0.2 \
    --device cuda --epochs 15 --batch_size 12 --lr 5e-5 --bf16

=== Round 3: Telugu-heavy (warm-start from Round 2) ===
  python scripts/train_t3_lora.py \
    --train_data data/processed/te_train.json data/processed/hi_train.json \
    --val_data data/processed/te_val.json data/processed/hi_val.json \
    --extended_model data/models/t3_extended.pt \
    --warm_start checkpoints/round2_telugu/best.pt \
    --conds_dir conds/ \
    --output_dir checkpoints/round3_telugu \
    --lang_weights te:0.7 hi:0.3 \
    --device cuda --epochs 20 --batch_size 12 --lr 3e-5 --bf16

=== Multi-GPU (4x RTX PRO 6000) ===
  torchrun --nproc_per_node=4 scripts/train_t3_lora.py \
    --train_data data/processed/hi_train.json \
    --val_data data/processed/hi_val.json \
    --conds_dir conds/ \
    --output_dir checkpoints/round1_hindi \
    --epochs 10 --batch_size 8 --lr 1e-4 --bf16

=== Quick sanity check ===
  python scripts/train_t3_lora.py \
    --train_data data/processed/hi_train.json \
    --val_data data/processed/hi_val.json \
    --conds_dir conds/ \
    --output_dir checkpoints/debug \
    --device cpu --epochs 1 --batch_size 2 --max_steps 10
"""

import argparse
import json
import logging
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────
def setup_distributed():
    """Auto-detect DDP from torchrun environment variables."""
    if "RANK" not in os.environ:
        return 0, 1, 0, False  # rank, world_size, local_rank, is_ddp

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, True


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank):
    return rank == 0


# ─────────────────────────────────────────────
# Speaker ID → conds filename mapping
# Must match SPEAKER_ID_MAP in preprocess_indictts.py
# ─────────────────────────────────────────────
SPEAKER_CONDS_MAP = {
    0: "hi_female",   1: "hi_male",
    2: "te_female",   3: "te_male",
    4: "kn_female",   5: "kn_male",
    6: "bn_female",   7: "bn_male",
    8: "ta_female",   9: "ta_male",
    10: "ml_female",  11: "ml_male",
    12: "mr_female",  13: "mr_male",
    14: "gu_female",  15: "gu_male",
}


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class T3Dataset(Dataset):
    """
    Loads preprocessed JSON training data.
    Each sample has: text_tokens, speech_tokens, speaker_id, etc.
    Adds BOT/EOT to text and BOS/EOS to speech at __getitem__ time.
    """

    def __init__(self, json_paths, hp, max_text_tokens=400, max_speech_tokens=1000):
        self.samples = []
        for p in json_paths:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            for s in data:
                if len(s["text_tokens"]) > max_text_tokens:
                    continue
                if len(s["speech_tokens"]) > max_speech_tokens:
                    continue
                self.samples.append(s)

        self.hp = hp

        # Stats (only log from rank 0, but compute always for filtering)
        stats = defaultdict(int)
        for s in self.samples:
            stats[f"{s.get('lang', '?')}_{s.get('gender', '?')}"] += 1
        self._stats = dict(stats)
        self._n_files = len(json_paths)

    def log_stats(self):
        logger.info(f"Loaded {len(self.samples)} samples from {self._n_files} file(s)")
        for k, v in sorted(self._stats.items()):
            logger.info(f"  {k}: {v} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Add BOT (255) / EOT (0) to text tokens
        text_tokens = [self.hp.start_text_token] + s["text_tokens"] + [self.hp.stop_text_token]

        # Add BOS (6561) / EOS (6562) to speech tokens
        speech_tokens = [self.hp.start_speech_token] + s["speech_tokens"] + [self.hp.stop_speech_token]

        return {
            "text_tokens": torch.tensor(text_tokens, dtype=torch.long),
            "speech_tokens": torch.tensor(speech_tokens, dtype=torch.long),
            "text_token_len": len(text_tokens),
            "speech_token_len": len(speech_tokens),
            "speaker_id": s["speaker_id"],
        }


def make_collate_fn(speaker_conds):
    """
    Returns a collate function that pads sequences and stacks T3Cond per speaker.
    """
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    def collate_fn(batch):
        text_lens = [b["text_token_len"] for b in batch]
        speech_lens = [b["speech_token_len"] for b in batch]
        max_text = max(text_lens)
        max_speech = max(speech_lens)

        # Pad sequences (pad value 0)
        text_tokens = torch.zeros(len(batch), max_text, dtype=torch.long)
        speech_tokens = torch.zeros(len(batch), max_speech, dtype=torch.long)

        for i, b in enumerate(batch):
            text_tokens[i, : text_lens[i]] = b["text_tokens"]
            speech_tokens[i, : speech_lens[i]] = b["speech_tokens"]

        text_token_lens = torch.tensor(text_lens, dtype=torch.long)
        speech_token_lens = torch.tensor(speech_lens, dtype=torch.long)

        # Stack T3Cond per speaker
        speaker_embs = []
        prompt_tokens_list = []
        emotion_advs = []

        for b in batch:
            sid = b["speaker_id"]
            if sid not in speaker_conds:
                raise KeyError(
                    f"No conds.pt loaded for speaker_id={sid}. "
                    f"Available: {list(speaker_conds.keys())}. "
                    f"Run extract_conds.py first."
                )
            cond = speaker_conds[sid]
            speaker_embs.append(cond.speaker_emb.squeeze(0))              # (256,)
            prompt_tokens_list.append(
                cond.cond_prompt_speech_tokens.squeeze(0)                  # (prompt_len,)
            )
            emotion_advs.append(cond.emotion_adv.flatten()[:1])            # (1,)

        t3_cond = T3Cond(
            speaker_emb=torch.stack(speaker_embs),                         # (B, 256)
            cond_prompt_speech_tokens=torch.stack(prompt_tokens_list),      # (B, prompt_len)
            emotion_adv=torch.stack(emotion_advs).unsqueeze(-1),           # (B, 1, 1)
        )

        return {
            "text_tokens": text_tokens,
            "text_token_lens": text_token_lens,
            "speech_tokens": speech_tokens,
            "speech_token_lens": speech_token_lens,
            "t3_cond": t3_cond,
        }

    return collate_fn


# ─────────────────────────────────────────────
# Speaker conditioning
# ─────────────────────────────────────────────
def load_speaker_conds(conds_dir, device="cpu"):
    """Load pre-extracted T3Cond from conds.pt files."""
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    conds = {}
    conds_dir = Path(conds_dir)

    for speaker_id, name in SPEAKER_CONDS_MAP.items():
        pt_path = conds_dir / f"{name}.pt"
        if pt_path.exists():
            data = torch.load(pt_path, map_location=device, weights_only=True)
            t3_cond = T3Cond(**data["t3"])
            conds[speaker_id] = t3_cond
            logger.info(f"Loaded conds for speaker {speaker_id} ({name})")

    if not conds:
        raise RuntimeError(f"No conds.pt files found in {conds_dir}. Run extract_conds.py first.")

    return conds


# ─────────────────────────────────────────────
# Loss computation
# ─────────────────────────────────────────────
def compute_loss(model, batch, device):
    """
    Compute next-token prediction loss with proper autoregressive shifting.

    Causal transformer: hidden[i] attends to tokens 0..i → predicts token i+1.
    Loss = CE(logits[:, :-1], targets[:, 1:])

    NOTE: Uses model() not model.forward() — required for DDP gradient sync hooks.
    """

    t3_cond = batch["t3_cond"].to(device=device)
    text_tokens = batch["text_tokens"].to(device)
    text_token_lens = batch["text_token_lens"].to(device)
    speech_tokens = batch["speech_tokens"].to(device)
    speech_token_lens = batch["speech_token_lens"].to(device)

    # Forward pass — use __call__ (not .forward()) for DDP compatibility
    out = model(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        text_token_lens=text_token_lens,
        speech_tokens=speech_tokens,
        speech_token_lens=speech_token_lens,
        training=True,
    )

    # ─── Speech loss (primary) ───
    # Shift: logits[i] predicts token[i+1]
    speech_logits = out.speech_logits[:, :-1].contiguous()   # (B, S-1, vocab)
    speech_targets = speech_tokens[:, 1:].contiguous()       # (B, S-1)

    # Mask padding in shifted targets
    shifted_speech_lens = speech_token_lens - 1
    seq_range = torch.arange(speech_targets.size(1), device=device)
    mask = seq_range[None] >= shifted_speech_lens[:, None]
    speech_targets = speech_targets.masked_fill(mask, -100)

    loss_speech = F.cross_entropy(
        speech_logits.reshape(-1, speech_logits.size(-1)),   # (B*(S-1), vocab)
        speech_targets.reshape(-1),                           # (B*(S-1),)
        ignore_index=-100,
    )

    # ─── Text loss (auxiliary) ───
    text_logits = out.text_logits[:, :-1].contiguous()
    text_targets = text_tokens[:, 1:].contiguous()

    shifted_text_lens = text_token_lens - 1
    seq_range_t = torch.arange(text_targets.size(1), device=device)
    mask_t = seq_range_t[None] >= shifted_text_lens[:, None]
    text_targets = text_targets.masked_fill(mask_t, -100)

    loss_text = F.cross_entropy(
        text_logits.reshape(-1, text_logits.size(-1)),
        text_targets.reshape(-1),
        ignore_index=-100,
    )

    return loss_speech, loss_text


# ─────────────────────────────────────────────
# Model setup
# ─────────────────────────────────────────────
def load_t3_model(device, lora_rank=32, lora_alpha=64, extended_model=None):
    """
    Load T3, apply LoRA to Llama backbone, freeze everything else.

    When extended_model is provided (from extend_t3_embeddings.py):
      - Loads extended vocab model instead of downloading from HuggingFace
      - Carries forward LoRA weights from previous training round
      - Unfreezes new embedding rows with gradient masking (only new rows train)

    Returns: (t3, config, original_vocab_size)
      original_vocab_size is non-None only when using extended model.
    """
    from peft import LoraConfig, get_peft_model
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config

    original_vocab_size = None
    carried_lora = None

    if extended_model:
        # ─── Load from extended checkpoint (Phase 2 output) ───
        logger.info(f"Loading extended T3 from {extended_model}")
        ext_ckpt = torch.load(extended_model, map_location=device, weights_only=False)

        config = T3Config.multilingual()
        config.text_tokens_dict_size = ext_ckpt["config"]["text_tokens_dict_size"]

        t3 = T3(hp=config)
        t3.load_state_dict(ext_ckpt["model_state_dict"])
        t3.to(device)

        # hf_base_vocab_size = rows to freeze (always 2454, the HF original)
        # original_vocab_size = what the extension started from (may differ for incremental)
        # For gradient masking, we use hf_base_vocab_size so all extended language
        # embeddings (Telugu, Kannada, etc.) remain trainable.
        original_vocab_size = ext_ckpt["config"].get(
            "hf_base_vocab_size",
            ext_ckpt["config"]["original_vocab_size"],  # backward compat for old configs
        )
        new_vocab_size = ext_ckpt["config"]["text_tokens_dict_size"]
        carried_lora = ext_ckpt.get("lora_state_dict")

        logger.info(
            f"Extended T3 loaded: vocab {original_vocab_size} → {new_vocab_size} "
            f"(+{new_vocab_size - original_vocab_size} new tokens, "
            f"freeze boundary={original_vocab_size})"
        )
        if carried_lora:
            logger.info(f"Found {len(carried_lora)} LoRA keys from previous round")
    else:
        # ─── Load from HuggingFace (original vocab) ───
        from safetensors.torch import load_file as load_safetensors
        from huggingface_hub import snapshot_download

        logger.info("Downloading T3 weights from HuggingFace...")
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

        logger.info("Loading T3 state dict...")
        state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
        if "model" in state:
            state = state["model"][0]
        t3.load_state_dict(state)
        t3.to(device)

    # Step 1: Freeze EVERYTHING
    for param in t3.parameters():
        param.requires_grad = False

    # Step 2: Apply LoRA to Llama backbone only
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    t3.tfmr = get_peft_model(t3.tfmr, lora_config)

    # Step 3: Carry forward LoRA weights from previous round
    if carried_lora:
        model_state = t3.tfmr.state_dict()
        loaded = 0
        for k, v in carried_lora.items():
            if k in model_state:
                model_state[k] = v.to(device)
                loaded += 1
        t3.tfmr.load_state_dict(model_state)
        logger.info(f"Carried forward {loaded}/{len(carried_lora)} LoRA weights from previous round")

    # Step 4: Unfreeze new embedding rows with gradient masking
    if original_vocab_size is not None:
        t3.text_emb.weight.requires_grad = True
        t3.text_head.weight.requires_grad = True

        orig_vs = original_vocab_size  # capture for closure

        def _emb_grad_mask(grad):
            """Zero gradients for original vocab rows, keep only new rows trainable."""
            grad = grad.clone()
            grad[:orig_vs] = 0.0
            return grad

        t3.text_emb.weight.register_hook(_emb_grad_mask)
        t3.text_head.weight.register_hook(_emb_grad_mask)

        n_new = t3.text_emb.weight.shape[0] - orig_vs
        hidden = t3.text_emb.weight.shape[1]
        logger.info(
            f"Unfroze new embedding rows [{orig_vs}:{t3.text_emb.weight.shape[0]}] "
            f"({n_new * hidden * 2:,} params in text_emb + text_head, gradient-masked)"
        )

    # Print parameter counts
    trainable = sum(p.numel() for p in t3.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in t3.parameters() if not p.requires_grad)
    total = trainable + frozen
    logger.info(
        f"Parameters — Trainable: {trainable:,}  Frozen: {frozen:,}  "
        f"Total: {total:,}  ({100*trainable/total:.1f}%)"
    )
    t3.tfmr.print_trainable_parameters()

    return t3, config, original_vocab_size


def get_raw_model(model):
    """Unwrap DDP / DataParallel to get the base T3 model."""
    return model.module if hasattr(model, "module") else model


def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, output_dir, name="checkpoint",
                    original_vocab_size=None):
    """Save LoRA weights + optional extended embeddings + optimizer state."""
    raw = get_raw_model(model)
    ckpt = {
        "lora_state_dict": {
            k: v.cpu() for k, v in raw.tfmr.state_dict().items() if "lora_" in k
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "loss": loss,
    }

    # Save extended embedding weights if applicable
    if original_vocab_size is not None:
        ckpt["text_emb_weight"] = raw.text_emb.weight.data.cpu()
        ckpt["text_head_weight"] = raw.text_head.weight.data.cpu()
        ckpt["original_vocab_size"] = original_vocab_size
        ckpt["new_vocab_size"] = raw.text_emb.weight.shape[0]

    path = Path(output_dir) / f"{name}.pt"
    torch.save(ckpt, path)
    logger.info(f"Saved checkpoint → {path}")
    return path


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Resume from a checkpoint (LoRA + optional extended embeddings)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    raw = get_raw_model(model)
    model_state = raw.tfmr.state_dict()
    for k, v in ckpt["lora_state_dict"].items():
        if k in model_state:
            model_state[k] = v.to(device)
    raw.tfmr.load_state_dict(model_state)

    # Restore extended embeddings if present in checkpoint
    if "text_emb_weight" in ckpt:
        raw.text_emb.weight.data = ckpt["text_emb_weight"].to(device)
        raw.text_head.weight.data = ckpt["text_head_weight"].to(device)
        logger.info(f"Restored extended embeddings (vocab={ckpt.get('new_vocab_size', '?')})")

    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])

    logger.info(f"Resumed from {checkpoint_path} (epoch={ckpt['epoch']}, step={ckpt['step']})")
    return ckpt["epoch"], ckpt["step"]


def warm_start_from_checkpoint(model, checkpoint_path, device):
    """
    Initialize model weights from a previous round's checkpoint WITHOUT
    restoring optimizer/scheduler state. Use this when starting a new
    training run (Round 3, etc.) with different hyperparameters but
    continuing from previously trained weights.

    Differs from --resume:
      --resume:     loads weights + optimizer + scheduler, continues from saved epoch
      --warm_start: loads weights only, fresh optimizer/scheduler, starts from epoch 0
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    raw = get_raw_model(model)

    # Load LoRA weights
    model_state = raw.tfmr.state_dict()
    loaded = 0
    for k, v in ckpt["lora_state_dict"].items():
        if k in model_state:
            model_state[k] = v.to(device)
            loaded += 1
    raw.tfmr.load_state_dict(model_state)

    # Load extended embeddings if present
    if "text_emb_weight" in ckpt:
        raw.text_emb.weight.data = ckpt["text_emb_weight"].to(device)
        raw.text_head.weight.data = ckpt["text_head_weight"].to(device)

    logger.info(
        f"Warm-started from {checkpoint_path} — "
        f"loaded {loaded} LoRA keys + embeddings "
        f"(from epoch={ckpt.get('epoch', '?')}, step={ckpt.get('step', '?')}, "
        f"loss={ckpt.get('loss', 0):.4f})"
    )


# ─────────────────────────────────────────────
# Learning rate schedule with warmup
# ─────────────────────────────────────────────
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = step / max(self.warmup_steps, 1)
        else:
            progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            scale = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1 + math.cos(math.pi * progress))
        return [base_lr * scale for base_lr in self.base_lrs]


# ─────────────────────────────────────────────
# Weighted multi-language sampler
# ─────────────────────────────────────────────
class WeightedLangSampler(torch.utils.data.Sampler):
    """
    Weighted random sampler for multi-language data mixing.

    Adjusts per-sample probability so each language appears at its target ratio,
    regardless of how many samples each language has in the dataset.

    Example:
        lang_weights = {"te": 0.5, "hi": 0.3, "en": 0.2}
        → Telugu samples drawn 50% of the time, Hindi 30%, English 20%
    """

    def __init__(self, dataset, lang_weights, num_samples=None, seed=42):
        self.num_samples = num_samples or len(dataset)
        self.seed = seed
        self._epoch = 0

        # Group sample indices by language
        lang_indices = defaultdict(list)
        for i, s in enumerate(dataset.samples):
            lang_indices[s.get("lang", "unknown")].append(i)

        # Compute per-sample weight = target_weight / count_for_that_lang
        self.weights = torch.zeros(len(dataset), dtype=torch.float64)
        for lang, indices in lang_indices.items():
            target_w = lang_weights.get(lang, 0.0)
            if target_w > 0 and len(indices) > 0:
                per_sample = target_w / len(indices)
                for idx in indices:
                    self.weights[idx] = per_sample

        # Warn about languages in data but not in weights
        for lang in lang_indices:
            if lang not in lang_weights:
                logger.warning(
                    f"Language '{lang}' has {len(lang_indices[lang])} samples "
                    f"but no weight specified — these samples will NOT be used"
                )

        # Warn about languages in weights but not in data
        for lang in lang_weights:
            if lang not in lang_indices:
                logger.warning(f"Weight specified for '{lang}' but no samples found in dataset")

        # Log effective ratios
        total_w = self.weights.sum().item()
        if total_w > 0:
            self.weights /= total_w  # normalize to sum=1
            for lang, indices in sorted(lang_indices.items()):
                lang_total = sum(self.weights[i].item() for i in indices)
                logger.info(f"  Lang '{lang}': {len(indices)} samples, effective ratio={lang_total:.1%}")

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self._epoch)
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g).tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        """For DDP compatibility and epoch-varying randomness."""
        self._epoch = epoch


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────
@torch.no_grad()
def validate(model, val_loader, device):
    """
    Compute mean speech & text loss on validation set.
    Uses the raw (unwrapped) model to avoid DDP collective ops during eval.
    """
    raw = get_raw_model(model)
    raw.eval()
    total_speech = 0.0
    total_text = 0.0
    n = 0

    for batch in val_loader:
        loss_speech, loss_text = compute_loss(raw, batch, device)
        total_speech += loss_speech.item()
        total_text += loss_text.item()
        n += 1

    raw.train()
    return total_speech / max(n, 1), total_text / max(n, 1)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="T3 LoRA fine-tuning for Indian languages",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--train_data", nargs="+", required=True, help="Training JSON file(s)")
    parser.add_argument("--val_data", nargs="+", required=True, help="Validation JSON file(s)")
    parser.add_argument("--conds_dir", required=True, help="Directory with speaker conds.pt files")

    # Output
    parser.add_argument("--output_dir", default="checkpoints/round1", help="Checkpoint output directory")

    # Model
    parser.add_argument("--extended_model", type=str, default=None,
                        help="Path to extended T3 model .pt (from extend_t3_embeddings.py). "
                             "When provided, loads extended vocab + carries forward LoRA from previous round.")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha (scaling)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--warm_start", type=str, default=None,
                        help="Initialize weights from a previous round's checkpoint (LoRA + embeddings) "
                             "without restoring optimizer/scheduler. For continuing with new hyperparams.")

    # Training hyperparameters
    parser.add_argument("--device", default="cuda", help="Device for single-GPU (ignored under torchrun)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size PER GPU")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup fraction of total steps")
    parser.add_argument("--text_loss_weight", type=float, default=0.1, help="Weight for auxiliary text loss")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 mixed precision (Blackwell/Ampere+)")

    # Multi-language mixing
    parser.add_argument("--lang_weights", nargs="+", default=None,
                        help="Language sampling weights as lang:weight pairs, e.g. te:0.5 hi:0.3 en:0.2. "
                             "Languages not listed get weight 0 (not sampled).")

    # Data filtering
    parser.add_argument("--max_text_tokens", type=int, default=400, help="Skip samples with more text tokens")
    parser.add_argument("--max_speech_tokens", type=int, default=1000, help="Skip samples with more speech tokens")

    # Logging & eval
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--val_every", type=int, default=200, help="Validate every N steps")
    parser.add_argument("--save_every_epoch", action="store_true", help="Save checkpoint after every epoch")
    parser.add_argument("--max_steps", type=int, default=None, help="Stop after N steps (for debugging)")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # Wandb
    parser.add_argument("--use_wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", default="indic-tts", help="Wandb project name")
    parser.add_argument("--wandb_run_name", default=None, help="Wandb run name")

    args = parser.parse_args()

    # ─── Distributed setup ───
    rank, world_size, local_rank, is_ddp = setup_distributed()

    if is_ddp:
        device = f"cuda:{local_rank}"
    else:
        device = args.device

    # ─── Logging (only rank 0 writes to file / stdout) ───
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    handlers = []
    if is_main(rank):
        handlers.append(logging.StreamHandler())
        handlers.append(logging.FileHandler(Path(args.output_dir) / "train.log", mode="a"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)

    if is_main(rank):
        with open(Path(args.output_dir) / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        logger.info(f"Arguments: {vars(args)}")
        logger.info(f"DDP: {is_ddp}  World size: {world_size}  Rank: {rank}  Device: {device}")
        eff_batch = args.batch_size * world_size * args.grad_accum
        logger.info(f"Effective batch size: {args.batch_size} × {world_size} GPUs × {args.grad_accum} accum = {eff_batch}")

    # ─── Wandb (rank 0 only) ───
    if args.use_wandb and is_main(rank):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # ─── Parse language weights ───
    lang_weights = None
    if args.lang_weights:
        lang_weights = {}
        for lw in args.lang_weights:
            lang, w = lw.split(":")
            lang_weights[lang] = float(w)
        if is_main(rank):
            logger.info(f"Language mixing weights: {lang_weights}")

    # ─── Load model ───
    t3, config, original_vocab_size = load_t3_model(
        device, args.lora_rank, args.lora_alpha,
        extended_model=args.extended_model,
    )

    # Freeze non-LoRA parts explicitly
    t3.cond_enc.eval()
    t3.speech_emb.requires_grad_(False)
    t3.speech_head.requires_grad_(False)
    if t3.text_pos_emb is not None:
        t3.text_pos_emb.requires_grad_(False)
    if t3.speech_pos_emb is not None:
        t3.speech_pos_emb.requires_grad_(False)

    # Only freeze text_emb/text_head when NOT using extended model
    # (extended model has gradient masking set up in load_t3_model)
    if original_vocab_size is None:
        t3.text_emb.requires_grad_(False)
        t3.text_head.requires_grad_(False)

    # ─── Warm-start from previous round (weights only, fresh optimizer) ───
    if args.warm_start:
        warm_start_from_checkpoint(t3, args.warm_start, device)

    # Wrap with DDP
    if is_ddp:
        t3 = DDP(t3, device_ids=[local_rank], find_unused_parameters=False)
        if is_main(rank):
            logger.info(f"Model wrapped with DDP on {world_size} GPUs")

    # ─── Load speaker conds ───
    speaker_conds = load_speaker_conds(args.conds_dir, device="cpu")

    # ─── Datasets ───
    train_dataset = T3Dataset(
        args.train_data, config,
        max_text_tokens=args.max_text_tokens,
        max_speech_tokens=args.max_speech_tokens,
    )
    val_dataset = T3Dataset(
        args.val_data, config,
        max_text_tokens=args.max_text_tokens,
        max_speech_tokens=args.max_speech_tokens,
    )

    if is_main(rank):
        train_dataset.log_stats()
        val_dataset.log_stats()

    collate = make_collate_fn(speaker_conds)

    # Sampler: DDP > weighted lang mixing > default shuffle
    if is_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    elif lang_weights:
        train_sampler = WeightedLangSampler(train_dataset, lang_weights)
        if is_main(rank):
            logger.info(f"Using WeightedLangSampler with {len(lang_weights)} languages")
    else:
        train_sampler = None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    # ─── Optimizer ───
    trainable_params = [p for p in t3.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # ─── Scheduler ───
    steps_per_epoch = max(len(train_loader) // args.grad_accum, 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    if is_main(rank):
        logger.info(f"Steps/epoch: {steps_per_epoch}  Total: {total_steps}  Warmup: {warmup_steps}")

    # ─── Mixed precision ───
    amp_dtype = None
    scaler = None
    if args.bf16:
        amp_dtype = torch.bfloat16
        if is_main(rank):
            logger.info("Using bf16 mixed precision (recommended for Blackwell)")
    elif args.fp16:
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda")
        if is_main(rank):
            logger.info("Using fp16 mixed precision")

    # ─── Resume ───
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step = load_checkpoint(t3, optimizer, scheduler, args.resume, device)

    # ═══════════════════════════════════════════
    # Training loop
    # ═══════════════════════════════════════════
    best_val_loss = float("inf")
    t3.train()

    if is_main(rank):
        logger.info("=" * 60)
        logger.info("TRAINING START")
        logger.info("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        # DDP: set epoch for sampler so each epoch shuffles differently
        if train_sampler:
            train_sampler.set_epoch(epoch)

        epoch_loss_speech = 0.0
        epoch_loss_text = 0.0
        epoch_batches = 0
        t0 = time.time()

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}",
                        disable=not is_main(rank), leave=True)

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress):

            # ─── Forward + Loss ───
            if amp_dtype:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    loss_speech, loss_text = compute_loss(t3, batch, device)
                    loss = (loss_speech + args.text_loss_weight * loss_text) / args.grad_accum
            else:
                loss_speech, loss_text = compute_loss(t3, batch, device)
                loss = (loss_speech + args.text_loss_weight * loss_text) / args.grad_accum

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss_speech += loss_speech.item()
            epoch_loss_text += loss_text.item()
            epoch_batches += 1

            # ─── Gradient accumulation step ───
            if (batch_idx + 1) % args.grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # ─── Logging ───
                if is_main(rank) and global_step % args.log_every == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    progress.set_postfix({
                        "speech": f"{loss_speech.item():.4f}",
                        "text": f"{loss_text.item():.4f}",
                        "lr": f"{lr:.2e}",
                        "step": global_step,
                    })

                    if args.use_wandb:
                        import wandb
                        wandb.log({
                            "train/loss_speech": loss_speech.item(),
                            "train/loss_text": loss_text.item(),
                            "train/loss_total": (loss_speech + args.text_loss_weight * loss_text).item(),
                            "train/lr": lr,
                            "train/epoch": epoch + (batch_idx / len(train_loader)),
                        }, step=global_step)

                # ─── Periodic validation (ALL ranks pause to stay in lockstep) ───
                if global_step % args.val_every == 0 and global_step > 0:
                    val_speech, val_text = validate(t3, val_loader, device)

                    if is_main(rank):
                        ppl = math.exp(min(val_speech, 20))
                        logger.info(
                            f"  [VAL] step={global_step}  speech_loss={val_speech:.4f}  "
                            f"text_loss={val_text:.4f}  ppl_speech={ppl:.1f}"
                        )
                        if args.use_wandb:
                            import wandb
                            wandb.log({
                                "val/loss_speech": val_speech,
                                "val/loss_text": val_text,
                                "val/ppl_speech": ppl,
                            }, step=global_step)

                        if val_speech < best_val_loss:
                            best_val_loss = val_speech
                            save_checkpoint(t3, optimizer, scheduler, epoch, global_step,
                                            val_speech, args.output_dir, "best",
                                            original_vocab_size=original_vocab_size)
                            logger.info(f"  ★ New best val loss: {val_speech:.4f}")

                    get_raw_model(t3).train()

                # ─── Max steps ───
                if args.max_steps and global_step >= args.max_steps:
                    if is_main(rank):
                        logger.info(f"Reached max_steps={args.max_steps}, stopping.")
                    break

        # ─── Epoch summary (logging on rank 0) ───
        if is_main(rank):
            elapsed = time.time() - t0
            avg_speech = epoch_loss_speech / max(epoch_batches, 1)
            avg_text = epoch_loss_text / max(epoch_batches, 1)
            ppl = math.exp(min(avg_speech, 20))

            logger.info(
                f"Epoch {epoch+1}/{args.epochs} done in {elapsed:.0f}s — "
                f"avg speech_loss={avg_speech:.4f}  avg text_loss={avg_text:.4f}  "
                f"ppl_speech={ppl:.1f}"
            )

            if args.save_every_epoch:
                save_checkpoint(t3, optimizer, scheduler, epoch + 1, global_step,
                                avg_speech, args.output_dir, f"epoch{epoch+1}",
                                original_vocab_size=original_vocab_size)

        # End-of-epoch validation — ALL ranks participate (prevents DDP deadlock)
        val_speech, val_text = validate(t3, val_loader, device)

        if is_main(rank):
            logger.info(
                f"  [VAL end-of-epoch] speech_loss={val_speech:.4f}  text_loss={val_text:.4f}"
            )
            if val_speech < best_val_loss:
                best_val_loss = val_speech
                save_checkpoint(t3, optimizer, scheduler, epoch + 1, global_step,
                                val_speech, args.output_dir, "best",
                                original_vocab_size=original_vocab_size)
                logger.info(f"  ★ New best val loss: {val_speech:.4f}")

        # Barrier to sync all ranks before next epoch
        if is_ddp:
            dist.barrier()

        if args.max_steps and global_step >= args.max_steps:
            break

    # ─── Final ───
    if is_main(rank):
        save_checkpoint(t3, optimizer, scheduler, args.epochs, global_step,
                        best_val_loss, args.output_dir, "final",
                        original_vocab_size=original_vocab_size)

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"  Best val speech loss: {best_val_loss:.4f}")
        logger.info(f"  Total steps: {global_step}")
        logger.info(f"  Checkpoints: {args.output_dir}")
        logger.info("=" * 60)

        if args.use_wandb:
            import wandb
            wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
