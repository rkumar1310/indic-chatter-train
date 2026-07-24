#!/usr/bin/env python3
"""
example_inference.py — Load a checkpoint produced by speedrun.sh / train_t3_lora.py
and generate speech, without any of the packaging from the inference-only fork.

This is the manual load path: swap in the extended tokenizer, resize the
embeddings from the checkpoint, merge the LoRA state dict into the transformer.

Usage:
  python example_inference.py --lang te \
    --checkpoint checkpoints/te_speedrun/best.pt \
    --tokenizer data/tokenizer/extended_tokenizer.json \
    --speaker conds/te_female.pt \
    --text "నమస్కారం, మీరు ఎలా ఉన్నారు?"
"""

import argparse

import soundfile as sf
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals
from chatterbox.models.tokenizers import MTLTokenizer


def main():
    parser = argparse.ArgumentParser(description="Run inference with a speedrun.sh checkpoint")
    parser.add_argument("--lang", required=True, help="Language ID, e.g. te, kn, bn")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt / final.pt")
    parser.add_argument("--tokenizer", required=True, help="Path to extended_tokenizer.json")
    parser.add_argument("--speaker", required=True, help="Path to a conds/*.pt speaker file")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", default="output.wav", help="Output WAV path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")

    print("Loading base Chatterbox-Multilingual...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=args.device)

    print(f"Loading extended tokenizer: {args.tokenizer}")
    model.tokenizer = MTLTokenizer(args.tokenizer)

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    dim = model.t3.text_emb.weight.shape[1]

    if "text_emb_weight" in checkpoint:
        vocab_size = checkpoint.get("new_vocab_size", checkpoint["text_emb_weight"].shape[0])
        print(f"Resizing text_emb/text_head to vocab={vocab_size}")

        new_text_emb = torch.nn.Embedding(vocab_size, dim).to(args.device)
        new_text_emb.weight.data = checkpoint["text_emb_weight"].to(args.device)
        model.t3.text_emb = new_text_emb

        new_text_head = torch.nn.Linear(dim, vocab_size, bias=False).to(args.device)
        new_text_head.weight.data = checkpoint["text_head_weight"].to(args.device)
        model.t3.text_head = new_text_head

    if "lora_state_dict" in checkpoint:
        model_state = model.t3.tfmr.state_dict()
        n_merged = 0
        for k, v in checkpoint["lora_state_dict"].items():
            if k in model_state:
                model_state[k] = v.to(args.device)
                n_merged += 1
        model.t3.tfmr.load_state_dict(model_state)
        print(f"Merged {n_merged} LoRA weight tensors")

    print(f"Loading speaker: {args.speaker}")
    model.conds = Conditionals.load(args.speaker, map_location=args.device).to(args.device)

    print("Generating...")
    wav = model.generate(args.text, language_id=args.lang)
    sf.write(args.output, wav.squeeze(0).cpu().numpy(), model.sr)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
