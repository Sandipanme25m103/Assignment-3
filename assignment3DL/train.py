from __future__ import annotations
import argparse
import copy
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import wandb

from dataset import PAD_IDX, get_dataloaders
from lr_scheduler import NoamScheduler
from model import (
    LabelSmoothingLoss,
    Transformer,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)
    print(f"[train] Checkpoint saved → {path}")


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> None:
    model.load_state_dict(torch.load(path, map_location=device))

def train_epoch(
    model:      nn.Module,
    loader,
    criterion:  nn.Module,
    optimizer:  torch.optim.Optimizer,
    scheduler:  Optional[NoamScheduler],
    device:     torch.device,
    clip:       float = 1.0,
    log_grad_norms: bool = False,
    use_fixed_lr: bool = False,
) -> Tuple[float, List[float]]:
    model.train()
    total_loss = 0.0
    lr_history: List[float] = []
    grad_norm_log: List[float] = []

    for src, tgt in loader:
        src = src.to(device)                
        tgt = tgt.to(device)                 

        tgt_input  = tgt[:, :-1]            
        tgt_output = tgt[:, 1:]             

        logits = model(src, tgt_input)       
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), tgt_output.reshape(B * T))

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)

        if log_grad_norms:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            grad_norm_log.append(math.sqrt(total_norm))

        optimizer.step()

        if scheduler is not None and not use_fixed_lr:
            lr = scheduler.step()
            lr_history.append(lr)

    avg_loss = total_loss / max(len(loader), 1)

    if log_grad_norms and grad_norm_log:
        wandb.log({"grad_norm": np.mean(grad_norm_log)})

    return avg_loss, lr_history


@torch.no_grad()
def evaluate(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total   = 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input  = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        logits = model(src, tgt_input)       
        B, T, V = logits.shape

        loss = criterion(logits.reshape(B * T, V), tgt_output.reshape(B * T))
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)        
        mask  = tgt_output != PAD_IDX
        correct += (preds[mask] == tgt_output[mask]).sum().item()
        total   += mask.sum().item()

    return total_loss / max(len(loader), 1), correct / max(total, 1)

@torch.no_grad()
def compute_corpus_bleu(
    model:      nn.Module,
    loader,
    tgt_vocab:  dict,
    device:     torch.device,
    max_len:    int = 100,
    max_batches: int = 20,
) -> float:
    from evaluate import load as eval_load      
    bleu_metric = eval_load("bleu")

    tgt_itos = {v: k for k, v in tgt_vocab.items()}
    sos_idx  = tgt_vocab["<sos>"]
    eos_idx  = tgt_vocab["<eos>"]
    pad_idx  = tgt_vocab.get("<pad>", PAD_IDX)

    hypotheses: List[str]        = []
    references:  List[List[str]] = []

    model.eval()
    for i, (src, tgt) in enumerate(loader):
        if i >= max_batches:
            break
        src = src.to(device)
        tgt = tgt.to(device)

        src_mask = model.make_src_mask(src)
        enc_out  = model.encoder(src, src_mask)

        B = src.size(0)
        tgt_ids = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            tgt_mask = model.make_tgt_mask(tgt_ids)
            dec_out  = model.decoder(tgt_ids, enc_out, src_mask, tgt_mask)
            logits   = model.fc_out(dec_out[:, -1, :])            
            next_ids = logits.argmax(dim=-1, keepdim=True)        
            tgt_ids  = torch.cat([tgt_ids, next_ids], dim=1)
            finished |= (next_ids.squeeze(1) == eos_idx)
            if finished.all():
                break

        for b in range(B):
            hyp_tokens, ref_tokens = [], []
            for idx in tgt_ids[b, 1:].tolist():
                if idx == eos_idx:
                    break
                tok = tgt_itos.get(idx, "")
                if tok and tok not in ("<pad>", "<unk>", "<sos>", "<eos>"):
                    hyp_tokens.append(tok)
            for idx in tgt[b, 1:].tolist():
                if idx == eos_idx:
                    break
                tok = tgt_itos.get(idx, "")
                if tok and tok not in ("<pad>", "<unk>", "<sos>", "<eos>"):
                    ref_tokens.append(tok)

            hypotheses.append(" ".join(hyp_tokens))
            references.append([" ".join(ref_tokens)])

    result = bleu_metric.compute(predictions=hypotheses, references=references)
    return result.get("bleu", 0.0)

@torch.no_grad()
def log_attention_maps(
    model:      nn.Module,
    src:        torch.Tensor,
    src_vocab:  dict,
    device:     torch.device,
    step:       int = 0,
) -> None:
    import matplotlib.pyplot as plt

    model.eval()
    src = src[:1].to(device)                               
    src_mask = model.make_src_mask(src)
    attn_weights_store: List[torch.Tensor] = []

    def hook(module, inp, out):
        if isinstance(out, tuple) and len(out) == 2:
            attn_weights_store.append(out[1].detach().cpu())

    last_encoder = model.encoder.layers[-1]
    handle = last_encoder.self_attn.attention.register_forward_hook(
        lambda m, i, o: attn_weights_store.append(o[1].detach().cpu())
    )

    _ = model.encoder(src, src_mask)
    handle.remove()

    if not attn_weights_store:
        return

    attn = attn_weights_store[0].squeeze(0)             
    num_heads = attn.shape[0]
    src_ids   = src[0].cpu().tolist()
    src_itos  = {v: k for k, v in src_vocab.items()}
    tokens    = [src_itos.get(i, "<unk>") for i in src_ids]

    fig, axes = plt.subplots(2, num_heads // 2, figsize=(4 * num_heads // 2, 8))
    axes = axes.flatten()
    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(attn[h].numpy(), cmap="Blues", aspect="auto")
        ax.set_title(f"Head {h + 1}")
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=7)
        plt.colorbar(im, ax=ax)

    plt.suptitle("Last Encoder Layer – Multi-Head Attention Maps", fontsize=13)
    plt.tight_layout()
    wandb.log({"attention_maps": wandb.Image(fig), "step": step})
    plt.close(fig)

def build_model(cfg: dict, src_vocab_size: int, tgt_vocab_size: int) -> nn.Module:
    from model import (
    NoamScheduler,
        Decoder, Encoder, FeedForward, MultiHeadAttention,
        PositionalEncoding, ScaledDotProductAttention, Transformer,
    )

    model = nn.Module.__new__(Transformer)
    nn.Module.__init__(model)

    model.src_pad_idx = PAD_IDX
    model.tgt_pad_idx = PAD_IDX
    model.device      = cfg["device"]

    model.encoder = Encoder(
        src_vocab_size, cfg["d_model"], cfg["num_heads"],
        cfg["num_encoder_layers"], cfg["d_ff"],
        cfg["max_seq_len"], cfg["dropout"],
        pad_idx=PAD_IDX,
        use_learned_pe=cfg.get("use_learned_pe", False),
        scale=cfg.get("scale", True),
    )
    model.decoder = Decoder(
        tgt_vocab_size, cfg["d_model"], cfg["num_heads"],
        cfg["num_decoder_layers"], cfg["d_ff"],
        cfg["max_seq_len"], cfg["dropout"],
        pad_idx=PAD_IDX,
        use_learned_pe=cfg.get("use_learned_pe", False),
        scale=cfg.get("scale", True),
    )
    model.fc_out = nn.Linear(cfg["d_model"], tgt_vocab_size)
    from model import Transformer as T
    model.make_src_mask = T.make_src_mask.__get__(model, type(model))
    model.make_tgt_mask = T.make_tgt_mask.__get__(model, type(model))
    model.forward       = T.forward.__get__(model, type(model))
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model.to(cfg["device"])


def run_training(cfg: dict, use_wandb: bool = False) -> nn.Module:
    device = cfg["device"]
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(
        batch_size=cfg["batch_size"],
        min_freq=cfg.get("min_freq", 2),
        max_len=cfg["max_seq_len"],
    )

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    model = build_model(cfg, src_vocab_size, tgt_vocab_size)
    print(f"[train] Model parameters: {count_parameters(model):,}")

    criterion = LabelSmoothingLoss(
        vocab_size=tgt_vocab_size,
        pad_idx=PAD_IDX,
        smoothing=cfg.get("label_smoothing", 0.1),
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9
    )

    use_noam = cfg.get("use_noam", True)
    scheduler: Optional[NoamScheduler] = None
    if use_noam:
        scheduler = NoamScheduler(optimizer, cfg["d_model"], cfg.get("warmup_steps", 4000))
    else:
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.get("fixed_lr", 1e-4)

    if use_wandb:
        wandb.init(project="da6401-assignment3", config=cfg, name=cfg.get("run_name", "run"))
    best_val_loss = float("inf")
    best_model_state = None
    log_grad = cfg.get("log_grad_norms", False)
    global_step = 0

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        correct, total = 0, 0

        for batch_idx, (src, tgt) in enumerate(train_loader):
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(src, tgt_input)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B * T, V), tgt_output.reshape(B * T))

            optimizer.zero_grad()
            loss.backward()

            if log_grad and use_wandb and global_step < 1000:
                total_norm = sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in model.parameters()
                    if p.grad is not None
                ) ** 0.5
                wandb.log({"grad_norm": total_norm, "global_step": global_step})

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if scheduler is not None:
                lr = scheduler.step()
                if use_wandb:
                    wandb.log({"lr": lr, "global_step": global_step})

            epoch_loss += loss.item()

            preds = logits.argmax(dim=-1)
            mask  = tgt_output != PAD_IDX
            correct += (preds[mask] == tgt_output[mask]).sum().item()
            total   += mask.sum().item()

            global_step += 1

        train_loss = epoch_loss / len(train_loader)
        train_acc  = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f} | {elapsed:.1f}s"
        )

        if use_wandb:
            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }

            if cfg.get("log_confidence", False):
                log_dict["smoothing_eps"] = cfg.get("label_smoothing", 0.1)

            wandb.log(log_dict)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            save_checkpoint(model, "model_best.pt")
            print(f"  ✓ New best val loss: {val_loss:.4f}")

        if use_wandb and cfg.get("log_attention_maps", False) and epoch == 5:
            sample_src, _ = next(iter(val_loader))
            log_attention_maps(model, sample_src, src_vocab, device, step=global_step)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    bleu = compute_corpus_bleu(model, test_loader, tgt_vocab, device)
    print(f"\n[train] Test BLEU (approx): {bleu:.4f}")
    if use_wandb:
        wandb.log({"test_bleu": bleu})
        wandb.finish()

    return model

BASE_CFG = {
    "d_model":            256,
    "num_heads":          8,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "d_ff":               512,
    "max_seq_len":        256,
    "dropout":            0.1,
    "batch_size":         128,
    "epochs":             20,
    "warmup_steps":       4000,
    "label_smoothing":    0.1,
    "use_noam":           True,
    "use_learned_pe":     False,
    "scale":              True,
    "fixed_lr":           1e-4,
    "log_grad_norms":     False,
    "log_attention_maps": False,
    "log_confidence":     False,
}

EXPERIMENTS: Dict[str, dict] = {
    "baseline": {
        **BASE_CFG,
        "run_name": "baseline_noam_sinusoidal",
    },

    "noam": {
        **BASE_CFG,
        "use_noam": True,
        "run_name": "exp2.1_noam",
    },
    "fixed_lr": {
        **BASE_CFG,
        "use_noam": False,
        "fixed_lr": 1e-4,
        "run_name": "exp2.1_fixed_lr",
    },

    "with_scale": {
        **BASE_CFG,
        "scale": True,
        "log_grad_norms": True,
        "run_name": "exp2.2_with_scale",
    },
    "no_scale": {
        **BASE_CFG,
        "scale": False,
        "log_grad_norms": True,
        "run_name": "exp2.2_no_scale",
    },

    "attention_maps": {
        **BASE_CFG,
        "log_attention_maps": True,
        "run_name": "exp2.3_attention_maps",
    },

    "sinusoidal_pe": {
        **BASE_CFG,
        "use_learned_pe": False,
        "run_name": "exp2.4_sinusoidal_pe",
    },
    "learned_pe": {
        **BASE_CFG,
        "use_learned_pe": True,
        "run_name": "exp2.4_learned_pe",
    },

    "smooth_0.1": {
        **BASE_CFG,
        "label_smoothing": 0.1,
        "log_confidence": True,
        "run_name": "exp2.5_smooth_0.1",
    },
    "smooth_0.0": {
        **BASE_CFG,
        "label_smoothing": 0.0,
        "log_confidence": True,
        "run_name": "exp2.5_smooth_0.0",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DA6401 Assignment 3 – Transformer NMT")
    parser.add_argument(
        "--exp",
        type=str,
        default="baseline",
        choices=list(EXPERIMENTS.keys()),
        help="Experiment configuration to run.",
    )
    parser.add_argument("--wandb",   action="store_true", help="Enable W&B logging.")
    parser.add_argument("--epochs",  type=int, default=None, help="Override epoch count.")
    parser.add_argument("--device",  type=str, default=None, help="Device (cuda/cpu).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EXPERIMENTS[args.exp].copy()
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    cfg["device"] = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"\n{'=' * 60}")
    print(f"  Experiment : {args.exp}")
    print(f"  Device     : {cfg['device']}")
    print(f"  Epochs     : {cfg['epochs']}")
    print(f"{'=' * 60}\n")

    run_training(cfg, use_wandb=args.wandb)


if __name__ == "__main__":
    main()
