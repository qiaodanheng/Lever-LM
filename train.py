"""
train.py  –  Train PointerSelector (Lever-LM) with multi-beam RCE on VQAv2.

Usage:
    # Full training (multi-GPU)
    python train.py \
        --train_file data/vqav2_train_beams.json \
        --val_file   data/vqav2_val_beams.json   \
        --loss_mode  rce                          \
        --gpus 4 --batch_size 64

    # Ablation: single-target CE
    python train.py \
        --train_file data/vqav2_train_beams.json \
        --val_file   data/vqav2_val_beams.json   \
        --loss_mode  ce

    # Small-scale smoke test (CPU)
    python train.py \
        --train_file data/vqav2_train_beams.json \
        --val_file   data/vqav2_val_beams.json   \
        --max_epochs 2 --batch_size 8 --gpus 0
"""

import argparse
import os
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from lever_lm.lever_lm_module import LeverLMModule
from lever_lm.dataset import VQAv2BeamDataModule


# --------------------------------------------------------------------------- #
#  Argument parsing
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Lever-LM PointerSelector")

    # Data
    p.add_argument("--train_file", required=True)
    p.add_argument("--val_file",   required=True)
    p.add_argument("--test_file",  default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--max_beams",  type=int, default=5)
    p.add_argument("--shot_num",   type=int, default=2)
    p.add_argument("--K",          type=int, default=64)

    # Model
    p.add_argument("--d_model",    type=int,   default=768)
    p.add_argument("--hidden_dim", type=int,   default=256)
    p.add_argument("--num_heads",  type=int,   default=4)
    p.add_argument("--num_layers", type=int,   default=2)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--attn_dropout",type=float,default=0.1)
    p.add_argument("--label_smoothing", type=float, default=0.1)

    # Training
    p.add_argument("--loss_mode",  default="rce", choices=["rce", "ce"])
    p.add_argument("--neg_weight", type=float, default=0.0,
                   help="Weight of the hard-negative InfoNCE term (0 = off)")
    p.add_argument("--num_neg",    type=int,   default=16,
                   help="Random negative sequences per query for InfoNCE")
    p.add_argument("--pool_file",  default=None,
                   help="Pool json for genuinely-bad distractor negatives")
    p.add_argument("--num_distractor", type=int, default=64,
                   help="# pool distractor embeddings appended per query")
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float,default=1e-2)
    p.add_argument("--warmup_steps",type=int,  default=500)
    p.add_argument("--max_steps",  type=int,   default=10_000)
    p.add_argument("--max_epochs", type=int,   default=50)

    # Hardware
    p.add_argument("--gpus",       type=int,   default=1)
    p.add_argument("--precision",  default="bf16-mixed",
                   choices=["32", "16-mixed", "bf16-mixed"])

    # I/O
    p.add_argument("--ckpt_dir",   default="checkpoints")
    p.add_argument("--log_dir",    default="logs")
    p.add_argument("--run_name",   default="lever_lm_pointer_rce")
    p.add_argument("--resume_from",default=None,
                   help="Resume from a checkpoint path")
    p.add_argument("--seed",       type=int,   default=42)

    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    # ------------------------------------------------------------------ #
    # DataModule
    # ------------------------------------------------------------------ #
    dm = VQAv2BeamDataModule(
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_beams=args.max_beams,
        shot_num=args.shot_num,
        K=args.K,
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = LeverLMModule(
        d_model=args.d_model,
        K=args.K,
        shot_num=args.shot_num,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        label_smoothing=args.label_smoothing,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        loss_mode=args.loss_mode,
        neg_weight=args.neg_weight,
        num_neg=args.num_neg,
        pool_file=args.pool_file,
        num_distractor=args.num_distractor,
    )

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    ckpt_dir = Path(args.ckpt_dir) / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="epoch{epoch:03d}-valloss{val/loss:.4f}",
            auto_insert_metric_name=False,
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=10,
            mode="min",
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # ------------------------------------------------------------------ #
    # Logger
    # ------------------------------------------------------------------ #
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.run_name,
    )

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    accelerator = "gpu" if args.gpus > 0 else "cpu"
    devices = args.gpus if args.gpus > 0 else 1
    strategy = "ddp" if args.gpus > 1 else "auto"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=args.precision if args.gpus > 0 else "32",
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=0.25,       # validate 4× per epoch
        gradient_clip_val=1.0,
        deterministic=False,
    )

    trainer.fit(
        model,
        datamodule=dm,
        ckpt_path=args.resume_from,
    )

    if args.test_file:
        trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
