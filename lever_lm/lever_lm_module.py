"""
LeverLMModule: PyTorch Lightning module for training PointerSelector with
multi-beam reward-weighted cross-entropy (RCE) loss on VQAv2.

Key design decisions vs. the original Lever-LM repo:
- Model:       PointerSelector (cross-attention + pointer) replaces tiny Transformer.
- Loss:        Multi-target RCE over beam_size reward-ranked ICD sequences instead
               of single-target CE over the single best sequence.
- Temperature: Learned log_temperature parameter inside PointerSelector, so it
               is jointly optimised with the attention weights.
- Optimiser:   AdamW + cosine LR schedule with linear warmup.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.types import STEP_OUTPUT
    _HAS_PL = True
except ImportError:
    _HAS_PL = False

from .pointer_selector import PointerSelector


if _HAS_PL:
    class LeverLMModule(pl.LightningModule):
        """
        Lightning wrapper around PointerSelector for Lever-LM training.

        Hparams (all saved via save_hyperparameters):
            d_model:        Input embedding dimension.
            K:              Candidate pool size.
            shot_num:       Shots per ICD sequence.
            hidden_dim:     Pointer attention hidden dim.
            num_heads:      Cross-attention heads.
            num_layers:     Cross-attention depth.
            dropout:        Dropout rate.
            attn_dropout:   Attention dropout.
            label_smoothing: For fallback single-target CE.
            lr:             Peak learning rate.
            weight_decay:   AdamW weight decay.
            warmup_steps:   Linear warmup steps.
            max_steps:      Total training steps (for cosine schedule).
            loss_mode:      "rce" (multi-beam) | "ce" (single-target, ablation).
        """

        def __init__(
            self,
            d_model: int = 768,
            K: int = 64,
            shot_num: int = 2,
            hidden_dim: int = 256,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout: float = 0.1,
            attn_dropout: float = 0.1,
            label_smoothing: float = 0.1,
            lr: float = 1e-4,
            weight_decay: float = 1e-2,
            warmup_steps: int = 500,
            max_steps: int = 10_000,
            loss_mode: str = "rce",
            neg_weight: float = 0.0,
            num_neg: int = 16,
            pool_file: Optional[str] = None,
            num_distractor: int = 64,
        ):
            super().__init__()
            self.save_hyperparameters()

            # Optional pool of distractor embeddings for genuinely-bad negatives.
            self._pool_embs = None
            if pool_file:
                import json as _json
                with open(pool_file) as f:
                    pool = _json.load(f)
                self._pool_embs = torch.tensor(
                    [d["embedding"] for d in pool], dtype=torch.float32
                )  # [N, D]

            self.model = PointerSelector(
                d_model=d_model,
                K=K,
                shot_num=shot_num,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                attn_dropout=attn_dropout,
                label_smoothing=label_smoothing,
            )

        # ------------------------------------------------------------------
        # Shared step
        # ------------------------------------------------------------------

        def _step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
            query_emb = batch["query_emb"]      # [B, D]
            cand_embs = batch["cand_embs"]      # [B, K, D]
            beam_labels = batch["beam_labels"]  # [B, nb, ns]
            beam_rewards = batch["beam_rewards"]# [B, nb]
            beam_mask = batch["beam_mask"]      # [B, nb]

            if self.hparams.loss_mode == "rce":
                neg_embs = None
                if self._pool_embs is not None and self.hparams.neg_weight > 0.0:
                    B = query_emb.size(0)
                    P = self.hparams.num_distractor
                    N = self._pool_embs.size(0)
                    idx = torch.randint(0, N, (B, P), device=self._pool_embs.device)
                    neg_embs = self._pool_embs[idx].to(query_emb.device)  # [B, P, D]
                out = self.model(
                    query_emb=query_emb,
                    cand_emb=cand_embs,
                    beam_labels=beam_labels,
                    beam_rewards=beam_rewards,
                    beam_mask=beam_mask,
                    neg_weight=self.hparams.neg_weight,
                    num_neg=self.hparams.num_neg,
                    neg_embs=neg_embs,
                )
                if "rce" in out:
                    self.log(f"{stage}/rce", out["rce"], sync_dist=True,
                             on_step=False, on_epoch=True)
                if "neg" in out:
                    self.log(f"{stage}/neg", out["neg"], sync_dist=True,
                             on_step=False, on_epoch=True)
            else:
                # Ablation: CE on the top beam (highest reward)
                top_beam_idx = beam_rewards.argmax(dim=-1)           # [B]
                # labels = first shot index of the top beam
                labels = beam_labels[
                    torch.arange(query_emb.size(0)), top_beam_idx, 0
                ]
                out = self.model(
                    query_emb=query_emb,
                    cand_emb=cand_embs,
                    labels=labels,
                )

            loss = out["loss"]
            self.log(f"{stage}/loss", loss, prog_bar=(stage == "train"),
                     sync_dist=True, on_step=(stage == "train"), on_epoch=True)

            # Log temperature
            temp = self.model.log_temperature.exp().item()
            self.log(f"{stage}/temperature", temp, sync_dist=True,
                     on_step=False, on_epoch=True)

            return loss

        # ------------------------------------------------------------------
        # Lightning hooks
        # ------------------------------------------------------------------

        def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
            return self._step(batch, "train")

        def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
            return self._step(batch, "val")

        def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
            return self._step(batch, "test")

        # ------------------------------------------------------------------
        # Optimiser & LR schedule
        # ------------------------------------------------------------------

        def configure_optimizers(self):
            hp = self.hparams
            no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
            param_groups = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": hp.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = torch.optim.AdamW(param_groups, lr=hp.lr)

            def lr_lambda(step: int) -> float:
                if step < hp.warmup_steps:
                    return float(step) / max(1, hp.warmup_steps)
                progress = float(step - hp.warmup_steps) / max(
                    1, hp.max_steps - hp.warmup_steps
                )
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item()))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        # ------------------------------------------------------------------
        # Inference convenience
        # ------------------------------------------------------------------

        @torch.no_grad()
        def select_icds(
            self,
            query_emb: torch.Tensor,
            cand_emb: torch.Tensor,
            shot_num: Optional[int] = None,
        ) -> torch.Tensor:
            """
            Select ICD indices for a batch of queries (inference).

            Args:
                query_emb: [B, D]
                cand_emb:  [B, K, D]
                shot_num:  Override model default shot_num.

            Returns:
                selected: [B, shot_num] integer indices into cand_emb.
            """
            return self.model.select(query_emb, cand_emb, shot_num=shot_num)

else:
    class LeverLMModule:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("pytorch_lightning is required for LeverLMModule")
