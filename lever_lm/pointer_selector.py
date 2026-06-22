"""
PointerSelector: autoregressive pointer-network decoder for ICD selection.

Why autoregressive?
    The previous version scored every candidate with ONE fixed logits vector and
    summed per-position log-probs.  Because addition is commutative, the sequence
    [3, 9] and [9, 3] received *identical* scores -- the model was permutation
    invariant and could not learn ordering.

    This version conditions each step on the ordered prefix of already-selected
    ICDs (like the original Lever-LM's GPT2/LSTM decoder):

        decoder input sequence = [query, sel_1, sel_2, ...]   (in order)

    At step t a causal self-attention (sees the ordered prefix) plus a
    cross-attention over the candidate pool produce a pointer distribution for
    the next selection.  Hence

        P([3, 9]) = P(3 | q) * P(9 | q, 3)
        P([9, 3]) = P(9 | q) * P(3 | q, 9)

    which are different in general -- ordering is now modelled explicitly.

The public interface (forward / compute_log_probs_per_beam /
multi_target_rce_loss / select) and the constructor signature are unchanged, so
the dataset / Lightning module / train.py keep working without edits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class PointerSelector(nn.Module):
    """
    Autoregressive pointer-network decoder for ordered ICD sequence selection.

    Args:
        d_model:        Input embedding dimension (matches LVLM feature dim).
        K:              Candidate pool size (original Lever-LM default: 64).
        shot_num:       Number of shots (ICDs) to select per query.
        label_smoothing: Label smoothing for single-target CE loss.
        dropout:        Dropout rate on projections / FFN.
        hidden_dim:     Internal attention dimension.
        num_heads:      Number of attention heads.
        attn_dropout:   Dropout inside MultiheadAttention.
        num_layers:     Number of stacked decoder layers.
    """

    def __init__(
        self,
        d_model: int = 768,
        K: int = 64,
        shot_num: int = 2,
        label_smoothing: float = 0.1,
        dropout: float = 0.1,
        hidden_dim: int = 256,
        num_heads: int = 4,
        attn_dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.K = K
        self.shot_num = shot_num
        self.num_layers = num_layers
        self.label_smoothing = label_smoothing

        # Project input features into the attention hidden space (shared by the
        # query and the candidates).
        self.input_proj = (
            nn.Linear(d_model, hidden_dim, bias=False)
            if d_model != hidden_dim
            else nn.Identity()
        )

        # Token-type embeddings distinguish the query token (position 0) from the
        # selected-ICD tokens, and positional embeddings encode the step index so
        # that order is representable. Sized generously for inference flexibility.
        self.max_positions = max(shot_num + 1, 8)
        self.pos_emb = nn.Embedding(self.max_positions, hidden_dim)
        self.type_emb = nn.Embedding(2, hidden_dim)  # 0 = query, 1 = selected ICD

        # Stacked decoder layers: causal self-attention over the decoder prefix,
        # then cross-attention over candidate memory, then FFN.
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=attn_dropout,
                                  batch_first=True)
            for _ in range(num_layers)
        ])
        self.self_attn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=attn_dropout,
                                  batch_first=True)
            for _ in range(num_layers)
        ])
        self.cross_attn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Pointer projections (decoder state vs. candidate keys).
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.cand_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Learnable temperature for RCE reward weighting
        self.log_temperature = nn.Parameter(torch.tensor([-2.3026]))  # init ≈ 0.1

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self):
        if not isinstance(self.input_proj, nn.Identity):
            nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.cand_proj.weight)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        for ffn in self.ffn_layers:
            nn.init.xavier_uniform_(ffn[0].weight)
            nn.init.zeros_(ffn[0].bias)
            nn.init.xavier_uniform_(ffn[3].weight)
            nn.init.zeros_(ffn[3].bias)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project(
        self, query_emb: torch.Tensor, cand_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project query [B, D] and candidates [B, K, D] into hidden_dim."""
        B, K, D = cand_emb.shape
        query_h = self.input_proj(query_emb)                    # [B, H]
        cand_h = self.input_proj(cand_emb.reshape(B * K, D))    # [B*K, H]
        cand_h = cand_h.reshape(B, K, self.hidden_dim)          # [B, K, H]
        return query_h, cand_h

    def _add_pos_type(self, seq: torch.Tensor) -> torch.Tensor:
        """Add positional + token-type embeddings to a decoder input sequence.

        seq: [B, T, H] where position 0 is the query token and positions 1..T-1
        are selected-ICD tokens.
        """
        B, T, H = seq.shape
        pos_ids = torch.arange(T, device=seq.device).clamp_max(self.max_positions - 1)
        type_ids = torch.zeros(T, dtype=torch.long, device=seq.device)
        type_ids[1:] = 1
        return seq + self.pos_emb(pos_ids).unsqueeze(0) + self.type_emb(type_ids).unsqueeze(0)

    def _decode(
        self, dec_in: torch.Tensor, cand_h: torch.Tensor
    ) -> torch.Tensor:
        """Run the causal decoder.

        Args:
            dec_in: [B, T, H] decoder input sequence ([query, sel_1, ...]).
            cand_h: [B, K, H] candidate memory.
        Returns:
            hidden states [B, T, H].
        """
        B, T, H = dec_in.shape
        x = self._add_pos_type(dec_in)
        # Causal mask: position t may attend to positions <= t (float additive).
        causal = torch.full((T, T), float("-inf"), device=dec_in.device)
        causal = torch.triu(causal, diagonal=1)
        for sa, sa_norm, ca, ca_norm, ffn, ffn_norm in zip(
            self.self_attn_layers, self.self_attn_norms,
            self.cross_attn_layers, self.cross_attn_norms,
            self.ffn_layers, self.ffn_norms,
        ):
            attended, _ = sa(x, x, x, attn_mask=causal, need_weights=False)
            x = sa_norm(x + self.dropout(attended))
            crossed, _ = ca(x, cand_h, cand_h, need_weights=False)
            x = ca_norm(x + self.dropout(crossed))
            x = ffn_norm(x + self.dropout(ffn(x)))
        return x

    def _pointer_from_state(
        self, state: torch.Tensor, cand_h: torch.Tensor
    ) -> torch.Tensor:
        """Pointer logits from a decoder state.

        Args:
            state:  [B, T, H] (or [B, H] for a single step).
            cand_h: [B, K, H].
        Returns:
            logits [B, T, K] (or [B, K] if state was [B, H]).
        """
        squeeze = False
        if state.dim() == 2:
            state = state.unsqueeze(1)
            squeeze = True
        q = self.query_proj(state)                              # [B, T, H]
        c = self.cand_proj(cand_h)                              # [B, K, H]
        scale = self.hidden_dim ** 0.5
        logits = torch.einsum("bth,bkh->btk", q, c) / scale    # [B, T, K]
        return logits.squeeze(1) if squeeze else logits

    def _first_step_logits(
        self, query_emb: torch.Tensor, cand_emb: torch.Tensor
    ) -> torch.Tensor:
        """Pointer logits for the FIRST selection (query only). [B, K].

        Used for the single-target CE fallback and as a convenience `logits`
        output in forward().
        """
        query_h, cand_h = self._project(query_emb, cand_emb)
        hidden = self._decode(query_h.unsqueeze(1), cand_h)     # [B, 1, H]
        return self._pointer_from_state(hidden[:, -1, :], cand_h)

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _single_target_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Standard CE with label smoothing. labels: [B] (first-shot index)."""
        return F.cross_entropy(
            logits, labels, label_smoothing=self.label_smoothing
        )

    def compute_log_probs_per_beam(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        beam_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        For each beam, compute the autoregressive log-prob of its ordered ICD
        sequence:  sum_t log P(label_t | query, label_<t).

        Because each step is conditioned on the ordered prefix, two beams that
        are permutations of each other (e.g. [3,9] vs [9,3]) receive different
        scores.

        Args:
            query_emb:   [B, D]
            cand_emb:    [B, K, D]  (may be a local subset)
            beam_labels: [B, num_beams, shot_num]  (indices into cand_emb)

        Returns:
            log_probs: [B, num_beams]
        """
        B, num_beams, shot_num = beam_labels.shape
        query_h, cand_h = self._project(query_emb, cand_emb)     # [B,H], [B,K,H]
        H = self.hidden_dim
        K = cand_h.size(1)

        # Build per-beam decoder inputs: [query, cand[label_0], ..., cand[label_{S-2}]]
        # Expand query / candidates across the beam dimension.
        q_exp = query_h.unsqueeze(1).expand(B, num_beams, H)            # [B,nb,H]
        cand_exp = cand_h.unsqueeze(1).expand(B, num_beams, K, H)       # [B,nb,K,H]

        # Gather embeddings of the first shot_num-1 selected candidates (teacher
        # forcing). For shot_num == 1 this prefix is empty.
        if shot_num > 1:
            prefix_idx = beam_labels[:, :, : shot_num - 1]              # [B,nb,S-1]
            idx = prefix_idx.unsqueeze(-1).expand(B, num_beams, shot_num - 1, H)
            sel_embs = torch.gather(cand_exp, 2, idx)                   # [B,nb,S-1,H]
            dec_in = torch.cat([q_exp.unsqueeze(2), sel_embs], dim=2)   # [B,nb,S,H]
        else:
            dec_in = q_exp.unsqueeze(2)                                 # [B,nb,1,H]

        # Flatten beams into the batch dimension for a single decoder pass.
        dec_in = dec_in.reshape(B * num_beams, shot_num, H)
        cand_flat = cand_exp.reshape(B * num_beams, K, H)

        hidden = self._decode(dec_in, cand_flat)                       # [B*nb,S,H]
        logits = self._pointer_from_state(hidden, cand_flat)          # [B*nb,S,K]
        log_probs = F.log_softmax(logits, dim=-1)                     # [B*nb,S,K]

        labels_flat = beam_labels.reshape(B * num_beams, shot_num)    # [B*nb,S]
        step_lp = torch.gather(
            log_probs, 2, labels_flat.unsqueeze(-1)
        ).squeeze(-1)                                                  # [B*nb,S]
        beam_lp = step_lp.sum(dim=-1).reshape(B, num_beams)           # [B,nb]
        return beam_lp

    @staticmethod
    def _sample_negative_sequences(
        B: int, K: int, shot_num: int, num_neg: int, device: torch.device
    ) -> torch.Tensor:
        """Sample `num_neg` random ordered sequences (no internal repeats) of
        length `shot_num` from K candidates, per batch element.

        Returns: [B, num_neg, shot_num] long.  Beam search keeps the *best*
        sequences, so uniformly-random sequences are almost surely worse and make
        good hard-ish negatives for teaching the model WHICH candidates to point
        at (and in roughly what order).
        """
        # random scores -> argsort -> take first shot_num as a random permutation
        rand = torch.rand(B, num_neg, K, device=device)
        perm = rand.argsort(dim=-1)               # [B, num_neg, K]
        return perm[:, :, :shot_num].contiguous()  # [B, num_neg, shot_num]

    def hard_negative_loss(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        beam_labels: torch.Tensor,
        beam_mask: torch.Tensor,
        num_neg: int = 16,
    ) -> torch.Tensor:
        """InfoNCE-style contrast: positive beams should get higher sequence
        log-prob than randomly sampled negative sequences.

        For each valid positive beam p:
            loss_p = -log( exp(lp_p) / (exp(lp_p) + Σ_j exp(lp_negj)) )

        This directly teaches the pointer to concentrate on the candidates that
        appear in high-reward beams (fixing the "select() never matches" failure),
        which is the prerequisite for any ordering to matter.
        """
        B, num_beams, shot_num = beam_labels.shape
        K = cand_emb.size(1)
        device = cand_emb.device

        pos_lp = self.compute_log_probs_per_beam(query_emb, cand_emb, beam_labels)  # [B, nb]
        neg_labels = self._sample_negative_sequences(B, K, shot_num, num_neg, device)
        neg_lp = self.compute_log_probs_per_beam(query_emb, cand_emb, neg_labels)   # [B, num_neg]

        # logsumexp over the shared negative pool (broadcast across positives)
        neg_lse = torch.logsumexp(neg_lp, dim=1, keepdim=True)            # [B, 1]
        # denom for each positive = logaddexp(pos_lp, neg_lse)
        denom = torch.logaddexp(pos_lp, neg_lse.expand_as(pos_lp))        # [B, nb]
        per_pos = pos_lp - denom                                          # [B, nb] = log softmax score
        # mask invalid beams, average over valid positives
        per_pos = per_pos.masked_fill(~beam_mask, 0.0)
        n_valid = beam_mask.sum(dim=1).clamp(min=1)                       # [B]
        loss = -(per_pos.sum(dim=1) / n_valid).mean()
        return loss

    def pool_negative_loss(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        beam_labels: torch.Tensor,
        beam_mask: torch.Tensor,
        neg_embs: torch.Tensor,
        num_neg: int = 16,
    ) -> torch.Tensor:
        """InfoNCE with GENUINELY-BAD negatives drawn from the broad pool.

        The per-query candidate set (top-K retrieval) consists of similar/good
        ICDs, so random sequences *within* it are not真负 (equally good).  Here we
        append `neg_embs` (random pool examples, mostly dissimilar -> poor ICDs)
        to the candidate set and ask the model to score the real beam sequences
        higher than sequences built purely from these distractors.

        Args:
            neg_embs: [B, P, D] distractor embeddings sampled from the pool.
        """
        B, num_beams, shot_num = beam_labels.shape
        K = cand_emb.size(1)
        P = neg_embs.size(1)
        device = cand_emb.device

        aug = torch.cat([cand_emb, neg_embs], dim=1)                  # [B, K+P, D]
        pos_lp = self.compute_log_probs_per_beam(query_emb, aug, beam_labels)  # [B,nb]

        rand = torch.rand(B, num_neg, P, device=device)
        neg_idx = rand.argsort(dim=-1)[:, :, :shot_num] + K           # slots in [K, K+P)
        neg_lp = self.compute_log_probs_per_beam(query_emb, aug, neg_idx)      # [B,num_neg]

        neg_lse = torch.logsumexp(neg_lp, dim=1, keepdim=True)        # [B,1]
        denom = torch.logaddexp(pos_lp, neg_lse.expand_as(pos_lp))    # [B,nb]
        per_pos = (pos_lp - denom).masked_fill(~beam_mask, 0.0)
        n_valid = beam_mask.sum(dim=1).clamp(min=1)
        return -(per_pos.sum(dim=1) / n_valid).mean()

    def multi_target_rce_loss(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        beam_labels: torch.Tensor,
        beam_rewards: torch.Tensor,
        beam_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reward-weighted cross-entropy (RCE) over multiple beam sequences.

        The weight for each beam is softmax over rewards / temperature, masked by
        beam_mask. The loss is the negative expected (autoregressive) log-prob
        under this reward-weighted distribution.
        """
        log_probs = self.compute_log_probs_per_beam(
            query_emb, cand_emb, beam_labels
        )  # [B, num_beams]

        temperature = self.log_temperature.exp().clamp(min=1e-4)

        masked_rewards = beam_rewards.masked_fill(~beam_mask, float("-inf"))
        weights = F.softmax(masked_rewards / temperature, dim=-1)  # [B, num_beams]
        weights = weights.nan_to_num(0.0)

        log_probs = log_probs.masked_fill(~beam_mask, 0.0)
        loss = -(weights * log_probs).sum(dim=-1).mean()
        return loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        beam_labels: Optional[torch.Tensor] = None,
        beam_rewards: Optional[torch.Tensor] = None,
        beam_mask: Optional[torch.Tensor] = None,
        neg_weight: float = 0.0,
        num_neg: int = 16,
        neg_embs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass.

        - If beam_labels + beam_rewards + beam_mask are provided → multi-target RCE
          (+ optional hard-negative InfoNCE term weighted by `neg_weight`).
        - Else if labels is provided → single-target CE (fallback / ablation).
        - Otherwise → inference mode, returns first-step logits only.

        Returns:
            dict with keys: "loss" (if supervised), "logits" [B, K] (first step),
            and "rce"/"neg" component scalars when applicable.
        """
        logits = self._first_step_logits(query_emb, cand_emb)
        out: Dict[str, torch.Tensor] = {"logits": logits}

        if beam_labels is not None and beam_rewards is not None and beam_mask is not None:
            rce = self.multi_target_rce_loss(
                query_emb, cand_emb, beam_labels, beam_rewards, beam_mask
            )
            out["rce"] = rce.detach()
            loss = rce
            if neg_weight > 0.0:
                if neg_embs is not None:
                    neg = self.pool_negative_loss(
                        query_emb, cand_emb, beam_labels, beam_mask,
                        neg_embs=neg_embs, num_neg=num_neg
                    )
                else:
                    neg = self.hard_negative_loss(
                        query_emb, cand_emb, beam_labels, beam_mask, num_neg=num_neg
                    )
                out["neg"] = neg.detach()
                loss = rce + neg_weight * neg
            out["loss"] = loss
        elif labels is not None:
            out["loss"] = self._single_target_loss(logits, labels)

        return out

    # ------------------------------------------------------------------
    # Inference: greedy autoregressive selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select(
        self,
        query_emb: torch.Tensor,
        cand_emb: torch.Tensor,
        shot_num: Optional[int] = None,
        exclude_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Greedily select `shot_num` candidates by TRUE autoregressive decoding:
        each step re-runs the decoder over the [query, sel_1, ...] prefix so the
        next pick is conditioned on the ordered already-selected ICDs.

        Args:
            query_emb:       [B, D]
            cand_emb:        [B, K, D]
            shot_num:        defaults to self.shot_num
            exclude_indices: [B, M] indices to mask out (already chosen)

        Returns:
            selected: [B, shot_num] indices into cand_emb
        """
        shot_num = shot_num or self.shot_num
        B, K, _ = cand_emb.shape
        query_h, cand_h = self._project(query_emb, cand_emb)

        mask = torch.zeros(B, K, dtype=torch.bool, device=query_emb.device)
        if exclude_indices is not None:
            mask.scatter_(1, exclude_indices, True)

        selected = []
        dec_in = query_h.unsqueeze(1)                            # [B, 1, H]
        for _ in range(shot_num):
            hidden = self._decode(dec_in, cand_h)               # [B, t+1, H]
            logits = self._pointer_from_state(hidden[:, -1, :], cand_h)  # [B, K]
            logits = logits.masked_fill(mask, float("-inf"))
            idx = logits.argmax(dim=-1)                          # [B]
            selected.append(idx)
            mask.scatter_(1, idx.unsqueeze(1), True)
            # Append the chosen candidate's projected embedding to the prefix.
            chosen = torch.gather(
                cand_h, 1, idx.view(B, 1, 1).expand(B, 1, self.hidden_dim)
            )                                                    # [B, 1, H]
            dec_in = torch.cat([dec_in, chosen], dim=1)

        return torch.stack(selected, dim=1)  # [B, shot_num]
