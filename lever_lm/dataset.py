"""
VQAv2 dataset with beam-search ICD sequence support for Lever-LM training.

Data layout produced by generate_data.py and consumed here:

    {
        "query_id":     str,
        "query_emb":    [D],            float32
        "cand_ids":     [K],            int  (indices into candidate pool)
        "cand_embs":    [K, D],         float32
        "beam_labels":  [num_beams, shot_num],  int  (indices into cand_embs)
        "beam_rewards": [num_beams],    float32  (log-prob scores)
        "beam_mask":    [num_beams],    bool
    }

The DataModule supports both local JSON/HDF5 files and Hugging Face online
loading (datasets library).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import pytorch_lightning as pl
    _HAS_PL = True
except ImportError:
    _HAS_PL = False


# --------------------------------------------------------------------------- #
#  Raw VQAv2 helpers (for online HF loading)
# --------------------------------------------------------------------------- #

def load_vqav2_hf(split: str = "validation", cache_dir: Optional[str] = None):
    """Load VQAv2 from HuggingFace datasets (requires internet / cache)."""
    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceM4/VQAv2",
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    return ds


# --------------------------------------------------------------------------- #
#  Core beam-dataset
# --------------------------------------------------------------------------- #

class VQAv2BeamDataset(Dataset):
    """
    Dataset of pre-computed query/candidate embeddings and beam-search ICD
    sequences with reward scores.  Everything is stored as tensors on CPU and
    moved to the target device by the DataLoader / Lightning.

    Args:
        data_file:  Path to a JSON file produced by generate_data.py, or a list
                    of already-parsed dicts.
        max_beams:  Cap number of beams per sample (pads/truncates to this).
        shot_num:   Shots per ICD sequence (must match data).
        K:          Candidate pool size (pads cand_embs to [K, D] if needed).
    """

    def __init__(
        self,
        data_file: Union[str, List[Dict]],
        max_beams: int = 5,
        shot_num: int = 2,
        K: int = 64,
    ):
        super().__init__()
        self.max_beams = max_beams
        self.shot_num = shot_num
        self.K = K

        if isinstance(data_file, (str, Path)):
            with open(data_file, "r") as f:
                raw = json.load(f)
        else:
            raw = data_file

        self.samples = [self._parse(r) for r in raw]

    # ------------------------------------------------------------------
    # Parsing & normalisation
    # ------------------------------------------------------------------

    def _parse(self, r: Dict) -> Dict[str, torch.Tensor]:
        query_emb = torch.tensor(r["query_emb"], dtype=torch.float32)
        cand_embs = torch.tensor(r["cand_embs"], dtype=torch.float32)  # [K', D]
        K_actual = cand_embs.size(0)
        D = cand_embs.size(1)

        # Pad candidate pool to self.K if necessary
        if K_actual < self.K:
            pad = torch.zeros(self.K - K_actual, D)
            cand_embs = torch.cat([cand_embs, pad], dim=0)
        else:
            cand_embs = cand_embs[: self.K]

        # Parse beam data
        raw_labels = r.get("beam_labels", [])
        raw_rewards = r.get("beam_rewards", [])

        beam_labels = torch.zeros(self.max_beams, self.shot_num, dtype=torch.long)
        beam_rewards = torch.zeros(self.max_beams, dtype=torch.float32)
        beam_mask = torch.zeros(self.max_beams, dtype=torch.bool)

        n_beams = min(len(raw_labels), self.max_beams)
        for i in range(n_beams):
            seq = raw_labels[i]
            for j in range(min(len(seq), self.shot_num)):
                beam_labels[i, j] = seq[j]
            beam_rewards[i] = raw_rewards[i] if i < len(raw_rewards) else 0.0
            beam_mask[i] = True

        return {
            "query_emb": query_emb,          # [D]
            "cand_embs": cand_embs,           # [K, D]
            "beam_labels": beam_labels,       # [num_beams, shot_num]
            "beam_rewards": beam_rewards,     # [num_beams]
            "beam_mask": beam_mask,           # [num_beams]
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def collate_beam_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Default collate: stack tensors along batch dimension."""
    return {k: torch.stack([s[k] for s in batch]) for k in batch[0]}


# --------------------------------------------------------------------------- #
#  Lightning DataModule
# --------------------------------------------------------------------------- #

if _HAS_PL:
    class VQAv2BeamDataModule(pl.LightningDataModule):
        """
        PyTorch Lightning DataModule for Lever-LM training on VQAv2.

        Args:
            train_file:   Path to training JSON.
            val_file:     Path to validation JSON.
            test_file:    Path to test JSON (optional).
            batch_size:   Training batch size.
            num_workers:  DataLoader worker count.
            max_beams:    Max beam sequences per sample.
            shot_num:     ICD sequence length.
            K:            Candidate pool size.
        """

        def __init__(
            self,
            train_file: str,
            val_file: str,
            test_file: Optional[str] = None,
            batch_size: int = 32,
            num_workers: int = 4,
            max_beams: int = 5,
            shot_num: int = 2,
            K: int = 64,
        ):
            super().__init__()
            self.save_hyperparameters()

        def setup(self, stage: Optional[str] = None):
            hp = self.hparams
            if stage in ("fit", None):
                self.train_ds = VQAv2BeamDataset(
                    hp.train_file, hp.max_beams, hp.shot_num, hp.K
                )
                self.val_ds = VQAv2BeamDataset(
                    hp.val_file, hp.max_beams, hp.shot_num, hp.K
                )
            if stage in ("test", None) and hp.test_file:
                self.test_ds = VQAv2BeamDataset(
                    hp.test_file, hp.max_beams, hp.shot_num, hp.K
                )

        def train_dataloader(self) -> DataLoader:
            return DataLoader(
                self.train_ds,
                batch_size=self.hparams.batch_size,
                shuffle=True,
                num_workers=self.hparams.num_workers,
                collate_fn=collate_beam_batch,
                pin_memory=True,
            )

        def val_dataloader(self) -> DataLoader:
            return DataLoader(
                self.val_ds,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.hparams.num_workers,
                collate_fn=collate_beam_batch,
                pin_memory=True,
            )

        def test_dataloader(self) -> DataLoader:
            return DataLoader(
                self.test_ds,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.hparams.num_workers,
                collate_fn=collate_beam_batch,
                pin_memory=True,
            )

else:
    # Stub so imports don't break when pytorch_lightning is absent
    class VQAv2BeamDataModule:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("pytorch_lightning is required for VQAv2BeamDataModule")


# --------------------------------------------------------------------------- #
#  Candidate pool helper used during inference
# --------------------------------------------------------------------------- #

class VQAv2CandidatePool:
    """
    Lightweight wrapper around a pre-computed candidate pool for inference.

    Supports:
      - Legacy: single .json with inline "embedding" lists
      - Large pool: {base}.jsonl + {base}.pt (embeddings stored separately)
    """

    def __init__(self, pool_file: str):
        path = Path(pool_file)
        if path.suffix == ".jsonl" or (path.with_suffix(".jsonl").exists() and path.suffix != ".json"):
            jsonl = path if path.suffix == ".jsonl" else path.with_suffix(".jsonl")
            pt = jsonl.with_suffix(".pt")
            if not jsonl.exists() or not pt.exists():
                raise FileNotFoundError(f"Pool requires both {jsonl} and {pt}")
            self.items = []
            with open(jsonl) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.items.append(json.loads(line))
            self.embeddings = torch.load(pt, map_location="cpu", weights_only=True).float()
            if self.embeddings.shape[0] != len(self.items):
                raise ValueError(
                    f"Pool size mismatch: {len(self.items)} items vs {self.embeddings.shape[0]} embeddings"
                )
        else:
            with open(pool_file, "r") as f:
                data = json.load(f)
            self.items = data
            self.embeddings = torch.tensor(
                [d["embedding"] for d in data], dtype=torch.float32
            )  # [N, D]

    def __len__(self) -> int:
        return len(self.items)

    def get_embeddings(self) -> torch.Tensor:
        return self.embeddings

    def get_item(self, idx: int) -> Dict:
        return self.items[idx]

    def get_items(self, indices: List[int]) -> List[Dict]:
        return [self.items[i] for i in indices]
