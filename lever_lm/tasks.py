"""
Task registry for multi-task Lever-LM (VQA / Captioning / SST-2).

Each task exposes a uniform record format so the rest of the pipeline
(generate_data.py beam generation, the selector, training) stays task-agnostic:

    {
        "id":       str,
        "image":    PIL.Image | None,   # None for text-only tasks (SST-2)
        "question": str,                # user prompt / question / sentence
        "answer":   str,                # ground-truth target the LVLM must produce
    }

A Task provides:
    - total(split)            -> int                       (#rows, no decode)
    - load(split, indices=, max_samples=) -> List[record]  (decodes images)
    - has_image: bool
    - default_prompt: str (used when the dataset has no per-item prompt)
"""

import io
import json
import os
from typing import Dict, List, Optional

from PIL import Image


def _to_pil(img) -> Optional[Image.Image]:
    """Coerce a parquet image cell (PIL, {'bytes':...}, or path) to RGB PIL."""
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB") if img.mode != "RGB" else img
    if isinstance(img, dict) and img.get("bytes") is not None:
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    if isinstance(img, str) and os.path.exists(img):
        return Image.open(img).convert("RGB")
    raise ValueError(f"Unrecognised image cell type: {type(img)}")


# --------------------------------------------------------------------------- #
#  VQAv2
# --------------------------------------------------------------------------- #

class VQATask:
    name = "vqa"
    has_image = True
    default_prompt = ""  # question comes per-item

    PARQUET = "/home/jiyi/lizhiheng/Lever-LM/data/datasets--lmms-lab--VQAv2/snapshots/32665d35052eb4a6d4414851c3c829a72754915a/data"

    def _ds(self, split: str):
        from datasets import load_dataset
        return load_dataset(
            "parquet",
            data_files={split: f"{self.PARQUET}/{split}-*.parquet"},
            split=split,
        )

    def total(self, split: str) -> int:
        return len(self._ds(split))

    def load(self, split: str, indices: Optional[List[int]] = None,
             max_samples: Optional[int] = None) -> List[Dict]:
        ds = self._ds(split)
        if indices is not None:
            ds = ds.select(indices)
        elif max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        return [
            {
                "id": str(r["question_id"]),
                "image": _to_pil(r["image"]),
                "question": r["question"],
                "answer": str(r["multiple_choice_answer"]),
            }
            for r in ds
        ]


# --------------------------------------------------------------------------- #
#  COCO Captioning (lmms-lab/COCO-Caption2017 parquet, val split bundled)
# --------------------------------------------------------------------------- #

class CaptionTask:
    name = "caption"
    has_image = True
    default_prompt = "Please carefully observe the image and come up with a caption for the image."

    DATA_DIR = "/home/jiyi/lizhiheng/Lever-LM/data/coco_caption2017/data"

    def _ds(self, split: str):
        from datasets import load_dataset
        # only the val split was downloaded; treat it as the universe
        return load_dataset(
            "parquet",
            data_files={"val": f"{self.DATA_DIR}/val-*.parquet"},
            split="val",
        )

    def total(self, split: str) -> int:
        return len(self._ds(split))

    def load(self, split: str, indices: Optional[List[int]] = None,
             max_samples: Optional[int] = None) -> List[Dict]:
        ds = self._ds(split)
        if indices is not None:
            ds = ds.select(indices)
        elif max_samples and max_samples < len(ds):
            ds = ds.select(range(max_samples))
        out = []
        for r in ds:
            ans = r["answer"]
            target = ans[0] if isinstance(ans, list) and ans else str(ans)
            prompt = r.get("question") or self.default_prompt
            out.append({
                "id": str(r["question_id"]),
                "image": _to_pil(r["image"]),
                "question": prompt,
                "answer": str(target),
            })
        return out


# --------------------------------------------------------------------------- #
#  SST-2 (text sentiment classification; no images)
# --------------------------------------------------------------------------- #

class SST2Task:
    name = "sst2"
    has_image = False
    default_prompt = (
        "Classify the sentiment of the sentence as positive or negative.\n"
        "Sentence: {text}\nSentiment:"
    )

    DATA_DIR = "/home/jiyi/lizhiheng/Lever-LM/data/sst2"

    def _file(self, split: str) -> str:
        fname = {"train": "train.jsonl", "validation": "dev.jsonl",
                 "dev": "dev.jsonl", "test": "test.jsonl"}.get(split, "train.jsonl")
        return os.path.join(self.DATA_DIR, fname)

    def _read(self, split: str) -> List[Dict]:
        rows = []
        with open(self._file(split)) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def total(self, split: str) -> int:
        return len(self._read(split))

    def load(self, split: str, indices: Optional[List[int]] = None,
             max_samples: Optional[int] = None) -> List[Dict]:
        rows = self._read(split)
        if indices is not None:
            rows = [rows[i] for i in indices]
        elif max_samples and max_samples < len(rows):
            rows = rows[:max_samples]
        out = []
        for i, r in enumerate(rows):
            text = r["text"].strip()
            # label_text may be absent in test.jsonl (label = -1); default mapping
            label = r.get("label_text")
            if label is None:
                label = {0: "negative", 1: "positive"}.get(r.get("label", 1), "positive")
            out.append({
                "id": str(r.get("idx", i)),
                "image": None,
                "question": self.default_prompt.format(text=text),
                "answer": str(label),
            })
        return out


_TASKS = {t.name: t for t in [VQATask(), CaptionTask(), SST2Task()]}


def get_task(name: str):
    if name not in _TASKS:
        raise ValueError(f"Unknown task '{name}'. Choices: {list(_TASKS)}")
    return _TASKS[name]
