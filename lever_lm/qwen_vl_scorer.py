"""
QwenVLScorer: wraps Qwen2-VL to compute ICD sequence scores for Lever-LM
data generation (Eq. 3 in the paper).

Given a query (image + question) and an ordered ICD sequence S^K, the scorer
computes the log-probability that the model generates the ground-truth answer,
i.e.  log P_M(y* | S^K, x).  This score is then used as the reward signal
during beam-search data generation.

Usage:
    scorer = QwenVLScorer("Qwen/Qwen2-VL-2B-Instruct")
    score  = scorer.score(icd_list, query_image, query_text, gt_answer)
    scores = scorer.score_batch_true(icd_lists, query_image, query_text, gt_answer)
    emb    = scorer.embed(image, text)   # for candidate pool embedding
"""

import math
from typing import Dict, List, Optional, Tuple, Union


import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _load_image(image: Union[str, Image.Image]) -> Image.Image:
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    return image.convert("RGB") if image.mode != "RGB" else image


def _answer_token_ids(
    processor: AutoProcessor, answer: str
) -> torch.Tensor:
    """Tokenise just the answer string (no special tokens around it)."""
    return processor.tokenizer(
        answer, add_special_tokens=False, return_tensors="pt"
    ).input_ids.squeeze(0)


def _all_answer_token_variants(
    processor: AutoProcessor, answer: str
) -> list:
    """Return several tokenisation variants of `answer` to improve matching.

    Qwen tokeniser may encode the answer differently depending on context
    (leading space, newline, capital, etc.). We try 4 variants and return
    all unique results as a list of 1-D tensors.
    """
    variants = set()
    for prefix in ("", " ", "\n", " \n"):
        ids = processor.tokenizer(
            prefix + answer, add_special_tokens=False
        ).input_ids
        # Strip the prefix tokens so only the answer tokens remain
        prefix_len = len(processor.tokenizer(
            prefix, add_special_tokens=False
        ).input_ids) if prefix else 0
        core = ids[prefix_len:]
        if core:
            variants.add(tuple(core))
    return [torch.tensor(list(v), dtype=torch.long) for v in variants]


# --------------------------------------------------------------------------- #
#  Main class
# --------------------------------------------------------------------------- #

class QwenVLScorer:
    """
    Qwen2-VL wrapper for:
      1. log P_M(y* | S^K, x)  → ICD sequence reward for data generation.
      2. Dense embedding of (image, text) pairs  → candidate pool features.

    Args:
        model_name:   HuggingFace model identifier or local path.
        device:       "cuda" / "cpu" / "auto" (uses accelerate device_map).
        torch_dtype:  Precision override; defaults to bfloat16 on CUDA.
        max_pixels:   Qwen-VL image resolution cap.
        embed_layer:  Which transformer layer to extract embeddings from
                      (-1 = last hidden state before LM head).
    """

    MODELSCOPE_PATH = "/home/jiyi/.cache/modelscope/qwen/Qwen2-VL-2B-Instruct"
    DEFAULT_MODEL = MODELSCOPE_PATH

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        max_pixels: int = 1280 * 28 * 28,
        embed_layer: int = -1,
    ):
        self.model_name = model_name
        self.embed_layer = embed_layer

        if model_name is None:
            model_name = self.DEFAULT_MODEL
        self.model_name = model_name

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        device_map = device if device == "auto" else None
        explicit_device = None if device == "auto" else device

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        if explicit_device is not None:
            self.model = self.model.to(explicit_device)

        self.processor = AutoProcessor.from_pretrained(
            model_name, max_pixels=max_pixels, trust_remote_code=True
        )
        self.model.eval()

        # Infer actual device (handles device_map="auto" multi-GPU case)
        self._device = next(self.model.parameters()).device

        # Left-padding required for correct log-prob extraction in batched mode
        self.processor.tokenizer.padding_side = "left"
        self._instruct = self._detect_instruct(model_name)

    @staticmethod
    def _detect_instruct(model_name: str) -> bool:
        n = model_name.lower()
        return "instruct" in n or "chat" in n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score(
        self,
        icd_list: List[Dict],
        query_image: Union[str, Image.Image],
        query_text: str,
        gt_answer: str,
    ) -> float:
        """
        Compute  log P_M(y* | S^K, x)  for a single (query, ICD-sequence) pair.

        Args:
            icd_list:    List of ICDs, each a dict with keys:
                           "image"  (str path or PIL Image)
                           "question" (str)
                           "answer"   (str)
            query_image: Query image (path or PIL).
            query_text:  Query question string.
            gt_answer:   Ground-truth answer string.

        Returns:
            Scalar log-prob (float, ≤ 0).
        """
        messages = self._build_messages(icd_list, query_image, query_text)
        inputs, images = self._prepare_inputs(messages)

        # Full forward pass to get logits
        outputs = self.model(**inputs, output_hidden_states=False)
        logits = outputs.logits  # [1, seq_len, vocab]

        # Try multiple tokenisation variants to reduce -inf occurrences
        answer_ids = _answer_token_ids(self.processor, gt_answer).to(self._device)
        answer_variants = [v.to(self._device)
                           for v in _all_answer_token_variants(self.processor, gt_answer)]
        log_prob = self._answer_log_prob(
            logits[0], inputs["input_ids"][0], answer_ids, answer_variants
        )
        return log_prob.item()

    @torch.no_grad()
    def score_batch(
        self,
        icd_lists: List[List[Dict]],
        query_images: List[Union[str, Image.Image]],
        query_texts: List[str],
        gt_answers: List[str],
    ) -> List[float]:
        """Batch version of `score` (independent samples, no true batching across
        different-length histories – loops internally for simplicity)."""
        return [
            self.score(icds, img, txt, ans)
            for icds, img, txt, ans in zip(
                icd_lists, query_images, query_texts, gt_answers
            )
        ]

    @torch.no_grad()
    def score_batch_true(
        self,
        icd_lists: List[List[Dict]],
        query_image: Union[str, Image.Image],
        query_text: str,
        gt_answer: str,
    ) -> List[float]:
        """
        True batched scoring: evaluates B different ICD sequences against the
        **same** query in a single forward pass.

        All sequences must have the same shot_num (same number of ICD items),
        which is always the case within one beam-search step.

        Args:
            icd_lists:   List of B ICD sequences (each a list of dicts).
            query_image: Shared query image (path or PIL).
            query_text:  Shared query question string.
            gt_answer:   Shared ground-truth answer string.

        Returns:
            List of B float log-probs (≤ 0).
        """
        B = len(icd_lists)
        if B == 0:
            return []
        if B == 1:
            return [self.score(icd_lists[0], query_image, query_text, gt_answer)]

        query_img = _load_image(query_image) if query_image is not None else None

        # Build per-sample chat texts and collect images in order
        all_texts: List[str] = []
        all_images_flat: List[Image.Image] = []  # flat list: img0_s0, img1_s0, img0_s1, ...

        for icd_list in icd_lists:
            messages = self._build_messages(icd_list, query_img, query_text)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_texts.append(text)
            for msg in messages:
                for item in msg.get("content", []):
                    if item.get("type") == "image":
                        img = item["image"]
                        all_images_flat.append(
                            img if isinstance(img, Image.Image) else _load_image(img)
                        )

        # Batch tokenise + image-encode; processor assigns images to samples by
        # matching the flat image list to <|image_pad|> tokens in each text.
        inputs = self.processor(
            text=all_texts,
            images=all_images_flat if all_images_flat else None,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=False)
        logits = outputs.logits  # [B, seq_len, vocab]

        answer_ids = _answer_token_ids(self.processor, gt_answer).to(self._device)
        answer_variants = [v.to(self._device)
                           for v in _all_answer_token_variants(self.processor, gt_answer)]
        return [
            self._answer_log_prob(
                logits[b], inputs["input_ids"][b], answer_ids, answer_variants
            ).item()
            for b in range(B)
        ]

    # ------------------------------------------------------------------
    # Teacher-forcing scoring (robust; replaces fragile span-search)
    # ------------------------------------------------------------------

    def _answer_ids_for(self, messages: List[Dict], gt_answer: str):
        """Return (full_text, answer_token_ids) by teacher-forcing the answer."""
        tok = self.processor.tokenizer
        if self._instruct:
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text, _ = self._flatten_base_messages(messages)
        full_text = prompt_text + str(gt_answer)
        prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
        full_ids = tok(full_text, add_special_tokens=False).input_ids
        answer_ids = full_ids[len(prompt_ids):]
        return full_text, answer_ids

    @torch.no_grad()
    def score_batch_tf(
        self,
        icd_lists: List[List[Dict]],
        query_image: Union[str, Image.Image],
        query_text: str,
        gt_answer: str,
    ) -> List[float]:
        """Teacher-forcing batched scoring: log P(y* | S^K, x) for B sequences.

        Appends the ground-truth answer to each prompt and reads off the
        answer-token log-probs at their exact (left-padded) positions. This
        is robust: the answer always exists in the input, so no -inf from
        failed span matching.
        """
        B = len(icd_lists)
        if B == 0:
            return []
        query_img = _load_image(query_image) if query_image is not None else None

        all_texts: List[str] = []
        all_images_flat: List[Image.Image] = []
        ans_ids_list: List[List[int]] = []
        for icd_list in icd_lists:
            messages = self._build_messages(icd_list, query_img, query_text)
            full_text, a_ids = self._answer_ids_for(messages, gt_answer)
            all_texts.append(full_text)
            ans_ids_list.append(a_ids)
            for msg in messages:
                for item in msg.get("content", []):
                    if item.get("type") == "image":
                        img = item["image"]
                        all_images_flat.append(
                            img if isinstance(img, Image.Image) else _load_image(img)
                        )

        inputs = self.processor(
            text=all_texts,
            images=all_images_flat if all_images_flat else None,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            k: (v.long() if k == "input_ids" else v).to(self._device)
            for k, v in inputs.items()
        }
        logits = self.model(**inputs, output_hidden_states=False).logits  # [B, L, V]
        L = inputs["input_ids"].shape[1]

        scores: List[float] = []
        for b in range(B):
            a_ids = ans_ids_list[b]
            La = len(a_ids)
            if La == 0:
                scores.append(float("-inf"))
                continue
            # Answer occupies the last La positions (left padding) -> predicted
            # by logits at positions [L-La-1, L-2].
            sub = logits[b, L - La - 1: L - 1, :].float()      # [La, V]
            lp = F.log_softmax(sub, dim=-1)                     # [La, V]
            idx = torch.tensor(a_ids, device=lp.device)
            scores.append(lp.gather(1, idx.unsqueeze(1)).sum().item())
        return scores

    @torch.no_grad()
    def score_tf(
        self,
        icd_list: List[Dict],
        query_image: Union[str, Image.Image],
        query_text: str,
        gt_answer: str,
    ) -> float:
        """Single-sequence teacher-forcing score (also used for the baseline
        log P(y*|x) when icd_list is empty)."""
        return self.score_batch_tf([icd_list], query_image, query_text, gt_answer)[0]

    @torch.no_grad()
    def embed(
        self,
        image: Optional[Union[str, Image.Image]],
        text: str,
        pool: str = "mean",
    ) -> torch.Tensor:
        """
        Produce a dense embedding for a single (image, text) pair.

        The embedding is extracted from the last hidden state of the vision-
        language encoder, then mean-pooled over the sequence dimension.

        Args:
            image:  Image path or PIL Image.
            text:   Caption / question string.
            pool:   "mean" (default) or "cls" (first token).

        Returns:
            1-D float32 tensor on CPU, shape [hidden_size].
        """
        messages = [
            {"role": "user", "content": self._user_content(image, text)}
        ]
        inputs, _ = self._prepare_inputs(messages)
        outputs = self.model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[self.embed_layer]  # [1, L, H]
        if pool == "cls":
            emb = hidden[0, 0]
        else:
            emb = hidden[0].mean(dim=0)
        return emb.float().cpu()

    @torch.no_grad()
    def embed_batch(
        self,
        images: List[Union[str, Image.Image]],
        texts: List[str],
        pool: str = "mean",
    ) -> torch.Tensor:
        """Embed a list of (image, text) pairs; returns [N, H] on CPU."""
        return torch.stack([
            self.embed(img, txt, pool) for img, txt in zip(images, texts)
        ])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _user_content(image, text: str) -> List[Dict]:
        """Build a user-turn content list. `image` may be None (text-only tasks
        like SST-2); in that case no image token is emitted."""
        content = []
        if image is not None:
            content.append({"type": "image", "image": _load_image(image)})
        content.append({"type": "text", "text": text})
        return content

    def _build_messages(
        self,
        icd_list: List[Dict],
        query_image: Optional[Union[str, Image.Image]],
        query_text: str,
    ) -> List[Dict]:
        """Construct the Qwen-VL chat message list from ICDs + query.

        Task-agnostic: each ICD/query may or may not carry an image. VQA &
        Captioning have images; SST-2 (text classification) does not.
        """
        messages = []
        for icd in icd_list:
            messages.append({
                "role": "user",
                "content": self._user_content(icd.get("image"), icd["question"]),
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": str(icd["answer"])}],
            })
        messages.append({
            "role": "user",
            "content": self._user_content(query_image, query_text),
        })
        return messages

    def _flatten_base_messages(self, messages: List[Dict]) -> Tuple[str, List[Image.Image]]:
        """Base Qwen2-VL has no chat template; build vision+text prompt manually."""
        parts: List[str] = []
        images: List[Image.Image] = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, str):
                parts.append(content)
                continue
            chunk = ""
            for item in content:
                if item.get("type") == "image":
                    chunk += "<|vision_start|><|image_pad|><|vision_end|>"
                    img = item["image"]
                    images.append(
                        img if isinstance(img, Image.Image) else _load_image(img)
                    )
                elif item.get("type") == "text":
                    chunk += item["text"]
            if msg.get("role") == "assistant":
                chunk = "\n" + chunk + "\n"
            parts.append(chunk)
        return "".join(parts), images

    def _prepare_inputs(
        self, messages: List[Dict]
    ) -> Tuple[Dict, List[Image.Image]]:
        """Tokenise messages and move tensors to model device."""
        if self._instruct:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            images: List[Image.Image] = []
            for msg in messages:
                for item in msg.get("content", []):
                    if item.get("type") == "image":
                        img = item["image"]
                        images.append(
                            img if isinstance(img, Image.Image) else _load_image(img)
                        )
        else:
            text, images = self._flatten_base_messages(messages)

        inputs = self.processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
        )
        inputs = {
            k: (v.long() if k == "input_ids" else v).to(self._device)
            for k, v in inputs.items()
        }
        return inputs, images

    @staticmethod
    def _find_answer_span(
        input_ids: torch.Tensor,   # [seq_len]
        answer_ids: torch.Tensor,  # [ans_len]
    ) -> int:
        """Return the last start position of answer_ids in input_ids, or -1."""
        ans_len = answer_ids.size(0)
        seq_len = input_ids.size(0)
        for i in range(seq_len - ans_len, -1, -1):
            if (input_ids[i: i + ans_len] == answer_ids).all():
                return i
        return -1

    @staticmethod
    def _answer_log_prob(
        logits: torch.Tensor,      # [seq_len, vocab]
        input_ids: torch.Tensor,   # [seq_len]
        answer_ids: torch.Tensor,  # [ans_len]  (primary variant)
        answer_variants: Optional[list] = None,  # extra tokenisation variants
    ) -> torch.Tensor:
        """
        Compute sum of log P(token_t | prefix) for each token in answer_ids.

        Tries multiple tokenisation variants of the answer to handle the common
        case where the answer is encoded with a leading space/newline inside the
        model context but without one when tokenised standalone.

        Searches from the end of the sequence to target the query answer span
        (not the ICD answer spans that appear earlier in the prompt).
        """
        log_probs = F.log_softmax(logits, dim=-1)   # [seq_len, vocab]

        def _score_span(a_ids: torch.Tensor):
            start = QwenVLScorer._find_answer_span(input_ids, a_ids)
            if start == -1:
                return None
            ans_len = a_ids.size(0)
            total = 0.0
            for k in range(ans_len):
                pred_pos = start - 1 + k
                if pred_pos < 0:
                    continue
                total += log_probs[pred_pos, a_ids[k]].item()
            return total

        # Try primary tokenisation first
        score = _score_span(answer_ids)
        if score is not None:
            return torch.tensor(score)

        # Fall back to alternative tokenisation variants
        if answer_variants:
            for alt in answer_variants:
                alt = alt.to(input_ids.device)
                score = _score_span(alt)
                if score is not None:
                    return torch.tensor(score)

        return torch.tensor(float("-inf"), device=logits.device)

    # ------------------------------------------------------------------
    # Beam-search data generation helper
    # ------------------------------------------------------------------

    def beam_score_sequences(
        self,
        candidate_pool: List[Dict],
        query_image: Union[str, Image.Image],
        query_text: str,
        gt_answer: str,
        beam_sequences: List[List[int]],
    ) -> List[float]:
        """
        Score multiple ICD sequences (from beam search) for the same query.

        Args:
            candidate_pool:  Full candidate list (indexed by beam_sequences).
            query_image:     Query image.
            query_text:      Query question.
            gt_answer:       Ground-truth answer.
            beam_sequences:  List of sequences, each a list of indices into
                             candidate_pool.

        Returns:
            List of float scores (log-probs), one per beam.
        """
        scores = []
        for seq in beam_sequences:
            icd_list = [candidate_pool[i] for i in seq]
            score = self.score(icd_list, query_image, query_text, gt_answer)
            scores.append(score)
        return scores
