from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
    TargetVideoStreamDataset,
    collect_target_videos,
    natural_key,
    resolve_dataset_root,
)
from utils import (
    build_prompt_tokens,
    compute_metrics,
    get_config_file,
    load_source_checkpoint,
    set_seed,
    write_json,
)


def unwrap(value):
    if torch.is_tensor(value):
        return value.item() if value.numel() == 1 else value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return unwrap(value[0])
    return value


def batch_item(value, index: int):
    """Return one item from a default-collated metadata field."""
    if torch.is_tensor(value):
        return unwrap(value[index])
    if isinstance(value, (list, tuple)):
        return unwrap(value[index])
    if index != 0:
        raise IndexError(
            f"Cannot index metadata value of type {type(value).__name__} "
            f"at position {index}."
        )
    return unwrap(value)


def entropy_of(probabilities: torch.Tensor) -> torch.Tensor:
    """Shannon entropy for a batch of categorical probabilities."""
    clipped = probabilities.clamp_min(1e-12)
    return -(clipped * clipped.log()).sum(dim=1)


# ---------------------------------------------------------------------------
# Per-video adaptive entropy statistics.
# ---------------------------------------------------------------------------


class RunningStats:
    """Online mean/variance tracker using Welford's algorithm."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self._m2 += delta * delta2

    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self._m2 / self.count

    def std(self) -> float:
        return math.sqrt(self.variance())


def adaptive_entropy_thresholds(
    stats: RunningStats,
    *,
    alpha: float,
    beta: float,
    warmup_windows: int,
    fallback_tau_positive: float,
    fallback_tau_negative: float,
) -> Tuple[float, float]:
    """
    Compute the video-adaptive thresholds

        tau_p = mu - alpha * sigma,
        tau_n = mu + beta  * sigma.

    Fixed thresholds are used during the first ``warmup_windows`` temporal
    representations of each video.
    """
    if stats.count < int(warmup_windows):
        return float(fallback_tau_positive), float(fallback_tau_negative)

    std = stats.std()
    tau_positive = max(0.0, stats.mean - float(alpha) * std)
    tau_negative = stats.mean + float(beta) * std

    if tau_positive >= tau_negative:
        tau_negative = tau_positive + 1e-6

    return tau_positive, tau_negative


# ---------------------------------------------------------------------------
# Positive and negative target caches.
# ---------------------------------------------------------------------------


@dataclass
class DynamicEntry:
    embedding: torch.Tensor
    entropy: float


@torch.no_grad()
def mean_featurewise_variance(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Mean feature-wise population variance used by the diversity gate:

        D(K) = (1/d) * sum_q Var({z_i[q]}_i).

    The diversity of an empty or one-element cache is defined as zero.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected embeddings with shape [N,D], got {tuple(embeddings.shape)}."
        )
    if embeddings.shape[0] <= 1:
        return embeddings.new_tensor(0.0)
    return embeddings.var(dim=0, unbiased=False).mean()


class DynamicCache:
    """
    Per-class fixed-capacity cache.

    All entries in a class partition contribute to its retrieval score. Cache
    updates combine reliability-based replacement with the diversity rule
    Delta D > 0:

    * Positive cache: when full, compare against the highest-entropy entry.
      The candidate must be more reliable (lower entropy) and increase class
      variance after replacement.
    * Negative cache: when full, compare against the lowest-entropy entry.
      The candidate must be more uncertain (higher entropy) and increase class
      variance after replacement.
    """

    def __init__(
        self,
        num_classes: int,
        capacity: int,
        device: torch.device,
        name: str,
    ) -> None:
        if name not in {"positive", "negative"}:
            raise ValueError("Cache name must be 'positive' or 'negative'.")

        self.num_classes = int(num_classes)
        self.capacity = max(0, int(capacity))
        self.device = device
        self.name = name
        self.buckets: Dict[int, List[DynamicEntry]] = {
            class_index: [] for class_index in range(self.num_classes)
        }

    def _entry(
        self,
        embedding: torch.Tensor,
        entropy: float,
    ) -> DynamicEntry:
        embedding = F.normalize(
            embedding.detach().float().to(self.device), dim=-1
        )
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        if embedding.ndim != 2 or embedding.shape[0] != 1:
            raise ValueError(
                "A cache entry must contain one embedding with shape [1,D]."
            )
        return DynamicEntry(embedding=embedding, entropy=float(entropy))

    @torch.no_grad()
    def class_scores(
        self,
        query: torch.Tensor,
        *,
        logit_scale: float,
    ) -> Tuple[torch.Tensor, List[bool]]:
        """
        Aggregate cosine similarities from all entries in every class bucket.

        The mean is used so that classes with fuller buckets do not receive a
        larger score solely because they contain more entries.
        """
        query = F.normalize(query.float().to(self.device), dim=-1)
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError("Cache retrieval expects one query with shape [1,D].")

        scores = torch.zeros(
            (1, self.num_classes), device=self.device, dtype=query.dtype
        )
        available: List[bool] = []

        for class_index in range(self.num_classes):
            bucket = self.buckets[class_index]
            if not bucket:
                available.append(False)
                continue

            keys = torch.cat([entry.embedding for entry in bucket], dim=0)
            keys = F.normalize(keys.float(), dim=-1)
            similarities = query @ keys.t()
            scores[0, class_index] = similarities.mean()
            available.append(True)

        return float(logit_scale) * scores, available

    @torch.no_grad()
    def evaluate_update(
        self,
        embedding: torch.Tensor,
        class_index: int,
        entropy: float,
    ) -> Dict:
        """
        Evaluate reliability and diversity before modifying a class bucket.

        The first entry initializes an empty bucket. For a non-full bucket, the
        candidate is appended only when it produces Delta D > 0. For a full
        bucket, entropy selects the reliability replacement candidate and the
        replacement is accepted only when it also produces Delta D > 0.
        """
        class_index = int(class_index)
        if class_index < 0 or class_index >= self.num_classes:
            raise IndexError(f"Invalid class index: {class_index}.")

        result = {
            "cache": self.name,
            "class_index": class_index,
            "pass": False,
            "reason": "disabled",
            "action": "none",
            "replacement_index": None,
            "cache_size_before": len(self.buckets[class_index]),
            "variance_before": 0.0,
            "variance_after": 0.0,
            "variance_gain": 0.0,
            "reliability_pass": False,
        }

        if self.capacity == 0:
            return result

        candidate = F.normalize(
            embedding.detach().float().to(self.device), dim=-1
        )
        if candidate.ndim == 1:
            candidate = candidate.unsqueeze(0)
        if candidate.ndim != 2 or candidate.shape[0] != 1:
            raise ValueError(
                "Diversity evaluation expects one embedding with shape [1,D]."
            )

        bucket = self.buckets[class_index]
        if not bucket:
            result.update(
                {
                    "pass": True,
                    "reason": "initialize_empty_bucket",
                    "action": "append",
                    "reliability_pass": True,
                }
            )
            return result

        keys = torch.cat([entry.embedding for entry in bucket], dim=0)
        keys = F.normalize(keys.float(), dim=-1)
        variance_before = float(mean_featurewise_variance(keys).item())

        if len(bucket) < self.capacity:
            trial = torch.cat([keys, candidate], dim=0)
            variance_after = float(mean_featurewise_variance(trial).item())
            variance_gain = variance_after - variance_before
            diversity_pass = variance_gain > 0.0

            result.update(
                {
                    "pass": bool(diversity_pass),
                    "reason": (
                        "positive_variance_gain"
                        if diversity_pass
                        else "non_positive_variance_gain"
                    ),
                    "action": "append",
                    "variance_before": variance_before,
                    "variance_after": variance_after,
                    "variance_gain": variance_gain,
                    "reliability_pass": True,
                }
            )
            return result

        if self.name == "positive":
            replacement_index = max(
                range(len(bucket)), key=lambda i: bucket[i].entropy
            )
            reliability_pass = float(entropy) < bucket[replacement_index].entropy
        else:
            replacement_index = min(
                range(len(bucket)), key=lambda i: bucket[i].entropy
            )
            reliability_pass = float(entropy) > bucket[replacement_index].entropy

        trial = keys.clone()
        trial[replacement_index : replacement_index + 1] = candidate
        variance_after = float(mean_featurewise_variance(trial).item())
        variance_gain = variance_after - variance_before
        diversity_pass = variance_gain > 0.0
        update_pass = reliability_pass and diversity_pass

        if not reliability_pass:
            reason = "reliability_replacement_failed"
        elif not diversity_pass:
            reason = "non_positive_variance_gain"
        else:
            reason = "reliable_positive_variance_gain"

        result.update(
            {
                "pass": bool(update_pass),
                "reason": reason,
                "action": "replace",
                "replacement_index": int(replacement_index),
                "variance_before": variance_before,
                "variance_after": variance_after,
                "variance_gain": variance_gain,
                "reliability_pass": bool(reliability_pass),
            }
        )
        return result

    def apply_update(
        self,
        embedding: torch.Tensor,
        class_index: int,
        entropy: float,
        decision: Dict,
    ) -> Dict:
        """Apply an update that has already passed ``evaluate_update``."""
        result = {
            "cache": self.name,
            "class_index": int(class_index),
            "inserted": False,
            "action": "reject",
        }

        if self.capacity == 0:
            result["action"] = "disabled"
            return result
        if not bool(decision.get("pass", False)):
            result["action"] = "reject_gate"
            return result

        entry = self._entry(embedding, entropy)
        bucket = self.buckets[int(class_index)]
        action = str(decision.get("action", "none"))

        if action == "append":
            if len(bucket) >= self.capacity:
                result["action"] = "reject_full_bucket"
                return result
            bucket.append(entry)
            result.update(inserted=True, action="append")
            return result

        if action == "replace":
            replacement_index = decision.get("replacement_index")
            if replacement_index is None:
                result["action"] = "reject_missing_replacement"
                return result
            replacement_index = int(replacement_index)
            result["removed_entropy"] = bucket[replacement_index].entropy
            result["removed_index"] = replacement_index
            bucket[replacement_index] = entry
            result.update(inserted=True, action="replace")
            return result

        result["action"] = "reject_unknown_action"
        return result

    def sizes(self) -> List[int]:
        return [
            len(self.buckets[class_index])
            for class_index in range(self.num_classes)
        ]


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Online EB-CaP with CLIP-text-guided SGLD, class-wise sampled "
            "cache aggregation, fused-score entropy, and diversity-gated "
            "positive/negative target caches."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument(
        "--backbone",
        default=None,
        choices=["RN50", "ViT-B/16", "ViT-B/32"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shuffle-within-subject", action="store_true")
    parser.add_argument("--stream-seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--clip-len", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--target-subjects", default=None)

    # Methodologically fixed reset behavior. This deprecated argument is kept
    # only so older shell commands still parse; its value is ignored.
    parser.add_argument(
        "--dynamic-reset-scope",
        choices=["video", "subject"],
        default="subject",
        help=(
            "Deprecated compatibility option. EB-CaP always resets target "
            "caches per subject and entropy statistics per video."
        ),
    )

    # Online class-conditioned SGLD generation.
    parser.add_argument("--sgld-steps", type=int, default=20)
    parser.add_argument("--sgld-step-size", type=float, default=1e-2)
    parser.add_argument("--sgld-noise-scale", type=float, default=0.10)
    parser.add_argument(
        "--ebm-samples-per-class",
        type=int,
        default=3,
        help="Number of independent sampled embeddings per class and window.",
    )

    # Deprecated compatibility options from the previous implementation.
    parser.add_argument("--sgld-min-extra-steps", type=int, default=0)
    parser.add_argument("--sgld-max-extra-steps", type=int, default=0)

    # Target caches.
    parser.add_argument("--positive-capacity", type=int, default=5)
    parser.add_argument("--negative-capacity", type=int, default=4)
    parser.add_argument(
        "--cache-topk",
        type=int,
        default=0,
        help=(
            "Deprecated compatibility option. All available class entries "
            "contribute to cache scores."
        ),
    )

    # Logit-fusion weights.
    parser.add_argument("--ebm-logit-weight", type=float, default=1.0)
    parser.add_argument("--positive-logit-weight", type=float, default=1.0)
    parser.add_argument("--negative-logit-weight", type=float, default=1.0)

    # Adaptive entropy gate.
    parser.add_argument("--entropy-alpha", type=float, default=1.0)
    parser.add_argument("--entropy-beta", type=float, default=1.0)
    parser.add_argument("--entropy-warmup-windows", type=int, default=5)
    parser.add_argument(
        "--entropy-fallback-tau-positive", type=float, default=0.5
    )
    parser.add_argument(
        "--entropy-fallback-tau-negative", type=float, default=0.8
    )

    # Deprecated diversity options retained so earlier commands still parse.
    # The methodology uses only Delta D > 0.
    parser.add_argument("--diversity-min-variance-gain", type=float, default=0.0)
    parser.add_argument("--diversity-min-cosine-distance", type=float, default=0.0)
    parser.add_argument(
        "--diversity-min-centroid-similarity", type=float, default=-1.0
    )

    parser.add_argument("--save-metrics", default=None)
    parser.add_argument("--save-metrics-txt", default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Batched CLIP-text-guided SGLD and sampled-cache scoring.
# ---------------------------------------------------------------------------


def generate_text_guided_samples(
    seed_embeddings: torch.Tensor,
    text_prototypes: torch.Tensor,
    *,
    steps: int,
    step_size: float,
    noise_scale: float,
    samples_per_class: int,
) -> Dict[str, torch.Tensor]:
    """
    Generate ``samples_per_class`` embeddings per class for each batch item.

    The class-conditional energy is

        E(z, c) = - normalize(z)^T e_c,

    and each chain is initialized from the corresponding current target-video
    representation. The chain stops when its current CLIP prediction equals
    the conditioning class or after ``steps`` iterations.

    Returns:
        prototypes:     [B,C,M,D]
        reached_target: [B,C,M]
        predictions:    [B,C,M]
        hit_steps:      [B,C,M]
    """
    if steps < 1:
        raise ValueError("SGLD steps must be at least 1.")
    if step_size <= 0:
        raise ValueError("SGLD step size must be positive.")
    if noise_scale < 0:
        raise ValueError("SGLD noise scale must be non-negative.")
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be at least 1.")

    if seed_embeddings.ndim == 1:
        seed_embeddings = seed_embeddings.unsqueeze(0)
    if seed_embeddings.ndim != 2:
        raise ValueError(
            f"Expected seed embeddings [B,D], got {tuple(seed_embeddings.shape)}."
        )

    text_prototypes = F.normalize(text_prototypes.detach().float(), dim=-1)
    seed_embeddings = F.normalize(seed_embeddings.detach().float(), dim=-1)

    device = seed_embeddings.device
    batch_size, embedding_dim = seed_embeddings.shape
    num_classes = int(text_prototypes.shape[0])

    if text_prototypes.shape[1] != embedding_dim:
        raise ValueError(
            "Target and text embedding dimensions do not match: "
            f"{embedding_dim} vs {text_prototypes.shape[1]}."
        )

    class_grid = torch.arange(num_classes, device=device, dtype=torch.long)
    target_classes = (
        class_grid.view(1, num_classes, 1)
        .expand(batch_size, num_classes, samples_per_class)
        .reshape(-1)
    )

    generated = (
        seed_embeddings[:, None, None, :]
        .expand(batch_size, num_classes, samples_per_class, embedding_dim)
        .reshape(-1, embedding_dim)
        .clone()
    )

    num_chains = int(generated.shape[0])
    active = torch.ones(num_chains, device=device, dtype=torch.bool)
    reached_target = torch.zeros(num_chains, device=device, dtype=torch.bool)
    hit_steps = torch.full(
        (num_chains,), -1, device=device, dtype=torch.long
    )

    for step_index in range(int(steps)):
        generated_var = generated.detach().requires_grad_(True)
        normalized = F.normalize(generated_var, dim=-1)

        class_scores = normalized @ text_prototypes.t()
        target_scores = class_scores.gather(
            1, target_classes.unsqueeze(1)
        ).squeeze(1)
        energies = -target_scores
        objective = (energies * active.float()).sum()

        gradient = torch.autograd.grad(
            objective, generated_var, only_inputs=True
        )[0]

        with torch.no_grad():
            noise = torch.randn_like(generated_var)
            proposal = (
                generated_var
                - 0.5 * float(step_size) * gradient
                + math.sqrt(float(step_size))
                * float(noise_scale)
                * noise
            )
            proposal = F.normalize(proposal, dim=-1)
            generated = torch.where(
                active.unsqueeze(1), proposal, generated_var.detach()
            )

            predictions = (generated @ text_prototypes.t()).argmax(dim=1)
            newly_reached = active & predictions.eq(target_classes)

            if newly_reached.any():
                reached_target[newly_reached] = True
                hit_steps[newly_reached] = step_index + 1
                active[newly_reached] = False

            if not bool(active.any().item()):
                break

    with torch.no_grad():
        prototypes = F.normalize(generated.detach(), dim=-1)
        final_predictions = (prototypes @ text_prototypes.t()).argmax(dim=1)

        shape = (batch_size, num_classes, samples_per_class)
        prototypes = prototypes.view(*shape, embedding_dim)
        reached_target = reached_target.view(*shape)
        final_predictions = final_predictions.view(*shape)
        target_classes = target_classes.view(*shape)
        hit_steps = hit_steps.view(*shape)

    return {
        "prototypes": prototypes,
        "reached_target": reached_target,
        "target_classes": target_classes,
        "predictions": final_predictions,
        "hit_steps": hit_steps,
    }


@torch.no_grad()
def sampled_cache_scores(
    query_embeddings: torch.Tensor,
    sampled_prototypes: torch.Tensor,
    *,
    logit_scale: float,
) -> torch.Tensor:
    """
    Aggregate all sampled prototypes associated with every class.

    Args:
        query_embeddings:   [B,D]
        sampled_prototypes: [B,C,M,D]

    Returns:
        Class-wise sampled-cache scores [B,C], where each class score is the
        mean cosine similarity between the query and all M sampled embeddings
        conditioned on that class.
    """
    if query_embeddings.ndim != 2:
        raise ValueError("query_embeddings must have shape [B,D].")
    if sampled_prototypes.ndim != 4:
        raise ValueError("sampled_prototypes must have shape [B,C,M,D].")
    if query_embeddings.shape[0] != sampled_prototypes.shape[0]:
        raise ValueError("Batch sizes of queries and sampled prototypes differ.")

    queries = F.normalize(query_embeddings.float(), dim=-1)
    prototypes = F.normalize(sampled_prototypes.float(), dim=-1)
    similarities = torch.einsum("bd,bcmd->bcm", queries, prototypes)
    return float(logit_scale) * similarities.mean(dim=2)


# ---------------------------------------------------------------------------
# Online TTA loop.
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_online_tta(
    model,
    loader: Iterable,
    class_tokens: torch.Tensor,
    classnames: Sequence[str],
    *,
    sgld_kwargs: Dict,
    positive_capacity: int,
    negative_capacity: int,
    ebm_logit_weight: float,
    positive_logit_weight: float,
    negative_logit_weight: float,
    entropy_alpha: float,
    entropy_beta: float,
    entropy_warmup_windows: int,
    entropy_fallback_tau_positive: float,
    entropy_fallback_tau_negative: float,
) -> Dict:
    device = next(model.parameters()).device
    num_classes = len(classnames)

    text_prototypes = F.normalize(
        model.encode_adapted_text(class_tokens).float(), dim=-1
    )
    logit_scale = float(
        model.logit_scale.exp().float().clamp(max=100.0).item()
    )

    current_subject_id: Optional[str] = None
    current_video_key: Optional[Tuple[str, str]] = None
    positive_cache: Optional[DynamicCache] = None
    negative_cache: Optional[DynamicCache] = None
    entropy_stats = RunningStats()

    video_logit_sums: Dict[Tuple[str, str], torch.Tensor] = defaultdict(
        lambda: torch.zeros((1, num_classes), device=device)
    )
    video_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    video_ground_truth: Dict[Tuple[str, str], int] = {}

    gate_counts = defaultdict(int)
    update_counts = defaultdict(int)
    frame_records: List[Dict] = []
    ebm_generation_batch_ms: List[float] = []
    ebm_generation_per_window_ms: List[float] = []
    ebm_reached_count = 0
    ebm_chain_count = 0
    num_windows = 0
    num_batches = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for videos, targets, metadata in tqdm(
        loader,
        desc="Online EB-CaP (text-guided sampled cache + target caches)",
    ):
        num_batches += 1
        videos = videos.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = int(videos.shape[0])
        num_windows += batch_size

        # Frozen visual and temporal encoding.
        z_batch = F.normalize(model.encode_video(videos).float(), dim=-1)
        base_scores_batch = logit_scale * (z_batch @ text_prototypes.t())

        # Construct one temporary class-wise sampled cache per target window.
        if device.type == "cuda":
            torch.cuda.synchronize()
        generation_start = time.perf_counter()
        with torch.enable_grad():
            sampled_result = generate_text_guided_samples(
                z_batch,
                text_prototypes,
                **sgld_kwargs,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()

        batch_generation_ms = 1000.0 * (
            time.perf_counter() - generation_start
        )
        per_window_generation_ms = batch_generation_ms / max(1, batch_size)
        ebm_generation_batch_ms.append(batch_generation_ms)
        ebm_generation_per_window_ms.extend(
            [per_window_generation_ms] * batch_size
        )

        sampled_scores_batch = sampled_cache_scores(
            z_batch,
            sampled_result["prototypes"],
            logit_scale=logit_scale,
        )
        ebm_reached_count += int(
            sampled_result["reached_target"].sum().item()
        )
        ebm_chain_count += int(sampled_result["reached_target"].numel())

        # Sequential processing preserves online cache semantics inside a batch.
        for batch_index in range(batch_size):
            z = z_batch[batch_index : batch_index + 1]
            base_scores = base_scores_batch[batch_index : batch_index + 1]
            sampled_scores = sampled_scores_batch[
                batch_index : batch_index + 1
            ]

            subject_id = str(batch_item(metadata["subject_id"], batch_index))
            video_id = str(batch_item(metadata["video_id"], batch_index))
            frame_index = int(batch_item(metadata["frame_index"], batch_index))
            video_key = (subject_id, video_id)

            # Target caches persist across videos of the same subject.
            if subject_id != current_subject_id:
                current_subject_id = subject_id
                positive_cache = DynamicCache(
                    num_classes,
                    positive_capacity,
                    device,
                    "positive",
                )
                negative_cache = DynamicCache(
                    num_classes,
                    negative_capacity,
                    device,
                    "negative",
                )
                current_video_key = None

            # Entropy statistics are reset at the beginning of every video.
            if video_key != current_video_key:
                current_video_key = video_key
                entropy_stats = RunningStats()

            assert positive_cache is not None
            assert negative_cache is not None

            # Retrieve all available target-cache evidence before insertion.
            positive_scores, positive_available = positive_cache.class_scores(
                z, logit_scale=logit_scale
            )
            negative_scores, negative_available = negative_cache.class_scores(
                z, logit_scale=logit_scale
            )

            fused_scores = (
                base_scores
                + float(ebm_logit_weight) * sampled_scores
                + float(positive_logit_weight) * positive_scores
                - float(negative_logit_weight) * negative_scores
            )
            fused_probabilities = fused_scores.softmax(dim=1)
            fused_entropy = float(
                entropy_of(fused_probabilities).squeeze(0).item()
            )
            predicted_class = int(
                fused_probabilities.argmax(dim=1).item()
            )
            least_likely_class = int(
                fused_probabilities.argmin(dim=1).item()
            )

            video_logit_sums[video_key] += fused_scores
            video_counts[video_key] += 1
            video_ground_truth.setdefault(
                video_key, int(targets[batch_index].item())
            )

            tau_positive, tau_negative = adaptive_entropy_thresholds(
                entropy_stats,
                alpha=entropy_alpha,
                beta=entropy_beta,
                warmup_windows=entropy_warmup_windows,
                fallback_tau_positive=entropy_fallback_tau_positive,
                fallback_tau_negative=entropy_fallback_tau_negative,
            )
            entropy_stats.update(fused_entropy)

            if fused_entropy < tau_positive:
                entropy_state = "confident"
                assigned_class = predicted_class
                active_cache = positive_cache
            elif fused_entropy < tau_negative:
                entropy_state = "uncertain"
                assigned_class = least_likely_class
                active_cache = negative_cache
            else:
                entropy_state = "reject"
                assigned_class = None
                active_cache = None

            diversity_result: Dict = {
                "pass": False,
                "reason": "not_evaluated_rejected_entropy",
                "variance_before": None,
                "variance_after": None,
                "variance_gain": None,
                "reliability_pass": None,
                "replacement_index": None,
            }
            action = "no_update"

            if active_cache is not None and assigned_class is not None:
                gate_counts[f"{active_cache.name}_diversity_evaluated"] += 1
                diversity_result = active_cache.evaluate_update(
                    z,
                    assigned_class,
                    fused_entropy,
                )
                gate_counts[f"{active_cache.name}_diversity_pass"] += int(
                    bool(diversity_result["pass"])
                )

                update_result = active_cache.apply_update(
                    z,
                    assigned_class,
                    fused_entropy,
                    diversity_result,
                )
                update_counts[
                    f"{active_cache.name}_{update_result['action']}"
                ] += 1

                if update_result["inserted"]:
                    action = f"{active_cache.name}_update"
                else:
                    update_counts[
                        f"{active_cache.name}_reject_"
                        + str(diversity_result["reason"])
                    ] += 1
            else:
                update_counts["no_update_entropy_reject"] += 1

            gate_counts["confident"] += int(entropy_state == "confident")
            gate_counts["uncertain"] += int(entropy_state == "uncertain")
            gate_counts["reject"] += int(entropy_state == "reject")

            frame_records.append(
                {
                    "subject_id": subject_id,
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "batch_index": batch_index,
                    "predicted_class": predicted_class,
                    "least_likely_class": least_likely_class,
                    "assigned_cache_class": assigned_class,
                    "fused_entropy": fused_entropy,
                    "tau_positive": tau_positive,
                    "tau_negative": tau_negative,
                    "entropy_state": entropy_state,
                    "diversity": diversity_result,
                    "action": action,
                    "sampled_cache": {
                        "samples_per_class": int(
                            sampled_result["prototypes"].shape[2]
                        ),
                        "reached_target": sampled_result[
                            "reached_target"
                        ][batch_index]
                        .cpu()
                        .tolist(),
                        "hit_steps": sampled_result["hit_steps"][
                            batch_index
                        ]
                        .cpu()
                        .tolist(),
                    },
                    "cache_availability": {
                        "positive": positive_available,
                        "negative": negative_available,
                    },
                    "cache_sizes": {
                        "positive": positive_cache.sizes(),
                        "negative": negative_cache.sizes(),
                    },
                    "score_components": {
                        "target": base_scores.squeeze(0).cpu().tolist(),
                        "sampled": sampled_scores.squeeze(0).cpu().tolist(),
                        "positive": positive_scores.squeeze(0).cpu().tolist(),
                        "negative": negative_scores.squeeze(0).cpu().tolist(),
                        "fused": fused_scores.squeeze(0).cpu().tolist(),
                    },
                    "ebm_generation_batch_ms": batch_generation_ms,
                    "ebm_generation_ms_per_window_equivalent": (
                        per_window_generation_ms
                    ),
                }
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    all_true: List[int] = []
    all_pred: List[int] = []
    per_subject_true: Dict[str, List[int]] = defaultdict(list)
    per_subject_pred: Dict[str, List[int]] = defaultdict(list)
    per_video: List[Dict] = []

    for video_key in sorted(
        video_logit_sums,
        key=lambda key: (natural_key(key[0]), natural_key(key[1])),
    ):
        subject_id, video_id = video_key
        average_scores = video_logit_sums[video_key] / max(
            1, video_counts[video_key]
        )
        prediction = int(average_scores.argmax(dim=1).item())
        target = int(video_ground_truth[video_key])

        all_true.append(target)
        all_pred.append(prediction)
        per_subject_true[subject_id].append(target)
        per_subject_pred[subject_id].append(prediction)
        per_video.append(
            {
                "subject_id": subject_id,
                "video_id": video_id,
                "n_windows": video_counts[video_key],
                "y_true": target,
                "y_pred": prediction,
                "scores": average_scores.squeeze(0).cpu().tolist(),
            }
        )

    per_subject = {
        subject_id: {
            "N_videos": len(per_subject_true[subject_id]),
            **compute_metrics(
                per_subject_true[subject_id],
                per_subject_pred[subject_id],
                num_classes,
            ),
        }
        for subject_id in sorted(per_subject_true, key=natural_key)
    }
    overall = {
        "N_videos": len(all_true),
        **compute_metrics(all_true, all_pred, num_classes),
    }

    pass_rates = {
        "confident": 100.0
        * gate_counts["confident"]
        / max(1, num_windows),
        "uncertain": 100.0
        * gate_counts["uncertain"]
        / max(1, num_windows),
        "reject": 100.0 * gate_counts["reject"] / max(1, num_windows),
        "positive_diversity_given_confident": 100.0
        * gate_counts["positive_diversity_pass"]
        / max(1, gate_counts["positive_diversity_evaluated"]),
        "negative_diversity_given_uncertain": 100.0
        * gate_counts["negative_diversity_pass"]
        / max(1, gate_counts["negative_diversity_evaluated"]),
    }

    mean_batch_ebm_ms = sum(ebm_generation_batch_ms) / max(
        1, len(ebm_generation_batch_ms)
    )
    mean_window_ebm_ms = sum(ebm_generation_per_window_ms) / max(
        1, len(ebm_generation_per_window_ms)
    )

    return {
        "overall": overall,
        "per_subject": per_subject,
        "per_video": per_video,
        "frames": frame_records,
        "gate_pass_rates": pass_rates,
        "cache_updates": dict(update_counts),
        "ebm_sampling": {
            "chains": ebm_chain_count,
            "reached_conditioning_class": ebm_reached_count,
            "reach_rate_percent": 100.0
            * ebm_reached_count
            / max(1, ebm_chain_count),
        },
        "runtime": {
            "n_batches": num_batches,
            "n_windows": num_windows,
            "n_videos": len(per_video),
            "total_seconds": elapsed,
            "ms_per_batch": 1000.0 * elapsed / max(1, num_batches),
            "ms_per_window": 1000.0 * elapsed / max(1, num_windows),
            "mean_ebm_generation_ms_per_batch": mean_batch_ebm_ms,
            "mean_ebm_generation_ms_per_window_equivalent": (
                mean_window_ebm_ms
            ),
        },
        "settings": {
            "procedure": "ebcap_text_guided_classwise_sampled_cache",
            "energy": "E(z,c) = -normalize(z)^T e_c",
            "sampled_cache_aggregation": "mean cosine over all samples per class",
            "target_cache_aggregation": "mean cosine over all entries per class",
            "entropy_source": "fused probabilities",
            "fusion_level": "class-score/logit level",
            "fusion_equation": (
                "target + lambda_s*sampled + lambda_p*positive "
                "- lambda_n*negative"
            ),
            "target_cache_reset_scope": "subject",
            "entropy_stats_reset_scope": "video",
            "diversity": "mean feature-wise variance; accept Delta D > 0",
            "negative_cache_diversity_gate": True,
            "sgld_kwargs": sgld_kwargs,
            "positive_capacity": positive_capacity,
            "negative_capacity": negative_capacity,
            "ebm_logit_weight": ebm_logit_weight,
            "positive_logit_weight": positive_logit_weight,
            "negative_logit_weight": negative_logit_weight,
            "entropy_alpha": entropy_alpha,
            "entropy_beta": entropy_beta,
            "entropy_warmup_windows": entropy_warmup_windows,
            "entropy_fallback_tau_positive": (
                entropy_fallback_tau_positive
            ),
            "entropy_fallback_tau_negative": (
                entropy_fallback_tau_negative
            ),
        },
    }


def shuffle_records_within_subject(records: Sequence, *, seed: int) -> List:
    grouped: Dict[str, List] = defaultdict(list)
    for record in records:
        grouped[str(record.subject_id)].append(record)

    rng = random.Random(int(seed))
    shuffled: List = []
    for subject_id in sorted(grouped, key=natural_key):
        subject_records = list(grouped[subject_id])
        rng.shuffle(subject_records)
        shuffled.extend(subject_records)
    return shuffled


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        args.device
        if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    config = get_config_file(args.config, args.dataset)
    dataset_cfg = config.get("dataset", {})

    model, preprocess, checkpoint = load_source_checkpoint(
        args.source_checkpoint,
        device,
        backbone_override=args.backbone,
    )
    print(f"Loaded model class: {model.__class__.__name__}")

    classnames = list(checkpoint["classnames"])
    templates = list(
        checkpoint.get(
            "templates",
            dataset_cfg.get("prompt_templates", []),
        )
    )
    class_tokens = build_prompt_tokens(classnames, templates, device)

    class_folders = list(dataset_cfg.get("class_folders", classnames))
    dataset_root = resolve_dataset_root(
        args.target_root,
        dataset_cfg.get("dataset_dir"),
    )
    target_records = collect_target_videos(
        dataset_root,
        class_folders,
        classnames,
        subject_regex=str(dataset_cfg.get("target_subject_regex", r".+")),
        exclude_dirs=dataset_cfg.get(
            "exclude_target_dirs",
            ["train", "validation", "val", "test"],
        ),
    )

    if args.target_subjects:
        requested_subjects = [
            value.strip()
            for value in args.target_subjects.split(",")
            if value.strip()
        ]
        requested_set = set(requested_subjects)
        target_records = [
            record
            for record in target_records
            if record.subject_id in requested_set
        ]
        found_subjects = {record.subject_id for record in target_records}
        missing_subjects = [
            subject
            for subject in requested_subjects
            if subject not in found_subjects
        ]
        if missing_subjects:
            raise RuntimeError(
                "Requested target subject folders not found: "
                f"{missing_subjects}. Dataset root: {dataset_root}"
            )
        if not target_records:
            raise RuntimeError(
                "No target videos remained after filtering subjects "
                f"{requested_subjects}."
            )

    stream_seed = args.seed if args.stream_seed is None else args.stream_seed
    if args.shuffle_within_subject:
        target_records = shuffle_records_within_subject(
            target_records,
            seed=stream_seed,
        )

    selected_subjects = sorted(
        {record.subject_id for record in target_records},
        key=natural_key,
    )

    if args.dynamic_reset_scope != "subject":
        print(
            "Warning: --dynamic-reset-scope is deprecated and ignored. "
            "Target caches are reset per subject, as specified by EB-CaP."
        )
    if args.cache_topk not in {0, None}:
        print(
            "Warning: --cache-topk is deprecated and ignored. All available "
            "entries in each class partition contribute to its cache score."
        )
    if args.sgld_min_extra_steps != 0 or args.sgld_max_extra_steps != 0:
        print(
            "Warning: extra SGLD steps after reaching the conditioning class "
            "are ignored; chains stop immediately, as specified in the method."
        )
    if (
        args.diversity_min_variance_gain != 0.0
        or args.diversity_min_cosine_distance != 0.0
        or args.diversity_min_centroid_similarity != -1.0
    ):
        print(
            "Warning: deprecated diversity thresholds are ignored. The "
            "methodological gate uses only a strictly positive variance gain."
        )

    print(f"Target dataset root: {dataset_root}")
    print(f"Target subjects: {selected_subjects}")
    print(f"Number of target videos: {len(target_records)}")
    print(f"DataLoader batch size: {args.batch_size}")
    print(
        "Energy: negative CLIP visual-text similarity, "
        "E(z,c) = -z^T e_c"
    )
    print(
        "Sampled cache: "
        f"{args.ebm_samples_per_class} embeddings per class and window; "
        "all class-wise samples contribute"
    )
    print(
        "Fusion: target scores + sampled-cache scores + positive-cache "
        "scores - negative-cache scores"
    )
    print(
        "Gates: fused-score adaptive entropy + positive variance gain for "
        "both positive and negative target caches"
    )
    print(
        "Reset scopes: target caches per subject; entropy statistics per video"
    )

    dataset = TargetVideoStreamDataset(
        target_records,
        preprocess,
        clip_len=args.clip_len,
        frame_stride=args.frame_stride,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    sgld_kwargs = {
        "steps": args.sgld_steps,
        "step_size": args.sgld_step_size,
        "noise_scale": args.sgld_noise_scale,
        "samples_per_class": args.ebm_samples_per_class,
    }

    results = run_online_tta(
        model,
        loader,
        class_tokens,
        classnames,
        sgld_kwargs=sgld_kwargs,
        positive_capacity=args.positive_capacity,
        negative_capacity=args.negative_capacity,
        ebm_logit_weight=args.ebm_logit_weight,
        positive_logit_weight=args.positive_logit_weight,
        negative_logit_weight=args.negative_logit_weight,
        entropy_alpha=args.entropy_alpha,
        entropy_beta=args.entropy_beta,
        entropy_warmup_windows=args.entropy_warmup_windows,
        entropy_fallback_tau_positive=(
            args.entropy_fallback_tau_positive
        ),
        entropy_fallback_tau_negative=(
            args.entropy_fallback_tau_negative
        ),
    )
    results["settings"]["batch_size"] = int(args.batch_size)
    results["settings"]["shuffle_within_subject"] = bool(
        args.shuffle_within_subject
    )
    results["settings"]["stream_seed"] = int(stream_seed)

    print("\nPer-subject results")
    for subject_id, values in results["per_subject"].items():
        print(
            f"  {subject_id}: N={values['N_videos']} | "
            f"WAR={values['WAR']:.2f} | F1={values['F1_macro']:.2f}"
        )

    overall = results["overall"]
    print(
        f"Overall: N={overall['N_videos']} | "
        f"WAR={overall['WAR']:.2f} | "
        f"F1={overall['F1_macro']:.2f}"
    )
    print(
        "Gate pass rates: "
        + ", ".join(
            f"{key}={value:.2f}%"
            for key, value in results["gate_pass_rates"].items()
        )
    )
    print(f"Cache updates: {results['cache_updates']}")
    print(f"EBM sampling: {results['ebm_sampling']}")
    print(
        f"Runtime: {results['runtime']['total_seconds']:.2f}s total, "
        f"{results['runtime']['ms_per_batch']:.2f} ms/batch, "
        f"{results['runtime']['ms_per_window']:.2f} ms/window "
        f"(EBM ~{results['runtime']['mean_ebm_generation_ms_per_batch']:.2f} "
        f"ms/batch; "
        f"~{results['runtime']['mean_ebm_generation_ms_per_window_equivalent']:.2f} "
        "ms/window equivalent)"
    )

    if args.save_metrics:
        write_json(args.save_metrics, results)
        print(f"Saved JSON metrics: {args.save_metrics}")

    if args.save_metrics_txt:
        path = Path(args.save_metrics_txt)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write("subject_id\tN_videos\tWAR\tF1_macro\n")
            for subject_id, values in results["per_subject"].items():
                handle.write(
                    f"{subject_id}\t{values['N_videos']}\t"
                    f"{values['WAR']:.4f}\t"
                    f"{values['F1_macro']:.4f}\n"
                )
            handle.write("\nOverall\n")
            handle.write(json.dumps(results["overall"], indent=2))
            handle.write("\n\nGate pass rates\n")
            handle.write(json.dumps(results["gate_pass_rates"], indent=2))
            handle.write("\n\nCache updates\n")
            handle.write(json.dumps(results["cache_updates"], indent=2))
            handle.write("\n\nEBM sampling\n")
            handle.write(json.dumps(results["ebm_sampling"], indent=2))
            handle.write("\n\nRuntime\n")
            handle.write(json.dumps(results["runtime"], indent=2))
            handle.write("\n\nSettings\n")
            handle.write(json.dumps(results["settings"], indent=2))
        print(f"Saved TXT metrics: {path}")


if __name__ == "__main__":
    main()