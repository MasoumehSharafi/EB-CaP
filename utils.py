from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import yaml

from models import SourceVideoModel


def natural_key(value: str) -> List[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_config_file(config: str | Path, dataset_name: str) -> Dict:
    config_path = Path(config)
    if config_path.is_dir():
        config_path = config_path / f"{dataset_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_path}")
    return payload


def build_prompt_tokens(
    classnames: Sequence[str],
    templates: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    if not classnames:
        raise ValueError("At least one class name is required.")
    if not templates:
        raise ValueError("At least one prompt template is required.")
    import clip

    per_class = []
    for class_name in classnames:
        prompts = [template.format(class_name.replace("_", " ")) for template in templates]
        per_class.append(clip.tokenize(prompts))
    return torch.stack(per_class, dim=0).to(device)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> Dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target, prediction in zip(y_true, y_pred):
        if 0 <= int(target) < num_classes and 0 <= int(prediction) < num_classes:
            confusion[int(target), int(prediction)] += 1
    total = int(confusion.sum().item())
    if total == 0:
        return {"WAR": 0.0, "F1_macro": 0.0}
    true_positive = confusion.diag().float()
    support = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    precision = true_positive / predicted.clamp_min(1.0)
    recall = true_positive / support.clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "WAR": 100.0 * float(true_positive.sum().item()) / total,
        "F1_macro": 100.0 * float(f1.mean().item()),
    }


def global_energy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    return -temperature * torch.logsumexp(logits.float() / temperature, dim=1)


def quantile_thresholds(
    energies: torch.Tensor,
    positive_quantile: float,
    negative_quantile: float,
) -> Tuple[float, float]:
    if energies.numel() == 0:
        raise ValueError("Cannot calibrate thresholds from an empty energy tensor.")
    if not 0.0 <= positive_quantile < negative_quantile <= 1.0:
        raise ValueError("Energy quantiles must satisfy 0 <= positive < negative <= 1.")
    values = energies.detach().float().cpu()
    tau_positive = float(torch.quantile(values, positive_quantile).item())
    tau_negative = float(torch.quantile(values, negative_quantile).item())
    if tau_positive >= tau_negative:
        tau_negative = tau_positive + 1e-6
    return tau_positive, tau_negative


def source_model_config_from_args(args, embed_dim: int, num_classes: int) -> Dict[str, object]:
    return {
        "num_classes": int(num_classes),
        "embed_dim": int(embed_dim),
        "temporal_layers": int(args.temporal_layers),
        "temporal_heads": int(args.temporal_heads),
        "temporal_ff": int(args.temporal_ff),
        "temporal_max_len": int(args.temporal_max_len),
        "temporal_dropout": float(args.temporal_dropout),
        "text_adapter_bottleneck": int(args.text_adapter_bottleneck),
        "freeze_visual": not bool(args.unfreeze_visual),
        "freeze_text": not bool(args.unfreeze_text),
    }


def make_source_model(clip_model: torch.nn.Module, model_config: Mapping[str, object]) -> SourceVideoModel:
    return SourceVideoModel(
        clip_model=clip_model,
        num_classes=int(model_config["num_classes"]),
        embed_dim=int(model_config["embed_dim"]),
        temporal_layers=int(model_config.get("temporal_layers", 4)),
        temporal_heads=int(model_config.get("temporal_heads", 8)),
        temporal_ff=int(model_config.get("temporal_ff", 2048)),
        temporal_max_len=int(model_config.get("temporal_max_len", 256)),
        temporal_dropout=float(model_config.get("temporal_dropout", 0.0)),
        text_adapter_bottleneck=int(model_config.get("text_adapter_bottleneck", 64)),
        freeze_visual=bool(model_config.get("freeze_visual", True)),
        freeze_text=bool(model_config.get("freeze_text", True)),
    )


def save_source_checkpoint(
    path: str | Path,
    model: SourceVideoModel,
    *,
    backbone: str,
    classnames: Sequence[str],
    templates: Sequence[str],
    model_config: Mapping[str, object],
    energy_calibration: Mapping[str, float],
    training: Mapping[str, object],
    include_clip: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "backbone": str(backbone),
        "classnames": list(classnames),
        "templates": list(templates),
        "model_config": dict(model_config),
        "model_state": model.adaptation_state_dict(include_clip=include_clip),
        "contains_clip_state": bool(include_clip),
        "energy_calibration": dict(energy_calibration),
        "training": dict(training),
    }
    torch.save(payload, path)


def load_source_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    backbone_override: str | None = None,
) -> Tuple[SourceVideoModel, object, Dict]:
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(
            "Expected a source checkpoint produced by train_source.py with model_state/model_config metadata."
        )
    import clip

    backbone = backbone_override or str(payload.get("backbone", "ViT-B/32"))
    clip_model, preprocess = clip.load(backbone, device=str(device), jit=False)
    model = make_source_model(clip_model, payload["model_config"]).to(device)
    model.load_adaptation_state_dict(payload["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, preprocess, payload


def write_json(path: str | Path, payload: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
