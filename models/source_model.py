from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .transformer import TemporalTransformer


class TextAdapter(nn.Module):
    """Small residual adapter that keeps the frozen CLIP text encoder intact."""

    def __init__(self, input_dim: int, output_dim: int | None = None, bottleneck: int = 64) -> None:
        super().__init__()
        output_dim = int(output_dim or input_dim)
        self.input_dim = int(input_dim)
        self.output_dim = output_dim
        self.residual_projection = (
            nn.Identity() if self.input_dim == self.output_dim else nn.Linear(self.input_dim, self.output_dim, bias=False)
        )
        self.adapter = nn.Sequential(
            nn.Linear(self.input_dim, int(bottleneck)),
            nn.GELU(),
            nn.Linear(int(bottleneck), self.output_dim),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        residual = self.residual_projection(text_features)
        return F.normalize(residual + self.adapter(text_features), dim=-1)


class SourceVideoModel(nn.Module):
    """
    Frozen CLIP encoders plus trainable temporal, text-adapter, and classifier modules.

    The classifier operates on the temporal video representation. Its normalized
    weight rows are therefore valid discriminative anchors in the same feature
    space used by target video embeddings during TTA.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        num_classes: int,
        embed_dim: int,
        temporal_layers: int = 4,
        temporal_heads: int = 8,
        temporal_ff: int = 2048,
        temporal_max_len: int = 256,
        temporal_dropout: float = 0.0,
        text_adapter_bottleneck: int = 64,
        freeze_visual: bool = True,
        freeze_text: bool = True,
    ) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.freeze_visual = bool(freeze_visual)
        self.freeze_text = bool(freeze_text)

        # Freeze the whole CLIP model first. The optional flags selectively
        # unfreeze an encoder; CLIP's logit scale remains fixed.
        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)
        if not self.freeze_visual:
            for parameter in self.clip_model.visual.parameters():
                parameter.requires_grad_(True)
        if not self.freeze_text:
            text_prefixes = (
                "transformer.",
                "token_embedding.",
                "ln_final.",
                "text_projection",
                "positional_embedding",
            )
            for name, parameter in self.clip_model.named_parameters():
                if name.startswith(text_prefixes):
                    parameter.requires_grad_(True)

        self.temporal = TemporalTransformer(
            input_dim=self.embed_dim,
            depth=int(temporal_layers),
            heads=int(temporal_heads),
            mlp_dim=int(temporal_ff),
            dim_head=self.embed_dim // int(temporal_heads),
            max_len=int(temporal_max_len),
            dropout=float(temporal_dropout),
        )
        self.text_adapter = TextAdapter(
            input_dim=self.embed_dim,
            output_dim=self.embed_dim,
            bottleneck=int(text_adapter_bottleneck),
        )
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.clip_model.logit_scale

    def _visual_context(self):
        return torch.no_grad() if self.freeze_visual else nullcontext()

    def _text_context(self):
        return torch.no_grad() if self.freeze_text else nullcontext()

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [B,T,C,H,W], got {tuple(video.shape)}")
        batch_size, time_steps, channels, height, width = video.shape
        flat = video.reshape(batch_size * time_steps, channels, height, width)
        with self._visual_context():
            frame_features = self.clip_model.encode_image(flat).float()
        frame_features = frame_features.reshape(batch_size, time_steps, -1)
        if frame_features.shape[-1] != self.embed_dim:
            raise ValueError(
                f"CLIP image feature dimension {frame_features.shape[-1]} does not match embed_dim={self.embed_dim}."
            )
        return F.normalize(self.temporal(frame_features), dim=-1)

    def encode_adapted_text(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens may be [C,77] or an ensemble [C,M,77].
        ensemble_shape = None
        if tokens.ndim == 3:
            ensemble_shape = tokens.shape[:2]
            tokens = tokens.reshape(-1, tokens.shape[-1])
        elif tokens.ndim != 2:
            raise ValueError(f"Expected text tokens [C,L] or [C,M,L], got {tuple(tokens.shape)}")
        with self._text_context():
            text_features = self.clip_model.encode_text(tokens).float()
        if text_features.shape[-1] != self.embed_dim:
            raise ValueError(
                f"CLIP text feature dimension {text_features.shape[-1]} does not match embed_dim={self.embed_dim}."
            )
        adapted = self.text_adapter(text_features)
        if ensemble_shape is not None:
            adapted = adapted.reshape(ensemble_shape[0], ensemble_shape[1], -1).mean(dim=1)
            adapted = F.normalize(adapted, dim=-1)
        return adapted

    def classifier_prototypes(self) -> torch.Tensor:
        return F.normalize(self.classifier.weight.float(), dim=-1)

    def forward(self, video: torch.Tensor, class_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        video_features = self.encode_video(video)
        text_features = self.encode_adapted_text(class_tokens)
        classifier_logits = self.classifier(video_features)
        scale = self.logit_scale.exp().float().clamp(max=100.0)
        text_logits = scale * video_features @ text_features.t()
        return {
            "video_features": video_features,
            "text_features": text_features,
            "classifier_logits": classifier_logits,
            "text_logits": text_logits,
        }

    def adaptation_state_dict(self, include_clip: bool = False) -> Dict[str, torch.Tensor]:
        prefixes = ("temporal.", "text_adapter.", "classifier.")
        state = self.state_dict()
        if include_clip:
            return state
        return {key: value for key, value in state.items() if key.startswith(prefixes)}

    def load_adaptation_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> Tuple[list, list]:
        result = self.load_state_dict(state_dict, strict=False)
        if strict:
            required_prefixes = ("temporal.", "text_adapter.", "classifier.")
            required = {key for key in self.state_dict() if key.startswith(required_prefixes)}
            missing_required = sorted(required.intersection(result.missing_keys))
            if missing_required:
                raise RuntimeError(f"Checkpoint is missing required adaptation parameters: {missing_required[:10]}")
        return list(result.missing_keys), list(result.unexpected_keys)


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)
