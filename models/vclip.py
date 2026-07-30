import torch
from torch import nn

from .transformer import TemporalTransformer


class VClip(nn.Module):
    """
    Wrapper around an already-loaded CLIP model.
    Keeps CLIP frozen and applies a temporal transformer over frame embeddings.
    """

    def __init__(
        self,
        backbone,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_forward=2048,
        max_len=256,
        dropout=0.0,
        freeze_backbone=True,
    ):
        super().__init__()

        self.backbone = backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.temporal = TemporalTransformer(
            input_dim=d_model,
            depth=num_layers,
            heads=nhead,
            mlp_dim=dim_forward,
            dim_head=d_model // nhead,
            max_len=max_len,
            dropout=dropout,
        )

        # expose these so the rest of your code still works
        self.visual = self.backbone.visual
        self.logit_scale = self.backbone.logit_scale

    def encode_image(self, x):
        """
        Supports:
          x: [B, C, H, W]     -> normal CLIP image encoding
          x: [B, T, C, H, W]  -> frame-wise CLIP + temporal transformer
        """
        if x.ndim == 4:
            return self.backbone.encode_image(x)

        if x.ndim != 5:
            raise ValueError(f"Expected 4D or 5D image input, got shape {tuple(x.shape)}")

        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        feat = self.backbone.encode_image(x)   # [B*T, D]
        feat = feat.reshape(b, t, -1)          # [B, T, D]
        feat = self.temporal(feat)             # [B, D]
        return feat

    def encode_text(self, text):
        return self.backbone.encode_text(text)

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        scale = self.logit_scale.exp()
        logits_per_image = scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text