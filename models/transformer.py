import math
import torch
from torch import nn, einsum
from einops import rearrange


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [rearrange(t, "b n (h d) -> b h n d", h=h) for t in qkv]

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
            ])
            for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        depth,
        heads,
        mlp_dim,
        dim_head,
        max_len=256,
        dropout=0.0,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len + 1, input_dim) * 0.02)
        self.transformer = Transformer(
            dim=input_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: [B, T, D]
        b, t, d = x.shape
        if t + 1 > self.pos_embedding.shape[1]:
            raise ValueError(
                f"Temporal length {t} exceeds max_len={self.pos_embedding.shape[1] - 1}. "
                f"Increase max_len in TemporalTransformer."
            )

        cls = self.cls_token.expand(b, -1, -1)     # [B, 1, D]
        x = torch.cat([cls, x], dim=1)             # [B, T+1, D]
        x = x + self.pos_embedding[:, : t + 1]
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0]                             # CLS token only