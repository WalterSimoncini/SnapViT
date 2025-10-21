import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from timm.models.vision_transformer import Attention


class DynamicHeadsAttention(Attention):
    """
        Simple attention wrapper to handle non-standard dimensions between
        the self-attention and the output projection
    """
    @classmethod
    def from_timm_attn(cls, attn: Attention) -> "DynamicHeadsAttention":
        attention = cls(
            dim=attn.head_dim * attn.num_heads,
            num_heads=attn.num_heads
        )

        attention.qkv = attn.qkv

        if hasattr(attn, "q_norm"):
            attention.q_norm = attn.q_norm

        if hasattr(attn, "k_norm"):
            attention.k_norm = attn.k_norm

        attention.attn_drop = attn.attn_drop
        attention.proj = attn.proj
        attention.proj_drop = attn.proj_drop

        return attention

    @classmethod
    def from_dino_attn(cls, attn: nn.Module) -> "DynamicHeadsAttention":
        attention = cls(
            dim=attn.qkv.in_features,
            num_heads=attn.num_heads
        )

        attention.qkv = attn.qkv
        attention.proj = attn.proj

        return attention

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.proj.weight.shape[-1])
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
