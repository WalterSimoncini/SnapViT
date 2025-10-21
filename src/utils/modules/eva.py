import torch
import torch.nn.functional as F

from timm.layers import apply_rot_embed_cat


def eva_forward(self, x, rope=None, attn_mask=None):
    """Custom EVA attention forward pass that is agnostic to the number of heads."""
    B, N, _ = x.shape

    if self.qkv is not None:
        if self.q_bias is None:
            qkv = self.qkv(x)
        else:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
            if self.qkv_bias_separate:
                qkv = self.qkv(x)
                qkv += qkv_bias
            else:
                qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    else:
        q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
        k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

    if rope is not None:
        npt = self.num_prefix_tokens
        q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2).type_as(v)
        k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2).type_as(v)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
    else:
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if attn_mask is not None:
            attn_mask = attn_mask.to(torch.bool)
            attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, -1)
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)

    return x
