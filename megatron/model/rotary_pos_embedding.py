# !/usr/bin/env python3
# coding=utf-8

# The following code has been taken from https://github.com/NVIDIA/NeMo/blob/ \
# 782b4e1652aaa43c8be390d9db0dc89544afa080/nemo/collections/nlp/modules/ \
# common/megatron/rotary_pos_embedding.py

import importlib.util
import torch

from torch import einsum, nn
import torch.nn.functional as F
from megatron.accelerator import get_accelerator

custom_profiler_decorator = get_accelerator().get_operator('custom_profiler_decorator')
custom_apply_rope_with_freqs = None
custom_apply_rope_no_freqs = None
custom_apply_rope_with_idx = None


__all__ = ['RotaryEmbedding',
           'RotaryEmbeddingWithPositionIds',
           'ComplexRotaryEmbedding',
           'apply_rotary_pos_emb',
           'apply_rotary_pos_emb_fused',
           'apply_rotary_pos_emb_index',
           'apply_rotary_pos_emb_index_fused']


def update_apply_rope_func():
    """update_apply_rope_func"""
    global custom_apply_rope_with_freqs
    global custom_apply_rope_no_freqs
    global custom_apply_rope_with_idx

    custom_apply_rope_with_freqs = get_accelerator().get_operator('custom_apply_rope_with_freqs')
    custom_apply_rope_no_freqs = get_accelerator().get_operator('custom_apply_rope_no_freqs')
    custom_apply_rope_with_idx = get_accelerator().get_operator('custom_apply_rope_with_idx')


class RotaryEmbedding(nn.Module):
    """
    Original RotaryEmbedding
    """

    def __init__(self, dim, seq_len_interpolation_factor=1):
        """__init__"""
        super().__init__()
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        if importlib.util.find_spec('einops') is None:
            raise RuntimeError("einops is required for Rotary Embedding")
        update_apply_rope_func()

    @custom_profiler_decorator("RotaryEmbedding forward", color="indigo")
    def forward(self, max_seq_len, offset=0):
        """forward"""
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq) / self.seq_len_interpolation_factor, self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        from einops import rearrange
        return rearrange(emb, 'n d -> n 1 1 d')

    def extra_repr(self):
        """
        extra_repr
        Args:
        Return: str
        """
        return 'inv_freq shape={}'.format(self.inv_freq.size())


class RotaryEmbeddingWithPositionIds(nn.Module):
    """
    RotaryEmbedding with position ids
    """

    def __init__(self, dim, base=10000, precision=torch.half, learnable=False, seq_len_interpolation_factor=1):
        """
        __init__ function
        Args:
            dim:
            base:
            precision:
            learnable:
        Return:
        """
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        update_apply_rope_func()

    @custom_profiler_decorator("RotaryEmbeddingWithPositionIds forward", color="indigo")
    def forward(self, x, seq_dim=1, seq_len=None):
        """
        forward function
        Args:
            x:
            seq_dim:
            seq_len:
        Return: (Tensor, Tensor)
        """
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype) / self.seq_len_interpolation_factor
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()
            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


class ComplexRotaryEmbedding(nn.Module):
    """ComplexRotaryEmbedding"""
    def __init__(self, dim):
        """__init__"""
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        if importlib.util.find_spec('einops') is None:
            raise RuntimeError("einops is required for Rotary Embedding")

    def forward(self, max_seq_len):
        """forward"""
        t = torch.arange(max_seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis


def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = custom_apply_rope_with_freqs(t, freqs)
    return torch.cat((t, t_pass), dim=-1)


def apply_rotary_pos_emb_fused(t, freqs):
    """
    Fused RoPE function
    """
    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    t = custom_apply_rope_no_freqs(t, freqs.cos(), freqs.sin())
    return torch.cat((t, t_pass), dim=-1)


def apply_rotary_pos_emb_index(q, k, cos, sin, position_id, original=False):
    """
    RoPE function using position_id
    增加参数original, 默认False, 如果是True, 走论文里的逻辑, 即chatglm2的处理逻辑
    """
    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, 1, hn] -> [sq, b, hn]
    cos, sin = F.embedding(position_id, cos.squeeze()).unsqueeze(2), \
               F.embedding(position_id, sin.squeeze()).unsqueeze(2)

    q, k =  custom_apply_rope_with_idx(q, k, cos, sin, original = original)

    return q, k


def apply_rotary_pos_emb_index_fused(q, k, cos, sin, position_id):
    """
    Fused RoPE function using position_id
    """
    custom_rope = get_accelerator().get_operator('custom_rope')

    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
               F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    q = custom_apply_rope_no_freqs(q, cos, sin)
    k = custom_apply_rope_no_freqs(k, cos, sin)
    return q, k


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """reshape_for_broadcast"""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[0], x.shape[-1])
    shape = [d if i == 0 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_complex_rotary_pos_emb(t, freqs):
    """apply_complex_rotary_pos_emb"""
    t_ = torch.view_as_complex(t.float().reshape(*t.shape[:-1], -1, 2))
    freqs = reshape_for_broadcast(freqs, t_)
    t_out = torch.view_as_real(t_ * freqs).flatten(3)
    return t_out.type_as(t)
