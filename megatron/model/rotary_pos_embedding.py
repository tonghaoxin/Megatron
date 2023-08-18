# !/usr/bin/env python3
# coding=utf-8

# The following code has been taken from https://github.com/NVIDIA/NeMo/blob/ \
# 782b4e1652aaa43c8be390d9db0dc89544afa080/nemo/collections/nlp/modules/ \
# common/megatron/rotary_pos_embedding.py

import importlib.util
import torch

from torch import einsum, nn
import torch.nn.functional as F

try:
    import nvtx
except ImportError:
    pass


__all__ = ['RotaryEmbedding', 'RotaryEmbeddingWithPositionIds',
           'apply_rotary_pos_emb', 'apply_rotary_pos_emb_fused',
           'apply_rotary_pos_emb_index', 'apply_rotary_pos_emb_index_fused']


class RotaryEmbedding(nn.Module):
    """
    Original RotaryEmbedding
    """

    def __init__(self, dim, seq_len_interpolation_factor=1):
        super().__init__()
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        if importlib.util.find_spec('einops') is None:
            raise RuntimeError("einops is required for Rotary Embedding")

    @nvtx.annotate("RotaryEmbedding forward", color="indigo")
    def forward(self, max_seq_len, offset=0):
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

    @nvtx.annotate("RotaryEmbeddingWithPositionIds forward", color="indigo")
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


class RotaryPositionalEmbeddingFunction(torch.autograd.Function):

    @staticmethod
    @nvtx.annotate("RotaryPositionalEmbeddingFunction forward", color="skyblue")
    def forward(ctx, q, cos, sin):
        import rotary_positional_embedding_cuda

        q_ = q.contiguous()
        cos_ = cos.contiguous()
        sin_ = sin.contiguous()
        output = rotary_positional_embedding_cuda.forward(*q.shape, q_, cos_, sin_)
        ctx.save_for_backward(cos_, sin_)

        return output

    @staticmethod
    @nvtx.annotate("RotaryPositionalEmbeddingFunction backward", color="violet")
    def backward(ctx, grad_output):
        import rotary_positional_embedding_cuda

        cos_, sin_ = ctx.saved_tensors
        grad_q = rotary_positional_embedding_cuda.backward(*grad_output.shape, grad_output, cos_, sin_)

        return grad_q, None, None


def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    from einops import rearrange
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def rotate_half(x):
    """
    another version of _rotate_half
    """
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    # dim=-1 triggers a bug in earlier torch versions
    return torch.cat((-x2, x1), dim=x1.ndim - 1)


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
    t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1)


def apply_rotary_pos_emb_fused(t, freqs):
    """
    Fused RoPE function
    """
    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    t = RotaryPositionalEmbeddingFunction.apply(t, freqs.cos(), freqs.sin())
    return torch.cat((t, t_pass), dim=-1)


def apply_rotary_pos_emb_index(q, k, cos, sin, position_id, original=False):
    """
    RoPE function using position_id
    增加参数original, 默认False, 如果是True, 走论文里的逻辑, 即chatglm2的处理逻辑
    """
    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, 1, hn] -> [sq, b, hn]
    cos, sin = F.embedding(position_id, cos.squeeze()).unsqueeze(2), \
               F.embedding(position_id, sin.squeeze()).unsqueeze(2)
    if original:
        q = q.reshape(q.size(0), -1, q.size(2), q.size(3)//2, 2)
        q = torch.cat([q[...,0], q[...,1]], dim=-1)
        k = k.reshape(k.size(0), -1, k.size(2), k.size(3)//2, 2)
        k = torch.cat([k[...,0], k[...,1]], dim=-1)
        q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
        s1, s2, s3, s4 = q.size(0), q.size(1), q.size(2), q.size(3)
        k1, k2, k3, k4 = k.size(0), k.size(1), k.size(2), k.size(3)
        q = torch.stack(torch.chunk(q, 2, -1), dim=-1)
        q = q.reshape(s1, s2, s3, s4)
        k = torch.stack(torch.chunk(k, 2, -1), dim=-1)
        k = k.reshape(k1, k2, k3, k4)
    else:
        q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k


def apply_rotary_pos_emb_index_fused(q, k, cos, sin, position_id):
    """
    Fused RoPE function using position_id
    """
    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
               F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
    q = RotaryPositionalEmbeddingFunction.apply(q, cos, sin)
    k = RotaryPositionalEmbeddingFunction.apply(k, cos, sin)
    return q, k
