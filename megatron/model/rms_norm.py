"""RMSNorm."""
import torch

from megatron.accelerator import get_accelerator


class RMSNorm(torch.nn.Module):
    """RMSNorm."""
    def __init__(self, dim, eps=1e-6, sequence_parallel=False, weight_empty=False):
        """
        Init method
        Args:
            dim:
            eps:
        Return:
        """
        super().__init__()
        self.eps = eps
        if weight_empty:
            self.weight = torch.nn.Parameter(torch.empty(dim))
        else:
            self.weight = torch.nn.Parameter(torch.ones(dim))
        self.sequence_parallel = sequence_parallel

        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)

    def _norm(self, x):
        """
        Internal norm func
        Args:
            x:
        Return:
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass
        Args:
            x:
        Return:
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self):
        """
        extra_repr
        Args:
        Return: str
        """
        return 'weight shape={}, eps={}'.format(self.weight.size(), self.eps)


# interface for RMSNorm
CustomRMSNorm = get_accelerator().get_operator('custom_rms_norm')

class FusedRMSNorm(CustomRMSNorm):
    """Fused RMS Norm"""
    def __init__(self, normalized_shape, eps=1e-5,
                 elementwise_affine=True,
                 sequence_parallel=False):

        super().__init__(normalized_shape,
                         eps=eps,
                         elementwise_affine=elementwise_affine)

        self.sequence_parallel = sequence_parallel

        # set sequence parallelism flag on weight parameters
        setattr(self.weight, 'sequence_parallel', self.sequence_parallel)
