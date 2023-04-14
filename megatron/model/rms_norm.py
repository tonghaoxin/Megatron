"""RMSNorm."""

import torch

class RMSNorm(torch.nn.Module):
    """RMSNorm."""
    def __init__(self, dim, eps=1e-6):
        """
        Init method
        Args:
            dim:
            eps:
        Return:
        """
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

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
