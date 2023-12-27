# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.


import torch
import torch.nn as nn
from megatron.model.enums import AttnMaskType
from megatron.accelerator import get_accelerator


class FusedScaleMaskSoftmax(nn.Module):
    """
    fused operation: scaling + mask + softmax

    Arguments:
        input_in_fp16: flag to indicate if input in fp16 data format.
        input_in_bf16: flag to indicate if input in bf16 data format.
        attn_mask_type: attention mask type (pad or causal)
        scaled_masked_softmax_fusion: flag to indicate user want to use softmax fusion
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.
    """

    def __init__(
        self,
        input_in_fp16,
        input_in_bf16,
        attn_mask_type,
        scaled_masked_softmax_fusion,
        mask_func,
        softmax_in_fp32,
        scale,
    ):
        super(FusedScaleMaskSoftmax, self).__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        assert not (
            self.input_in_fp16 and self.input_in_bf16
        ), "both fp16 and bf16 flags cannot be active at the same time."
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale

        self.scaled_upper_triang_masked_softmax = get_accelerator().get_operator('scaled_upper_triang_masked_softmax')
        self.scaled_masked_softmax = get_accelerator().get_operator('scaled_masked_softmax')
        self.scaled_softmax = get_accelerator().get_operator('scaled_softmax')
        self.custom_softmax = get_accelerator().get_operator('custom_softmax')

        assert (
            self.scale is None or softmax_in_fp32
        ), "softmax should be in fp32 when scaled"

    def forward(self, input, mask):
        # [b, np, sq, sk]
        assert input.dim() == 4

        # interface for softmax kernel replacement
        if self.custom_softmax is not None:
            return self.forward_custom_softmax(input, mask)
        elif self.is_kernel_available(mask, *input.size()):
            return self.forward_fused_softmax(input, mask)
        else:
            return self.forward_torch_softmax(input, mask)

    def is_kernel_available(self, mask, b, np, sq, sk):
        attn_batches = b * np

        if (
            self.scaled_masked_softmax_fusion  # user want to fuse
            and self.input_in_float16  # input must be fp16
            and 16 < sk <= 4096  # sk must be 16 ~ 2048
            and sq % 4 == 0  # sq must be divisor of 4
            and sk % 4 == 0  # sk must be divisor of 4
            and attn_batches % 4 == 0  # np * b must be divisor of 4
        ):
            if 0 <= sk <= 4096:
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)

                if self.attn_mask_type == AttnMaskType.causal:
                    if attn_batches % batch_per_block == 0:
                        return True
                else:
                    if sq % batch_per_block == 0:
                        return True
        return False

    def forward_custom_softmax(self, input, mask):
        """forward_custom_softmax"""
        # TODO(changtao02): set self.custom_softmax to a list of funcs, including:
        # scaled_upper_triang_masked_softmax, scaled_masked_softmax, masked_softmax, etc.
        # if self.scaled_masked_softmax_fusion and self.scale is None:
        if self.custom_softmax.is_kernel_available(mask, *input.size(),
                                                   scaled_masked_softmax_fusion=self.scaled_masked_softmax_fusion,
                                                   input_in_float16=self.input_in_float16,
                                                   without_scale=self.scale is None):
            return self.custom_softmax(input, mask, self.scale)
        else:
            return self.forward_torch_softmax(input, mask)

    def forward_fused_softmax(self, input, mask):
        b, np, sq, sk = input.size()
        scale = self.scale if self.scale is not None else 1.0

        if self.attn_mask_type == AttnMaskType.causal:
            assert sq == sk, "causal mask is only for self attention"

            # input is 3D tensor (attn_batches, sq, sk)
            input = input.view(-1, sq, sk)
            probs = self.scaled_upper_triang_masked_softmax(input, scale)
            return probs.view(b, np, sq, sk)
        else:
            # input is 4D tensor (b, np, sq, sk)
            if mask is not None:
                return self.scaled_masked_softmax(input, mask, scale)
            else:
                return self.scaled_softmax(input, scale)

    def forward_torch_softmax(self, input, mask):
        if self.input_in_float16 and self.softmax_in_fp32:
            input = input.float()

        if self.scale is not None:
            input = input * self.scale
        mask_output = self.mask_func(input, mask) if mask is not None else input
        probs = torch.nn.Softmax(dim=-1)(mask_output)

        if self.input_in_float16 and self.softmax_in_fp32:
            if self.input_in_fp16:
                probs = probs.half()
            else:
                probs = probs.bfloat16()

        return probs

    def get_batch_per_block(self, sq, sk, b, np):
        """get_batch_per_block"""
        return self.scaled_masked_softmax.get_batch_per_block(sq, sk, b, np)

    def extra_repr(self):
        """
        extra_repr
        Args:
        Return: str
        """
        return ("input_in_float16={}, attn_mask_type={}, "
                "scaled_masked_softmax_fusion={}, "
                "softmax_in_fp32={}, scale={}"
                ).format(self.input_in_float16, self.attn_mask_type,
                         self.scaled_masked_softmax_fusion, self.softmax_in_fp32,
                         self.scale)
