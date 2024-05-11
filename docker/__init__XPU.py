#!/usr/bin/env python3
"""init"""
def load_module(mod):
    """load_module"""
    if mod == "adam":
        from apex.optimizers import FusedAdam
        return FusedAdam
    elif mod == "sgd":
        from apex.optimizers import FusedSGD
        return FusedSGD
    elif mod == "fused_weight_gradient_mlp_cuda":
        import fused_weight_gradient_mlp_cuda
        return fused_weight_gradient_mlp_cuda
    elif mod == "flashattn":
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
        return flash_attn_varlen_func
    elif mod == "multi_tensor_apply":
        from apex.multi_tensor_apply import multi_tensor_applier
        from amp_C import multi_tensor_l2norm
        from amp_C import multi_tensor_scale
        return multi_tensor_applier, multi_tensor_l2norm, multi_tensor_scale

def load(args):
    """load"""
    import parallel_attention
