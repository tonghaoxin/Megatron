# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# !/usr/bin/env python3
"""SFT Training"""

from megatron.training import pretrain


class SFTTrainer:
    """sft trainer"""
    def __init__(
            self,
            datasets_provider,
            model_provider,
            model_type,
            forward_step,
            args_defaults
    ):
        self.dataset_provider = datasets_provider
        self.model_provider = model_provider
        self.model_type = model_type
        self.forward_step = forward_step
        self.args_defaults = args_defaults

    def train(self):
        """train function for sft"""
        pretrain(
            self.dataset_provider,
            self.model_provider,
            self.model_type,
            self.forward_step,
            args_defaults=self.args_defaults
        )
