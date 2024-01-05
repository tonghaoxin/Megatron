"""Megatron offload distributed optimizer."""

import time
import math
import torch

from megatron import get_args
from megatron import get_timers
from megatron import print_rank_0
from megatron.core import mpu, tensor_parallel
from megatron.utils import param_is_not_shared
from megatron.optimizer.chunk.chunk_mgr import ChunkMgr


from .optimizer import MixedPrecisionOptimizer, _zero_grad_group_helper
from .distrib_optimizer import DistributedOptimizer
from .clip_grads import clip_grad_norm_fp32

class OffloadDistributedOptimizer(DistributedOptimizer):
     """Distributed optimizer, for all data types (fp16, bf16, and fp32)."""

     def __init__(self, optimizer, clip_grad, log_num_zeros_in_grad,
                 params_have_main_grad, use_contiguous_buffers_in_local_ddp,
                 fp16, bf16, params_dtype, grad_scaler, models):
        """
        See top of class definition for argument descriptions.

        The steps in this method create the core mapping between DDP grad
        buffers, parameters, and parameter shard ranges, that is needed for
        converting between model param indexes and main parameter shard
        indexes. This method also updates the optimizer parameter groups
        with the newly created shards.
        """

        super().__init__(
            optimizer, clip_grad, log_num_zeros_in_grad,
            params_have_main_grad, use_contiguous_buffers_in_local_ddp,
            fp16, bf16, params_dtype, grad_scaler, models)

        self.shard_fp32_duplicate_grads = []
        self.copy_cpu_grads = []
        self.shard_fp16_groups_overlap = []
        self.offoad_overlap_init = False

     def get_duplicate_grads(self):
        """get duplicate grads"""
        grads = []
        for group in self.shard_fp32_duplicate_grads:
            for grad in group:
                grads.append(grad)
        return grads

     def zero_duplicate_grads(self):
        """zero duplicate grads"""
        self.shard_fp32_duplicate_grads = []

     def get_main_grads_for_grad_norm(self):
        """get duplicate grads for norm.
           grad should not be none.
           parameter should not be shared.
           should not be a replica due to tensor model parallelism.
        """
        params = self.get_parameters()
        grads = self.get_duplicate_grads()
        grads_for_norm = []
        for param, grad in zip (params, grads):
            grad_not_none = grad is not None
            is_not_shared = param_is_not_shared(param)

            is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(param)

            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(grad)

        return grads_for_norm

     def clip_grad_norm(self, clip_grad):
        """
        clip grads.

        """
        grads = self.get_duplicate_grads()
        grads_for_norm = self.get_main_grads_for_grad_norm()
        return clip_grad_norm_fp32(
            grads, grads_for_norm, clip_grad,
            model_parallel_group=self.get_model_parallel_group())

     def _collect_main_grad_data_for_unscaling(self):
        """
        Note: this should be equivalent to the float-16 optimizer's method,
        but writtent differently, so the two should be combined.
        """
        main_grads = []
        for group in self.shard_fp32_duplicate_grads:
            for grad in group:
                main_grads.append(grad.data)
        return main_grads


     def _copy_model_grads_to_main_grads(self):
        """
        Copy model grads to shard_fp32_duplicate_grads.

        """
        # Utility method for copying group grads.
        def copy_grads(model_groups, main_groups):
            """
            copy grads.

            """
            self.shard_fp32_duplicate_grads = []

            for model_group, main_group in zip(model_groups, main_groups):
                check_and_clip_grads_per_group = []
                for model_param, main_param in zip(model_group, main_group):
                    param_range_map = self.get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]

                    model_grad = model_param.main_grad
                    shard_model_param = model_param.view(-1) \
                        [param_range.start:param_range.end]
                    shard_model_grad = model_grad.view(-1) \
                        [param_range.start:param_range.end]

                    if main_param.device == model_param.device:
                        main_param.grad = None

                    shard_check_and_clip_grad = torch.as_tensor(shard_model_grad.float().data,
                                                                dtype=main_param.dtype,
                                                                device=model_param.device)

                    check_and_clip_grads_per_group.append(shard_check_and_clip_grad)
                self.shard_fp32_duplicate_grads.append(check_and_clip_grads_per_group)

        copy_grads(self.model_float16_groups, self.shard_fp32_from_float16_groups)

     def _copy_duplicate_grads_to_main_grads(self):
        """
        Copy duplicate grads to main grads.

        """
        args = get_args()
        # Utility method for copying group grads.
        def copy_group_grads(shard_duplicate_groups, shard_main_groups):
            """
            copy group grads.

            """
            for duplicate_groups, shard_main_group in zip(shard_duplicate_groups,
                                                     shard_main_groups):
                copy_cpu_grads_per_group = []
                for grad, shard_main_param in zip(duplicate_groups,
                                                         shard_main_group):
                    if shard_main_param.device == torch.device('cpu'):
                        if args.overlap_offload_optimizer:
                            copy_cpu_grads_per_group.append(grad)
                        else:
                            shard_main_param.grad.copy_(grad)
                    else:
                        shard_main_param.grad = grad

                if len(copy_cpu_grads_per_group):
                    self.copy_cpu_grads.append(copy_cpu_grads_per_group)

        copy_group_grads(self.shard_fp32_duplicate_grads,
                         self.shard_fp32_from_float16_groups)
        self.shard_fp32_duplicate_grads = []

     def _copy_main_params_to_model_params(self):
        """
        Copy main params to model params.

        Since this step is followed by an all-gather through the DDP's grad
        buffer, this method is responsible for copying the updated params
        from the main shards into the correct position in the grad buffer.
        """

        # Utility method for copying group params.
        def copy_group_params(shard_main_groups, model_groups):
            """
            Copy main params to model params.
            """
            args = get_args()
            for shard_main_group, model_group in zip(shard_main_groups,
                                                     model_groups):
                for shard_main_param, model_param in zip(shard_main_group,
                                                         model_group):
                    if args.overlap_offload_optimizer:
                        if shard_main_param.device == torch.device('cpu'):
                            continue
                    param_range_map = self.get_model_param_range_map(model_param)
                    world_range = param_range_map["gbuf_world"]

                    assert world_range.size == shard_main_param.nelement()

                    model_id, dtype = self.model_param_gbuf_map[model_param]
                    model_param_buffer = self.param_buffers[model_id][dtype]

                    shard_model_param = model_param_buffer.view(-1) \
                        [world_range.start:world_range.end]

                    shard_model_param.data.copy_(shard_main_param)

        # Copy shard groups to model groups.
        copy_group_params(self.shard_fp32_from_float16_groups,
                          self.model_float16_groups)

     def get_fp16_group_params(self):
        """
        Get fp16 params for overlap cpu adam.

        """
        for shard_main_group, model_group in zip(self.shard_fp32_from_float16_groups,
                                                   self.model_float16_groups):
            shard_fp16_per_group = []
            for shard_main_param, model_param in zip(shard_main_group,
                                                         model_group):
                    param_range_map = self.get_model_param_range_map(model_param)
                    world_range = param_range_map["gbuf_world"]

                    assert world_range.size == shard_main_param.nelement()

                    model_id, dtype = self.model_param_gbuf_map[model_param]
                    model_param_buffer = self.param_buffers[model_id][dtype]

                    shard_model_param = model_param_buffer.view(-1) \
                        [world_range.start:world_range.end]
                    shard_fp16_per_group.append(shard_model_param)
            self.shard_fp16_groups_overlap.append(shard_fp16_per_group)

     @torch.no_grad()
     def step(self, args, timers):
        """
        Exec step.

        """
        # Copy gradients from model params to main params.
        timers('optimizer-copy-to-main-grad', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self._copy_model_grads_to_main_grads()
        timers('optimizer-copy-to-main-grad').stop()

        if self.grad_scaler:

            # Unscale and check for inf/nan.
            timers('optimizer-unscale-and-check-inf', log_level=1).start(
                barrier=args.barrier_with_L1_time)
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()
            timers('optimizer-unscale-and-check-inf').stop()

            # We are done with scaling gradients
            # so we can update the loss scale.
            self.grad_scaler.update(found_inf_flag)

            # If we found inf/nan, skip the update.
            if found_inf_flag:
                self.zero_duplicate_grads()
                return False, None, None
        # Clip the main gradients.
        timers('optimizer-clip-main-grad', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        grad_norm = None
        if self.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.clip_grad)
        timers('optimizer-clip-main-grad').stop()

        # Count the zeros in the grads.
        timers('optimizer-count-zeros', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        num_zeros_in_grad = self.count_zeros() if \
                            self.log_num_zeros_in_grad else None
        timers('optimizer-count-zeros').stop()

        args = get_args()
        if not args.overlap_offload_optimizer:
            timers('copy-duplicate-grads-to-main-grads', log_level=1).start(
                barrier=args.barrier_with_L1_time)
            self._copy_duplicate_grads_to_main_grads()
            timers('copy-duplicate-grads-to-main-grads').stop()

        if not self.offoad_overlap_init:
            self.get_fp16_group_params()
            self.offoad_overlap_init = True

        # Step the optimizer.
        timers('optimizer-inner-step', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        if args.overlap_offload_optimizer:
            self.optimizer.step(self.shard_fp32_duplicate_grads, self.shard_fp16_groups_overlap)
        else:
            self.optimizer.step()

        timers('optimizer-inner-step').stop()

        # Update params from main params.
        timers('optimizer-copy-main-to-model-params', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self._copy_main_params_to_model_params()
        timers('optimizer-copy-main-to-model-params').stop()

        self.shard_fp32_duplicate_grads = []
        # Successful update.
        return True, grad_norm, num_zeros_in_grad
