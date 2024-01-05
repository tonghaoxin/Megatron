# Copyright (c) 2023, BAIDU CORPORATION.  All rights reserved.

"""Megatron ZeRO optimizer."""

from typing import List, Optional
import contextlib
import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch import distributed as dist

from .optimizer import MixedPrecisionOptimizer
from megatron import get_args
from megatron.core import mpu
from megatron.utils import print_rank_0, param_is_not_shared
from megatron.accelerator import get_accelerator


class GradientBuffer:
    """GradientBuffer"""

    def __init__(self, numel: int, dtype, device) -> None:
        """__init__"""
        self._buffer = torch.empty(numel,
                                   dtype=dtype,
                                   device=device)

    def get_real_buffer(self, numel):
        """get_real_buffer"""
        return self._buffer.narrow(0, 0, numel)


class GradientBufferPool:
    """GradientBufferPool"""

    def __init__(self, numel: int, dtype, device) -> None:
        """__init__"""
        self._numel = numel
        self._dtype = dtype
        self._device = device

        self._buffer_pool = set()

        self._created_buffer = int(0)

    def _create_buffer(self):
        """_create_buffer"""
        buffer = GradientBuffer(self._numel,
                                dtype=self._dtype,
                                device=self._device)

        self._created_buffer += 1
        print(f'Creating gradient buffer. Created: {self._created_buffer}')

        return buffer

    def get_buffer(self) -> torch.Tensor:
        """get_buffer"""
        if not self._buffer_pool:
            buffer = self._create_buffer()
        else:
            buffer = self._buffer_pool.pop()

        return buffer

    def return_buffer(self, buffer):
        """return_buffer"""
        self._buffer_pool.add(buffer)


class Bucket:
    """Bucket"""

    _grad_buffer_pool = None
    _largest_bucket_numel = int(0)

    def __init__(self,
                 param_list: List[torch.Tensor],
                 grad_dtype,
                 grad_device,
                 num_partitions,
                 dp_group,
                 dp_rank) -> None:
        """__init__"""
        # Create a new object, avoiding being modified outside
        self._param_list = list(param_list)
        self._grad_dtype = grad_dtype
        self._grad_device = grad_device
        self._num_partitions = num_partitions
        self._dp_group = dp_group
        self._dp_rank = dp_rank

        self._num_params = len(self._param_list)
        self._total_size = int(0)
        for p in self._param_list:
            self._total_size += p.numel()
        assert self._total_size % self._num_partitions == 0

        self._partition_size = self._total_size // self._num_partitions

        self._param_starting_position_map = {}
        self._assign_starting_position()
        self._borrowed_grad_buffer = None
        self._grad_buffer = None
        self._collected_grad = int(0)

        self._is_reduced = False

    def _assign_starting_position(self):
        """
        Only store the starting position for scattering param.grad,
        generate the index tensor on the fly to avoid storing a tensor with the same size as param
        """
        filled_size_in_partition = int(0)

        for p in self._param_list:
            self._param_starting_position_map[p] = filled_size_in_partition
            filled_size_in_partition += p.numel()

    def _init_grad_buffer(self):
        """_init_grad_buffer"""
        self._borrowed_grad_buffer = Bucket._grad_buffer_pool.get_buffer()
        self._grad_buffer = self._borrowed_grad_buffer.get_real_buffer(self._total_size)

    def is_param_in_bucket(self, param: torch.Tensor) -> bool:
        """is_param_in_bucket"""
        return True if param in self._param_starting_position_map else False

    def collect_param_grad(self, param: torch.Tensor) -> None:
        """collect_param_grad"""
        if self._grad_buffer is None:
            self._init_grad_buffer()

        assert self.is_param_in_bucket(param)
        assert param.grad is not None
        grad = param.grad if param.grad.dtype == self._grad_dtype \
            else param.grad.to(self._grad_dtype)
        grad = grad.view(-1)

        starting_position = self._param_starting_position_map[param]
        target = self._grad_buffer.narrow(0, starting_position, grad.numel())
        target.copy_(grad)

        # Keep it from being released until copied
        param.grad.record_stream(get_accelerator().current_stream())
        self._collected_grad += 1
        # Discard grad right away after copied to the buffer
        param.grad = None

    def is_all_grad_collected(self) -> bool:
        """is_all_grad_collected"""
        return self._collected_grad == self._num_params

    def reduce_scatter_grad(self, target_buffer):
        """reduce_scatter_grad"""
        assert self.is_all_grad_collected()
        self._grad_buffer.div_(self._num_partitions)

        # self._grad_buffer /= self._num_partitions
        dist.reduce_scatter_tensor(output=target_buffer,
                                            input=self._grad_buffer,
                                            group=self._dp_group,
                                            async_op=False)

        Bucket._grad_buffer_pool.return_buffer(self._borrowed_grad_buffer)
        self._borrowed_grad_buffer = None
        self._grad_buffer = None

    def reset(self):
        """reset"""
        self._grad_buffer = None
        self._borrowed_grad_buffer = None
        self._collected_grad = int(0)

    @staticmethod
    def setup_grad_buffer_pool(largest_bucket_numel: int, dtype, device):
        """setup_grad_buffer_pool"""
        if largest_bucket_numel <= Bucket._largest_bucket_numel:
            return

        print(f'Setting up GradientBufferPool with size: {largest_bucket_numel}')
        Bucket._largest_bucket_numel = largest_bucket_numel
        Bucket._grad_buffer_pool = GradientBufferPool(numel=largest_bucket_numel,
                                                      dtype=dtype,
                                                      device=device)


    def get_param_list(self) -> List[torch.Tensor]:
        """get_param_list"""
        return list(self._param_list)

    def get_total_size(self) -> int:
        """get_total_size"""
        return self._total_size

    def get_partitioned_size(self) -> int:
        """get_partitioned_size"""
        return self._partition_size


class AsyncHandler:
    """AsyncHandler"""
    def __init__(self) -> None:
        """__init__"""
        self._handle = None
        self._is_waited  = False

    # Can only handle one stream
    def wait(self):
        """wait"""
        assert self._handle is not None

        if self._is_waited:
            return

        self._is_waited = True
        return self._handle.wait()

    def reset_handle(self, handle):
        """reset_handle"""
        self._handle = handle
        self._is_waited = False


class AsyncHandlerManager:
    """AsyncHandlerManager"""
    def __init__(self, bucket_list: list) -> None:
        """__init__"""
        self._bucket_list = list(bucket_list)
        self._bucket_to_handler_map = {}
        self._param_to_handler_map = {}

        for bucket in self._bucket_list:
            async_handler = AsyncHandler()
            self._bucket_to_handler_map[bucket] = async_handler
            for param in bucket.get_param_list():
                self._param_to_handler_map[param] = async_handler

    def set_bucket_handle(self, bucket, handle):
        """set_bucket_handle"""
        async_handler = self._bucket_to_handler_map[bucket]
        async_handler.reset_handle(handle)

    def get_param_handle(self, param) -> AsyncHandler:
        """get_param_handle"""
        return self._param_to_handler_map[param]

class BucketAssignment:
    """BucketAssignment"""

    def __init__(self, param_list, bucket_size) -> None:
        """
        Assign each param to a bucket
        The order of param in the param_list should be the order of getting the grad
        """
        args = get_args()

        # Create a new object, avoiding modified outside
        self._param_list = list(param_list)
        self._bucket_size = bucket_size

        self._dp_world_size = mpu.get_data_parallel_world_size()
        self._dp_group = mpu.get_data_parallel_group()
        self._dp_rank = mpu.get_data_parallel_rank()
        self._device = param_list[0].device
        self._dtype = torch.float32 if args.accumulate_allreduce_grads_in_fp32 \
            else param_list[0].dtype
        self._duplicated_args = {'grad_dtype': self._dtype,
                           'grad_device': self._device,
                           'num_partitions': self._dp_world_size,
                           'dp_group': self._dp_group,
                           'dp_rank': self._dp_rank}

        self._buckets = []
        self._bucket_to_param = {}
        self._param_to_bucket = {}

        self._assign_buckets()
        print(f'Rank: {torch.distributed.get_rank()}, Partitioned Buckets: {len(self._buckets)}')
        if args.last_bucket_split_count is not None:
            self._split_last_bucket()
        else:
            print_rank_0("Splitting the last bucket is disabled")

    def _create_bucket(self, param_list, largest_bucket_numel, **kwargs):
        """_create_bucket"""
        bucket = Bucket(param_list=param_list, **kwargs)
        self._buckets.append(bucket)
        self._bucket_to_param[bucket] = list(param_list)
        for p in param_list:
            self._param_to_bucket[p] = bucket

        current_bucket_numel = bucket.get_total_size()
        return current_bucket_numel \
          if current_bucket_numel > largest_bucket_numel \
          else largest_bucket_numel

    def _split_last_bucket(self):
        """_split_last_bucket"""
        if len(self._buckets) == 0:
            return
        last_bucket = self._buckets[-1]
        if len(last_bucket.get_param_list()) == 1:
            print_rank_0(f"Unable to split the last bucket, because it contains only one param: \
                         {last_bucket.get_param_list()[0].numel()}")
            return
        print_rank_0(f"Splitting the last bucket, buckets: {len(self._buckets)}")
        self._buckets.pop()
        self._bucket_to_param.pop(last_bucket)
        for p in last_bucket.get_param_list():
            self._param_to_bucket.pop(p)
        args = get_args()
        split_count = args.last_bucket_split_count
        param_count = len(last_bucket.get_param_list())
        step = param_count // split_count
        for i in range(0, param_count, step):
            self._create_bucket(param_list=last_bucket.get_param_list()[i : i + step],
                                largest_bucket_numel=self._largest_bucket_numel, **self._duplicated_args)


    def _assign_buckets(self):
        """_assign_buckets"""
        current_assigned_numel = int(0)
        current_assigned_list = []
        largest_bucket_numel = int(0)
        for p in self._param_list:
            # Assign XXL param to its own bucket
            if p.numel() >= self._bucket_size:
                # Keep the order of the bucket against parameter buffer
                if current_assigned_list:
                    largest_bucket_numel = self._create_bucket(
                        param_list=current_assigned_list, largest_bucket_numel=largest_bucket_numel,
                        **self._duplicated_args)
                    current_assigned_list.clear()
                    current_assigned_numel = int(0)
                largest_bucket_numel = self._create_bucket(
                    param_list=[p], largest_bucket_numel=largest_bucket_numel, **self._duplicated_args)
            else:
                current_assigned_numel += p.numel()
                current_assigned_list.append(p)
                if current_assigned_numel >= self._bucket_size:
                    largest_bucket_numel = self._create_bucket(
                        param_list=current_assigned_list, largest_bucket_numel=largest_bucket_numel,
                        **self._duplicated_args)
                    current_assigned_list.clear()
                    current_assigned_numel = int(0)

        # The last bucket
        if current_assigned_list:
            largest_bucket_numel = self._create_bucket(
                param_list=current_assigned_list, largest_bucket_numel=largest_bucket_numel, **self._duplicated_args)

        Bucket.setup_grad_buffer_pool(largest_bucket_numel=largest_bucket_numel,
                                      dtype=self._dtype,
                                      device=self._device)

    def reset_buckets(self) -> None:
        """reset_buckets"""
        for bucket in self._buckets:
            bucket.reset()

    def get_param_bucket(self, param) -> Bucket:
        """get_param_bucket"""
        return self._param_to_bucket[param]

    def get_buckets(self) -> List[Bucket]:
        """get_buckets"""
        return list(self._buckets)


class Range:
    """Range"""

    def __init__(self, start: int, size: int) -> None:
        """__init__"""
        self.start = start
        self.size = size


class ParameterBuffer:
    """
    Create a contiguous buffer holding all params. i.e. self._flatted_buffer
    Create a contiguous buffer for partitioned params and grads. self._partitioned_param and self._partitioned_grad
    Implement gathering partitioned params back to the complete buffer.
    """

    def __init__(self,
                 param_list: List[torch.Tensor],
                 num_partitions: int,
                 rank: int,
                 dp_group,
                 verify_grad_order: bool,
                 overlap_param: bool=False) -> None:
        """__init__"""
        # Create a new object, avoiding modified outside
        self._param_list = list(param_list)
        self._num_partitions = num_partitions
        self._rank = rank
        self._dp_group = dp_group
        self._verify_grad_order = verify_grad_order

        self._flatted_buffer = _flatten_dense_tensors(self._param_list)
        assert self._flatted_buffer.numel() % self._num_partitions == 0
        self._make_params_to_be_view_of_buffer()

        self._bucket_ranges = {}
        self._bucket_partitioned_ranges = {}
        self._param_ranges = {}
        self._partition_size = self._flatted_buffer.numel() // self._num_partitions

        self._partitioned_param = None
        self._fp32_partitioned_param = None
        self._partitioned_grad = None
        self._partitioned_grad_receiving_buffer = None
        self._overlap_param = overlap_param

    def _make_params_to_be_view_of_buffer(self):
        """_make_params_to_be_view_of_buffer"""
        unflatted = _unflatten_dense_tensors(self._flatted_buffer,
                                             self._param_list)
        for param, view in zip(self._param_list, unflatted):
            param.data = view.data

    # Speed does not matter
    def scatter_flatted_param_to_partitioned_param(self, bucket_assignment: BucketAssignment):
        """scatter_flatted_param_to_partitioned_param"""
        target = self._partitioned_param
        start_in_flatted = int(0)
        start_in_partitioned = int(0)

        # For verification
        param_copied = int(0)
        for bucket in bucket_assignment.get_buckets():
            assert id(bucket.get_param_list()[0]) == id(
                self._param_list[param_copied])
            bucket_partition_size = bucket.get_partitioned_size()
            bucket_range = self._bucket_partitioned_ranges[bucket]
            bucket_size = bucket.get_total_size()
            start_in_bucket = bucket_partition_size * self._rank

            target_per_bucket = target.narrow(
                0, start_in_partitioned, bucket_range.size)
            source = self._flatted_buffer.narrow(
                0, start_in_flatted + start_in_bucket, target_per_bucket.numel())
            target_per_bucket.copy_(source)

            start_in_partitioned += target_per_bucket.numel()
            start_in_flatted += bucket_size
            param_copied += len(bucket.get_param_list())

    def _allocate_other_buffers(self):
        """_allocate_other_buffers"""
        self._fp32_partitioned_param = self._partitioned_param \
            if self._partitioned_param.dtype == torch.float32 \
            else self._partitioned_param.clone().detach().to(torch.float32)

        args = get_args()
        dtype = torch.float32 if args.accumulate_allreduce_grads_in_fp32 \
            else self._partitioned_param.dtype
        self._partitioned_grad = torch.zeros_like(
            self._partitioned_param, dtype=dtype)
        self._partitioned_grad_receiving_buffer = self._partitioned_grad.clone().detach()

    def _init_param_range(self, buckets, rank):
        """_init_param_range"""
        # All intervals are left-closed and right-open
        def construct_interval_params(bucket):
            interval_params = {}
            disp = 0
            for params in bucket.get_param_list():
                interval_params[params] = (disp, disp + params.numel())
                disp += params.numel()
            return interval_params

        def construct_interval_rank(rank, stride):
            return (rank * stride, (rank + 1) * stride)

        def is_intersect(interval1, interval2):
            return not (interval1[0] >= interval2[1] or interval1[1] <= interval2[0])

        def get_intersect_size(interval1, interval2):
            return min(interval1[1], interval2[1]) - max(interval1[0], interval2[0])

        param_ranges = {}

        outter_disp = 0
        for bucket in buckets:
            inner_disp = 0
            interval_params = construct_interval_params(bucket)
            interval_rank = construct_interval_rank(rank, stride=bucket.get_partitioned_size())
            for params in bucket._param_list:
                if is_intersect(interval_params[params], interval_rank):
                    intersect_size = get_intersect_size(interval_params[params], interval_rank)
                    param_ranges[params] = Range(outter_disp + inner_disp, intersect_size)
                    inner_disp += intersect_size
                else:
                    param_ranges[params] = None
            outter_disp += bucket.get_partitioned_size()

        return param_ranges

    def init_partitioned_buffer(self, bucket_assignment: BucketAssignment):
        """init_partitioned_buffer"""
        total_size = int(0)
        partitioned_total_size = int(0)
        for bucket in bucket_assignment.get_buckets():
            bucket_size = bucket.get_total_size()
            bucket_size_for_this_rank = bucket.get_partitioned_size()
            self._bucket_ranges[bucket] = Range(
                start=total_size, size=bucket_size)

            self._bucket_partitioned_ranges[bucket] = Range(
                start=partitioned_total_size, size=bucket_size_for_this_rank)

            total_size += bucket_size
            partitioned_total_size += bucket_size_for_this_rank


        self._param_ranges = self._init_param_range(bucket_assignment.get_buckets(), self._rank)

        self._partitioned_param = torch.empty(partitioned_total_size,
                                              device=self._flatted_buffer.device,
                                              dtype=self._flatted_buffer.dtype)

        print(f'Rank: {torch.distributed.get_rank()}, DP Rank: {self._rank}, '
              f'DP World Size: {self._num_partitions}, Partitioned Size: {partitioned_total_size}')
        self.scatter_flatted_param_to_partitioned_param(bucket_assignment)

        self._allocate_other_buffers()

    def update_partitioned_param(self, bucket_assignment: BucketAssignment):
        """update_partitioned_param"""
        self.scatter_flatted_param_to_partitioned_param(bucket_assignment)

        self._allocate_other_buffers()

    def get_bucket_receiving_buffer(self, bucket: Bucket):
        """get_bucket_receiving_buffer"""
        bucket_range = self._bucket_partitioned_ranges[bucket]
        # Assuming the grad has the same order with the buffer
        target = self._partitioned_grad_receiving_buffer.narrow(
            0, bucket_range.start, bucket_range.size)

        return target

    def accumulate_reduced_grad(self):
        """accumulate_reduced_grad"""
        self._partitioned_grad.add_(self._partitioned_grad_receiving_buffer)

    def bond_param_and_grad(self):
        """bond_param_and_grad"""
        self._fp32_partitioned_param.grad = self._partitioned_grad \
            if self._fp32_partitioned_param.dtype == self._partitioned_grad.dtype \
            else self._partitioned_grad.to(self._fp32_partitioned_param.dtype)

    def copy_updated_fp32_param(self):
        """copy_updated_fp32_param"""
        if id(self._fp32_partitioned_param) != id(self._partitioned_param):
            self._partitioned_param.copy_(self._fp32_partitioned_param)

    def gather_partitioned_param(self,
                                 bucket_assignment: BucketAssignment,
                                 async_handler_manager: AsyncHandlerManager=None):
        """gather_partitioned_param"""
        for bucket in bucket_assignment.get_buckets():
            bucket_range = self._bucket_ranges[bucket]
            bucket_partitioned_range = self._bucket_partitioned_ranges[bucket]
            target = self._flatted_buffer.narrow(
                0, bucket_range.start, bucket_range.size)
            source = self._partitioned_param.narrow(
                0, bucket_partitioned_range.start, bucket_partitioned_range.size)

            # 有同步改成异步
            if self._overlap_param:
                handle = torch.distributed.all_gather_into_tensor(
                    target, source, group=self._dp_group, async_op=True)

                if async_handler_manager is not None:
                    async_handler_manager.set_bucket_handle(bucket=bucket, handle=handle)
            else:
                torch.distributed.all_gather_into_tensor(target, source, group=self._dp_group)

    def zero_grad(self):
        """zero_grad"""
        self._partitioned_grad.zero_()

    def get_param_list(self) -> List[torch.Tensor]:
        """get_param_list"""
        return list(self._param_list)

    def get_partitioned_size(self):
        """get_partitioned_size"""
        return self._partitioned_param.numel()

    def get_partitioned_param(self):
        """get_partitioned_param"""
        return self._partitioned_param

    def get_fp32_partitioned_param(self):
        """get_fp32_partitioned_param"""
        return self._fp32_partitioned_param

    def get_partitioned_grad(self):
        """get_partitioned_grad"""
        return self._partitioned_grad


class ZeroOptimizer(MixedPrecisionOptimizer):
    """Zero optimizer, for all data types (fp16, bf16, and fp32).

    Arguments:
        optimizer: base optimizer such as Adam or SGD
        clip_grad: clip gradeints with this global L2 norm. Note
            that clipping is ignored if clip_grad == 0
        log_num_zeros_in_grad: return number of zeros in the gradients.
        params_have_main_grad: flag indicating if parameters have
            a `main_grad` field. If this is set, we are assuming
            that the model parameters are store in the `main_grad`
            field instead of the typical `grad` field. This happens
            for the DDP cases where there is a continuous buffer
            holding the gradients. For example for bfloat16, we want
            to do gradient accumulation and all-reduces in float32
            and as a result we store those gradients in the main_grad.
            Note that main grad is not necessarily in float32.
        use_contiguous_buffers_in_local_ddp: if true, the local DDP model
            is using a contiguous buffer to hold the model grads.
        fp16: if true, the model is running in fp16.
        bf16: if true, the model is running in bfloat16.
        grad_scaler: used for scaling gradients. Note that this can be
            None. This case happens when `bf16 = True` and we don't
            use any loss scale. Note that for `bf16 = True`, we can have
            a constnat gradient scaler. Also for `bf16 = False`, we
            always require a grad scaler.
        models: list of models (i.e., the virtual pipelining models). This
            is used by the distributed optimizer for mapping parameters.
        zero_stage: 1=ZeRO-1,2=ZeRO-2
    """
    _async_handler_manager = None
    _is_forward_prologue_ready_to_trigger = False
    _model_chunk_group_map = {}
    def __init__(self,
                 optimizer,
                 clip_grad,
                 log_num_zeros_in_grad,
                 params_have_main_grad,
                 use_contiguous_buffers_in_local_ddp,
                 fp16,
                 bf16,
                 params_dtype,
                 grad_scaler,
                 models,
                 zero_stage,
                 overlap_param=False):
        """__init__"""
        super().__init__(optimizer, clip_grad, log_num_zeros_in_grad, params_have_main_grad,
                         use_contiguous_buffers_in_local_ddp, fp16, bf16, params_dtype, grad_scaler, models)
        # assert len(models) == 1, \
        #         'ZeroOptimizer only supports one model for now'
        assert not self.use_contiguous_buffers_in_local_ddp, \
                'ZeroOptimizer use its own contiguous buffers instead of the ones of local ddp'

        self._args = get_args()
        self._dp_world_size = mpu.get_data_parallel_world_size()
        self._dp_rank = mpu.get_data_parallel_rank()
        self._dp_group = mpu.get_data_parallel_group()
        self._bucket_size = self._args.reduce_bucket_size

        self._bucket_assignment = []
        self._param_buffer = []  # each param_group has a ParameterBuffer

        self._reduction_stream = get_accelerator().Stream()
        self._require_backward_grad_sync = True
        self._zero_stage = zero_stage
        # All-reduce layernorm parameters across model parallel nodes
        # when sequence parallelism is used
        self._is_need_allreduce_layer_norm_grads = \
            mpu.get_tensor_model_parallel_world_size() > 1 and self._args.sequence_parallel

        if self._args.use_chunked_param_groups:
            for i in range(len(models)):
                ZeroOptimizer._model_chunk_group_map[i] = []

        bucket_list = []
        print(f"number of param_groups={len(self.optimizer.param_groups)}")
        for group_idx, param_group in enumerate(self.optimizer.param_groups):
            trainable_parameters = [
                param for param in param_group['params'] if param.requires_grad]

            for param in trainable_parameters:
                if param.numel() % self._dp_world_size != 0:
                    raise ValueError(f'ZeroOptimizer only supports \
                                     that every parameter.numel() is divisible by DataParallel world size. \n \
                                     parameter.numel(): {param.numel()}, DataParallel world size: \
                                     {self._dp_world_size}')

            param_buffer = ParameterBuffer(param_list=trainable_parameters,
                                           num_partitions=self._dp_world_size,
                                           rank=self._dp_rank,
                                           dp_group=self._dp_group,
                                           verify_grad_order=self._args.verify_grad_order,
                                           overlap_param=overlap_param)

            bucket_assignment = BucketAssignment(param_list=trainable_parameters,
                                                 bucket_size=self._bucket_size)

            param_buffer.init_partitioned_buffer(bucket_assignment)

            self._verify_assignment(
                param_buffer=param_buffer, bucket_assignment=bucket_assignment)

            self._param_buffer.append(param_buffer)
            self._bucket_assignment.append(bucket_assignment)
            bucket_list += bucket_assignment.get_buckets()

            param_group['params'] = [param_buffer.get_fp32_partitioned_param()]
            self._register_hooks(
                hook=self._grad_hook, param_list=trainable_parameters, group_idx=group_idx)

            if self._args.use_chunked_param_groups:
                ZeroOptimizer._model_chunk_group_map[param_group['model_chunk_id']].append(group_idx)

        self._overlap_param = overlap_param
        if self._overlap_param:
            ZeroOptimizer._async_handler_manager = AsyncHandlerManager(bucket_list)
            self._register_forward_async_hooks()

        # Add authentication processing.
        import parallel_attention
        self.update_func = parallel_attention.update

    def _verify_assignment(self, param_buffer: ParameterBuffer, bucket_assignment: BucketAssignment):
        """_verify_assignment"""
        buckets = bucket_assignment.get_buckets()
        assigned_params = []
        assigned_numel = int(0)
        for bucket in buckets:
            assigned_params += bucket.get_param_list()
            assigned_numel += bucket.get_partitioned_size()

        assert assigned_numel == param_buffer.get_partitioned_size(), \
            f'{assigned_numel} vs {param_buffer.get_partitioned_size()}'
        for src, target in zip(assigned_params, param_buffer.get_param_list()):
            assert id(src) == id(target)

    def _all_reduce_layer_norm_grads(self, param):
        """_all_reduce_layer_norm_grads"""
        if getattr(param, 'sequence_parallel', False):
            torch.distributed.all_reduce(param.grad, group=mpu.get_tensor_model_parallel_group())

    def _collect_grad(self, param, group_idx):
        """_collect_grad"""
        bucket = self._bucket_assignment[group_idx].get_param_bucket(param)
        bucket.collect_param_grad(param)

        if bucket.is_all_grad_collected():
            target_buffer = self._param_buffer[group_idx].get_bucket_receiving_buffer(bucket)
            bucket.reduce_scatter_grad(target_buffer)

    @torch.no_grad()
    def _grad_hook(self, param, group_idx):
        """_grad_hook"""
        # ZeRO-2 must perform gradient communication, regardless of whether gradient synchronization is enabled.
        if not self._require_backward_grad_sync and self._zero_stage == 1:
            return
        if self._is_need_allreduce_layer_norm_grads:
            self._all_reduce_layer_norm_grads(param=param)

        with get_accelerator().stream(self._reduction_stream):
            get_accelerator().current_stream().wait_stream(get_accelerator().default_stream())
            self._collect_grad(param=param, group_idx=group_idx)

    def _register_hooks(self, hook, param_list, group_idx):
        """_register_hooks"""
        self.grad_accs = []
        for param in param_list:
            if param.requires_grad:

                def wrapper(param, group_idx):
                    param_tmp = param.expand_as(param)
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]

                    def hook_wrapper(*notneeded):
                        hook(param, group_idx)

                    grad_acc.register_hook(hook_wrapper)
                    self.grad_accs.append(grad_acc)

                wrapper(param, group_idx)

    def _collect_main_grad_data_for_unscaling(self):
        """_collect_main_grad_data_for_unscaling"""
        main_grads = []
        for group in self.optimizer.param_groups:
            for param in group['params']:
                main_grads.append(param.grad.data)

        return main_grads

    @contextlib.contextmanager
    def no_sync(self):
        r"""
        A context manager to disable gradient synchronizations across DDP
        processes. Within this context, gradients will be accumulated on module
        variables, which will later be synchronized in the first
        forward-backward pass exiting the context.
        """
        old_require_backward_grad_sync = self._require_backward_grad_sync
        self._require_backward_grad_sync = False
        try:
            yield
        finally:
            self._require_backward_grad_sync = old_require_backward_grad_sync

    def get_model_parallel_group(self):
        """
        With the distributed optimizer, the model parallel group is the
        entire world.
        """
        return None

    def copy_updated_parameters(self):
        """copy_updated_parameters"""
        for param_buffer in self._param_buffer:
            param_buffer.copy_updated_fp32_param()

    def gather_parameters(self):
        """gather_parameters"""
        for param_buffer, bucket_assignment in \
                zip(self._param_buffer, self._bucket_assignment):
            param_buffer.gather_partitioned_param(bucket_assignment, self._async_handler_manager)

    @staticmethod
    def _forward_async_hook(module: torch.nn.Module, inp):
        """_forward_async_hook"""
        if not ZeroOptimizer._is_forward_prologue_ready_to_trigger:
            return

        is_leaf = False
        try:
            module.children().__next__()
        except StopIteration:
            is_leaf = True

        if is_leaf:
            async_handler_manager = \
                ZeroOptimizer.get_async_handler_manager()
            if async_handler_manager is None:
                return
            for param in module.parameters():
                async_handler_manager.get_param_handle(param).wait()

    def _register_forward_async_hooks(self):
        """_register_forward_async_hooks"""
        torch.nn.modules.module.register_module_forward_pre_hook(self._forward_async_hook)

    @torch.no_grad()
    def step(self, args, timers):
        """step"""
        # update state
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                self.update_func()

        get_accelerator().synchronize()

        # All-reduce embedding grads.
        timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self.allreduce_embedding_grads(args)
        timers('embedding-grads-all-reduce').stop()

        # Bond param and grad
        timers('optimizer-bond-param-and-grad', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        for param_buffer in self._param_buffer:
            param_buffer.bond_param_and_grad()
        timers('optimizer-bond-param-and-grad').stop()

        # Do unscale, check for inf, and update grad scaler only for
        # the case that grad scaler is provided.
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
                return False, None, None

        # Calculate grad norm across all groups
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

        # Step the optimizer.
        timers('optimizer-inner-step', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self.optimizer.step()
        timers('optimizer-inner-step').stop()

        timers('optimizer-copy-updated-parameters', log_level=1).start(
            barrier=args.barrier_with_L1_time)
        self.copy_updated_parameters()
        timers('optimizer-copy-updated-parameters').stop()

        if not self._overlap_param:
            timers('optimizer-gather-parameters', log_level=1).start(
                barrier=args.barrier_with_L1_time)
            self.gather_parameters()
            timers('optimizer-gather-parameters').stop()

        # Successful update.
        return True, grad_norm, num_zeros_in_grad

    def forward_prologue(self):
        """forward_prologue"""
        if not ZeroOptimizer._is_forward_prologue_ready_to_trigger \
                or not self._overlap_param:
            return

        self.gather_parameters()

    def backward_epilogue(self, model_chunk_id=-1):
        """backward_epilogue"""
        ZeroOptimizer._is_forward_prologue_ready_to_trigger = False

        # ZeRO-2 must perform gradient finishing work, regardless of whether synchronization is enabled
        if not self._require_backward_grad_sync and self._zero_stage == 1:
            return
        with get_accelerator().stream(self._reduction_stream):
            if model_chunk_id == -1 or self._args.use_chunked_param_groups:
                for bucket_assignment in self._bucket_assignment:
                    bucket_assignment.reset_buckets()

                for param_buffer in self._param_buffer:
                    param_buffer.accumulate_reduced_grad()
            else:
                group_ids = ZeroOptimizer._model_chunk_group_map[model_chunk_id]
                for group_id in group_ids:
                    self._bucket_assignment[group_id].reset_buckets()
                    self._param_buffer[group_id].accumulate_reduced_grad()

        if self._require_backward_grad_sync and model_chunk_id <= 0:
            ZeroOptimizer._is_forward_prologue_ready_to_trigger = True

    def zero_grad(self, set_to_none=True):
        """zero_grad"""
        for param_buffer in self._param_buffer:
            param_buffer.zero_grad()

    def _copy_model_params_to_main_params(self):
        """_copy_model_params_to_main_params"""
        """
        Copy model params to main params.

        During finetuning, this method is used to reload the main params from
        the model params.
        """
        for param_buffer, bucket_assignment in zip(self._param_buffer, self._bucket_assignment):
            param_buffer.update_partitioned_param(bucket_assignment)

        for param_buffer, param_group in zip(self._param_buffer, self.optimizer.param_groups):
            param_group['params'] = [param_buffer.get_fp32_partitioned_param()]

        get_accelerator().empty_cache()

    # Just keep the interface
    def reduce_model_grads(self, args, timers):
        """reduce_model_grads"""
        pass

    def gather_model_params(self, args, timers):
        """gather_model_params"""
        pass

    def state_dict(self):
        """
        The state dict contains all non-DP-rank-dependent (i.e., non-parameter-
        related) optimizer variables. The returned state dict can be stored in
        the standard model/RNG checkpoint file. The parameter and dependent
        optimizer state (e.g., exp_avg, exp_avg_sq) are stored in a separate
        checkpoint file by calling 'save_parameter_state()'.
        """
        state_dict = {}
        # Optimizer state (do not store parameter state here).
        state_dict['optimizer'] = {
            k : v
            for k, v in self.optimizer.state_dict().items()
            if k != "state"
        }
        for param_group in state_dict["optimizer"]["param_groups"]:
            del param_group["params"]

        # Grad scaler state.
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()

        state_dict = self.parameter_state_in_parallel(state_dict)
        return state_dict


    def load_state_dict(self, state_dict):
        """Load the state dict.

        As detailed in state_dict(), the state dict contains all non-
        parameter-related variables. This method is notably longer than
        state_dict(), because the Torch optimizers state has yet to be
        allocated at this point, and so we must do a cross referencing between
        the optimizers state (and the ordering it expects for parameter state)
        and this DP rank's shards. The optimizer at this point does not contain
        any tensor dimension information, so we must get these dimensions from
        the DP shards mapped during DistributedOptimizer.__init__().

        The tensor parameter state is loaded via load_parameter_state(), and
        so this method also must populate the loaded state dict with dummy
        tensor data (i.e., via torch.empty() below). This will be overwritten
        during load_parameter_state().

        ** Note: Torch optimizer's state structure. **
        The Torch optimizer stores its state in two levels. The top level is a
        list of groups, where each group contains a list of integer indexes
        (corresponding to parameters) that index into a master parameter list
        that is shared by all groups. As such, three values are necessary for
        maintaining this ordering:

        - group_index : The group to which a parameter belongs.
        - group_order : The index of a parameter within its group.
        - state_order : The index of a parameter within the shared parameter
            list.
        """

        # Get the Torch optimizer's state dict.
        # - This 'inner' optimizer at this point is unallocated, and only
        #   contains an integer ordering of parameters within each group, and
        #   the ordering of parameters within its flattened parameter state
        #   list.
        inner_state_dict = self.optimizer.state_dict()
        state_dict_param_groups = [{
            **group,
            "params" : list(inner_state_dict["param_groups"][idx]["params"]),
        } for idx, group in enumerate(state_dict["optimizer"]["param_groups"])]


        # Allocate 'dummy' data for optimizer state (i.e., torch.empty() below)
        # - Real data is overwritten during load_parameter_state().
        state_dict_state = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            num_elements = self._param_buffer[i].get_partitioned_size()
            init_shard = lambda : torch.empty(
                        num_elements,
                        dtype=torch.float32,
                        device=get_accelerator().current_device())

            state_dict_state.append((i, {
                "exp_avg" : init_shard(),
                "exp_avg_sq" : init_shard(),
            }))
        state_dict_state.sort(key=lambda s: s[0])
        state_dict_state = {s[0]: s[1] for s in state_dict_state}
        self.optimizer.load_state_dict({
            "state": state_dict_state,
            "param_groups": state_dict_param_groups,
        })

        # Grad scaler.
        if "grad_scaler" not in state_dict:
            if self.fp16:
                print_rank_0(
                    "***WARNING*** found an old checkpoint, will not " "load grad scaler ..."
                )
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
            else:
                print_rank_0(
                    "***WARNING*** fould the grad scaler in the "
                    "checkpoint but it is None in the class. "
                    "Skipping loading grad scaler ..."
                )
        self.load_parameter_state_in_parallel(state_dict)

    def parameter_state_in_parallel(self, state):
        """Save parameter state(i.e., parameter & optimizer tensors) via multiple ranks.

        This method performs three steps:
        - For each DP rank, copy param & optimizer shards to contiguous CPU
          buffers. (e.g., one buffer each for main_param, exp_avg, and
          exp_avg_sq).
        - Save world buffers to disk (i.e., distrib_opt.pt).
        """
        # Collect param states.
        for i, param_group in enumerate(self.optimizer.param_groups):
            inner_params = param_group["params"]
            # j is the dtype dimension in case there are multiple types in param_group
            assert len(inner_params) == 1, 'ZeroOptimizer only supports one dtype for now'
            inner_param = inner_params[0]
            optim_state = self.optimizer.state[inner_param]
            # copy to CPU buffer
            num_elements = self._param_buffer[i].get_partitioned_size()
            local_shards = {
                key: torch.empty(
                    num_elements,
                    dtype=torch.float32,
                    device="cpu",
                )
                for key in ("param", "exp_avg", "exp_avg_sq")
            }
            tensors = {
                "param": self._param_buffer[i].get_fp32_partitioned_param(),
                **optim_state,
            }
            for key in local_shards:
                local_shards[key].data.copy_(tensors[key].detach().cpu())

            state[i] = local_shards
        return state

    def load_parameter_state_in_parallel(self, loaded_state):
        """Load parameter state (i.e., parameter & optimizer tensors) via multiple ranks.

        This method performs the reverse of save_parameter_state_distributed():
        - Each rank loads world buffers from disk (i.e., distrib_opt.pt).
        - Copy param & optimizer shards from contiguous CPU
          buffers to GPU. (e.g., one buffer each for main_param, exp_avg, and
          exp_avg_sq).
        """
        for i, param_group in enumerate(self.optimizer.param_groups):
            inner_params = param_group["params"]
            assert len(inner_params) == 1, 'ZeroOptimizer only supports one dtype for now'
            inner_param = inner_params[0]
            num_elements = self._param_buffer[i].get_partitioned_size()
            local_shards = {
                key: torch.empty(num_elements, dtype=torch.float32, device="cpu")
                for key in ("param", "exp_avg", "exp_avg_sq")
            }
            for key in local_shards:
                local_shards[key] = loaded_state[i][key]

            # Copy from CPU buffer to GPU.
            optim_state = self.optimizer.state[inner_param]
            tensors = {
                # "param": self._param_buffer[i].get_fp32_partitioned_param(),
                "param": inner_param,
                **optim_state,
            }
            for key in local_shards:
                tensors[key].data.copy_(local_shards[key])

        self.copy_updated_parameters()
        self.gather_parameters()

    @staticmethod
    def get_async_handler_manager():
        """get_async_handler_manager"""
        return ZeroOptimizer._async_handler_manager

