# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Megatron checkpoint实现类，针对megatron框架实现模型和优化器状态的保存逻辑
"""
import os
import time
import shutil
import psutil
import torch
import queue
import threading
import multiprocessing

from torch.utils.tensorboard import SummaryWriter

from megatron.core import mpu
from megatron.utils import (unwrap_model,
                            print_rank_0)
from megatron.global_vars import get_args
from megatron.checkpointing import (save_checkpoint,
                                    load_checkpoint,
                                    load_model_checkpoint_for_zero,
                                    load_optim_checkpoint_for_zero,
                                    get_rng_state,
                                    get_checkpoint_names,
                                    get_checkpoint_tracker_filename)
from megatron.accelerator import get_accelerator

PREFIX = "done_"
SUFFIX = ".txt"
ITER_PREFIX = 'iter_'

def _ensure_directory_exists(filename):
    """_ensure_directory_exists"""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def _mark_checkpoint_done(iteration_path):
    """_mark_checkpoint_done"""
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    filename = PREFIX + str(rank) + SUFFIX
    filepath = os.path.join(iteration_path, filename)
    with open(filepath, 'w') as f:
        pass

def _validate_checkpoint_done(iteration_path):
    """Validate checkpoint done."""
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    count = 0
    for filename in os.listdir(iteration_path):
        if filename.startswith(PREFIX) and filename.endswith(SUFFIX):
            count += 1
            os.remove(os.path.join(iteration_path, filename))
    return world_size == count

def _clean_checkpoint_done(iteration_path):
    """_clean_checkpoint_done"""
    for filename in os.listdir(iteration_path):
        if filename.startswith(PREFIX) and filename.endswith(SUFFIX):
            os.remove(os.path.join(iteration_path, filename))

def _get_iteration_path_for_megatron(checkpoints_path, iteration):
    """_get_iteration_path_for_megatron"""
    directory = '{}{:07d}'.format(ITER_PREFIX, iteration)
    iteration_path = os.path.join(checkpoints_path, directory)
    return iteration_path

def _validate_checkpoint_done_for_megatron(checkpoints_path):
    """Validate checkpoint done."""
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        if os.path.exists(checkpoints_path):
            tracker_filename = get_checkpoint_tracker_filename(checkpoints_path)
            if not os.path.exists(tracker_filename):
                folders = [folder for folder in os.listdir(checkpoints_path) if folder.startswith(ITER_PREFIX)]
                folders.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
                find_last_iteration = False
                for folder in folders:
                    path_name = os.path.join(checkpoints_path, folder)
                    if not find_last_iteration:
                        if _validate_checkpoint_done(path_name):
                            iteration = int(folder.split('_')[1])
                            with open(tracker_filename, 'w') as f:
                                f.write(str(iteration))
                            find_last_iteration = True
                    else:
                        _clean_checkpoint_done(path_name)
    # Make sure tracker_file has been written before each rank comes here.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


class CheckpointManager():
    """Checkpoint manager"""
    def __init__(self):
        """__init__"""
        self._memory_buffer = {}
        self._manager = multiprocessing.Manager()
        self._share_memory = self._manager.dict()
        self._saved_iterations = self._manager.list()
        self._memory_buffer_lock = multiprocessing.Lock()
        self._save_to_storage_event = multiprocessing.Event()
        self._save_to_storage_queue = multiprocessing.Queue(maxsize=1)
        self._finish_save_to_storage_event = multiprocessing.Event()
        self._finish_save_to_storage_queue = multiprocessing.Queue(maxsize=1)
        self._sync_save_to_storage_group = torch.distributed.new_group(backend="gloo")
        self._sync_save_to_storage_thread = threading.Thread(target=self._sync_save_to_storage, daemon=True)
        self._sync_save_to_storage_thread.start()
        self._ckpt_process = multiprocessing.Process(target=self._do_save_ckpt_to_storage, daemon=True)
        self._ckpt_process.start()
        self._ckpt_process_join_thread = threading.Thread(target=self._join_ckpt_process, daemon=False)
        self._ckpt_process_join_thread.start()
        self._save_to_storage_done = torch.Tensor([0])

        if get_accelerator().is_available():
            self._save_to_storage_done = get_accelerator().h2d_copy(self._save_to_storage_done)

    def _build_state_dict_buffer(self, state_dict):
        """Build state dict buffer."""
        def traversal_state_dict(value):
            if isinstance(value, dict):
                temp_dict = {}
                for k, v in value.items():
                    temp_dict[k] = traversal_state_dict(v)
                return temp_dict
            if isinstance(value, list):
                temp_list = []
                for item in value:
                    temp_list.append(traversal_state_dict(item))
                return temp_list
            if torch.is_tensor(value):
                return torch.empty_like(value.cpu(), pin_memory=True)
            return value
        state_dict_buffer = traversal_state_dict(state_dict)
        return state_dict_buffer

    def _build_memory_buffer(self, kwargs):

        """Build memory buffer."""
        for state_dict_name, state_dict_value in kwargs.items():
            if state_dict_value:
                self._memory_buffer[state_dict_name] = self._build_state_dict_buffer(state_dict_value)

    def _sync_save_to_storage(self):
        """Sync whether each rank has finished save_to_storage."""
        while self._finish_save_to_storage_event.wait():
            self._finish_save_to_storage_event.clear()
            try:
                torch.distributed.barrier(group=self._sync_save_to_storage_group)
                self._finish_save_to_storage_queue.put(True)
            except Exception as e:
                message = 'Exit exception when do barrier in _sync_save_to_storage: {}'.format(e)
                print_rank_0(message)
                self._finish_save_to_storage_queue.put(False)

    def _copy_state_to_buffer(self, source_state, target_buffer):
        """Copy state_dict to memory buffer."""
        def traversal_copy(state_dict, memory_buffer):
            if isinstance(state_dict, dict):
                for k, v in state_dict.items():
                    if isinstance(v, (dict, list)):
                        traversal_copy(state_dict[k], memory_buffer[k])
                    elif torch.is_tensor(v):
                        memory_buffer[k].copy_(state_dict[k])
                    else:
                        memory_buffer[k] = state_dict[k]
            elif isinstance(state_dict, list):
                for i in range(len(state_dict)): # pylint: disable=consider-using-enumerate
                    if isinstance(state_dict[i], (dict, list)):
                        traversal_copy(state_dict[i], memory_buffer[i])
                    elif torch.is_tensor(state_dict[i]):
                        memory_buffer[i].copy_(state_dict[i])
                    else:
                        memory_buffer[i] = state_dict[i]
        traversal_copy(source_state, target_buffer)

    def _save_to_memory_buffer(self, iteration, kwargs):
        """Write to memory buffer."""
        acq = self._memory_buffer_lock.acquire(block=True)
        self._save_to_storage_done[0] = 0 if acq else 1
        if torch.distributed.is_initialized():
            try:
                torch.distributed.all_reduce(self._save_to_storage_done, op=torch.distributed.ReduceOp.SUM)
            except Exception as e:
                message = 'Exit exception when do all_reduce in save_to_memory_buffer: {}'.format(e)
                print_rank_0(message)
        if self._save_to_storage_done.item() > 0:
            print_rank_0('Still save checkpoints to storage, skip save_to_memory_buffer.')
            self._save_to_storage_done[0] = 0
            if acq:
                self._memory_buffer_lock.release()
        else:
            # We assume that args.save_interval(the freqeunce of save_to_storage()) is not frequent,
            # and we know that the overhead of save_to_storage() is less than each train_step,
            # so we can write to pin_memory after at most few iterations been skipped.
            message = 'Start to save in memory checkpoint at iteration {}.'.format(iteration)
            print_rank_0(message)
            for state_dict_name, state_dict_value in kwargs.items():
                if state_dict_value and state_dict_name in self._memory_buffer:
                    self._copy_state_to_buffer(state_dict_value, self._memory_buffer[state_dict_name])
            self._memory_buffer["iteration"] = iteration
            for key, value in self._memory_buffer.items():
                self._share_memory[key] = value
            self._memory_buffer_lock.release()
            self._save_to_storage_event.set()
            message = 'Successfully saved in memory checkpoint at iteration {}.'.format(iteration)
            print_rank_0(message)

    def _save_to_storage(self, state_dict, save_path):
        """Save to storage."""
        if state_dict:
            _ensure_directory_exists(save_path)
            torch.save(state_dict, save_path)

    def _save_ckpt_to_storage(self):
        """Save checkopint to storage"""
        args = get_args()

        iteration = self._share_memory["iteration"]
        # Checkpoint file names.
        model_checkpoint_name, optim_checkpoint_name = get_checkpoint_names(args.save,
                                                                            iteration,
                                                                            args.use_distributed_optimizer)

        model_state_dict = {}
        if "model_state_dict" in self._share_memory:
            model_state_dict = self._share_memory["model_state_dict"]

        optim_state_dict = {}
        if "optim_state_dict" in self._share_memory:
            optim_state_dict = self._share_memory["optim_state_dict"]

        if args.use_distributed_optimizer:
            # Save model separate from optimizer.
            self._save_to_storage(model_state_dict, model_checkpoint_name)
            self._save_to_storage(optim_state_dict, optim_checkpoint_name)
        else:
            # Save model and optimizer together.
            state_dict = {**model_state_dict, **optim_state_dict}
            self._save_to_storage(state_dict, model_checkpoint_name)

        print_rank_0('Successfully saved checkpoints to storage at iteration {:7d} to {}'.format(iteration, args.save))

        # The write_to_storage thread need to know whether the in_memory_ckpt of this iteration has been saved.
        self._saved_iterations.append(iteration)

        # In Megatron, when the process of each rank has saved ckpt to storage,
        # latest_checkpointed_iteration.txt will be revised at args.save(checkpoints_path). Here, we
        # write to storage asyncly, so we can not use torch.distributed.barrier() to make sure each process
        # has finished. In our way, We will revise latest_checkpointed_iteration.txt before load_checkpoint().
        iteration_path = _get_iteration_path_for_megatron(args.save, iteration)
        _mark_checkpoint_done(iteration_path)

        # Start to wait other rank.
        self._finish_save_to_storage_event.set()
        sync_result = self._finish_save_to_storage_queue.get()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if sync_result and rank == 0:
            tracker_filename = get_checkpoint_tracker_filename(args.save)
            with open(tracker_filename, 'w') as f:
                f.write(str(iteration))

    def _do_save_ckpt_to_storage(self):
        """Do save checkpoint to storage"""
        while True:
            iteration = self._save_to_storage_queue.get()
            if iteration is None:
                print_rank_0('save_checkpoint_to_storage subprocess exit.')
                break
            self._save_to_storage_event.wait()
            self._save_to_storage_event.clear()
            if self._share_memory and  "iteration" in self._share_memory and \
                self._share_memory["iteration"] not in self._saved_iterations:
                with self._memory_buffer_lock:
                    message = 'Start to save checkpoints to storage {}.'.format(int(self._share_memory["iteration"]))
                    print_rank_0(message)
                    self._save_ckpt_to_storage()

    def _join_ckpt_process(self):
        """Wait for subprocess to exit"""
        if self._ckpt_process.is_alive():
            self._ckpt_process.join()
            if self._ckpt_process.exitcode is not None and self._ckpt_process.exitcode:
                os._exit(self._ckpt_process.exitcode)

    def save_ckpt_to_memory(self, iteration, **kwargs):
        """Save checkpoint to memory"""
        # Build memory buffer.
        if not self._memory_buffer:
            self._build_memory_buffer(kwargs)
        # Save checkpoint to memory buffer.
        self._save_to_memory_buffer(iteration, kwargs)

    def async_save_ckpt_to_storage(self, iteration):
        """Async save checkpoint to storage"""
        try:
            self._save_to_storage_queue.put(iteration, block=False)
        except queue.Full:
            message = 'Ready to save checkpoints to storage {}.'.format(iteration)
            print_rank_0(message)

    def save_checkpoint_for_megatron(self, iteration, model, optimizer, opt_param_scheduler):
        """Save checkpoint for megatron"""
        args = get_args()

        # Only rank zero of the data parallel writes to the storage.
        model = unwrap_model(model)

        print_rank_0('saving checkpoint at iteration {:7d} to {}'.format(
            iteration, args.save))

        # Collect rng state across data parallel ranks.
        rng_state = get_rng_state()

        # Collect args, model, RNG.
        model_state_dict = {}
        if not torch.distributed.is_initialized() or mpu.get_data_parallel_rank() == 0:
            # Arguments, iteration, and model.
            model_state_dict['args'] = args

            model_state_dict['checkpoint_version'] = 3.0
            model_state_dict['iteration'] = iteration
            if len(model) == 1:
                model_state_dict['model'] = model[0].state_dict_for_save_checkpoint()
            else:
                for i in range(len(model)):
                    mpu.set_virtual_pipeline_model_parallel_rank(i)
                    model_state_dict['model%d' % i] = model[i].state_dict_for_save_checkpoint()

            # RNG states.
            if not args.no_save_rng:
                model_state_dict["rng_state"] = rng_state

        # Collect optimizer state. (Optimizer is saved separately from the model, due
        # to the conflicting data pattern when using the distributed optimizer.)
        optim_state_dict = {}
        if not args.no_save_optim and (not torch.distributed.is_initialized()
            or mpu.get_data_parallel_rank() == 0 or args.use_distributed_optimizer):
            # Optimizer stuff.
            if optimizer is not None:
                optim_state_dict['optimizer'] = optimizer.state_dict()
            if opt_param_scheduler is not None:
                optim_state_dict['opt_param_scheduler'] = opt_param_scheduler.state_dict()

        self.save_ckpt_to_memory(iteration, model_state_dict=model_state_dict, optim_state_dict=optim_state_dict)

    def finish_ckpt_process(self):
        """Finish checkpoint subprocess
        """
        self._save_to_storage_queue.put(None, block=True)


ckpt_mgr = None
writer = None


def is_rank0():
    """Is rank0."""
    flag = False
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            flag = True
    else:
        flag = True
    return flag

def save_checkpoint_for_megatron(iteration, model, optimizer, opt_param_scheduler, release=False):
    """Save a model checkpoint."""
    args = get_args()

    start_time = time.time()
    if args.enable_accelerate_checkpoint is None or not args.enable_accelerate_checkpoint:
        save_checkpoint(iteration, model, optimizer, opt_param_scheduler, release)

        memory_used = (psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024 ** 3)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print_rank_0(f'sync save checkpoint at {iteration}, memory used {memory_used} GB, '
                     f'start timestamp {start_time}, end timestamp {end_time}, cost time {elapsed_time} seconds')
        return

    global writer
    if args.tensorboard_dir is not None and is_rank0():
        if writer is None:
            writer = SummaryWriter(args.tensorboard_dir)
    global ckpt_mgr
    if ckpt_mgr is None:
        ckpt_mgr = CheckpointManager()
    ckpt_mgr.save_checkpoint_for_megatron(iteration, model, optimizer, opt_param_scheduler)
    ckpt_mgr.async_save_ckpt_to_storage(iteration)

    memory_used = (psutil.virtual_memory().total - psutil.virtual_memory().available) / (1024 ** 3)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print_rank_0(f'async save checkpoint at {iteration}, memory used {memory_used} GB, '
                 f'start timestamp {start_time}, end timestamp {end_time}, cost time {elapsed_time} seconds')

    if args.tensorboard_dir is not None and is_rank0():
        writer.add_scalar('checkpoint/{}'.format(os.getenv('HOSTNAME')), elapsed_time, iteration)
        writer.close()


def load_checkpoint_for_megatron(model, optimizer, opt_param_scheduler, load_arg='load', strict=True):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    args = get_args()
    if args.enable_accelerate_checkpoint is None or not args.enable_accelerate_checkpoint:
        print_rank_0(f'sync load checkpoint')
        return load_checkpoint(model, optimizer, opt_param_scheduler, load_arg, strict)

    print_rank_0(f'async load checkpoint')
    load_dir = getattr(args, load_arg)
    _validate_checkpoint_done_for_megatron(load_dir)
    return load_checkpoint(model, optimizer, opt_param_scheduler, load_arg, strict)


def load_megatron_model_for_zero(model, load_arg='load', strict=True):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    """
    args = get_args()
    if args.enable_accelerate_checkpoint is None or not args.enable_accelerate_checkpoint:
        print_rank_0(f'sync load model checkpoint for zero')
        return load_model_checkpoint_for_zero(model, load_arg, strict)

    print_rank_0(f'async load model checkpoint for zero')
    load_dir = getattr(args, load_arg)
    _validate_checkpoint_done_for_megatron(load_dir)
    return load_model_checkpoint_for_zero(model, load_arg, strict)


def load_megatron_optim_for_zero(optimizer, opt_param_scheduler, load_arg='load'):
    """Load a optim checkpoint.
    """
    args = get_args()
    if args.enable_accelerate_checkpoint is None or not args.enable_accelerate_checkpoint:
        print_rank_0(f'sync load optim checkpoint for zero')
        return load_optim_checkpoint_for_zero(optimizer, opt_param_scheduler, load_arg)

    print_rank_0(f'async load optim checkpoint for zero')
    load_dir = getattr(args, load_arg)
    _validate_checkpoint_done_for_megatron(load_dir)
    return load_optim_checkpoint_for_zero(optimizer, opt_param_scheduler, load_arg)


def finish_checkpoint_process():
    """Finish checkpoint subprocess
    """
    global ckpt_mgr
    if ckpt_mgr is not None:
        ckpt_mgr.finish_ckpt_process()
