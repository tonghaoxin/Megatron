# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron arguments."""

import argparse
import collections
import json
import os
import torch
import torch.nn.functional
import types

from megatron.global_vars import set_retro_args, get_retro_args
from megatron.accelerator import get_accelerator
from tools.retro.utils import get_args_path as get_retro_args_path
from megatron.arguments_utils import extract_group_args, parse_group_args_to_dict


def parse_args(extra_args_provider=None, ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
                                     allow_abbrev=False)

    # Standard arguments.
    parser = _add_network_size_args(parser)
    parser = _add_regularization_args(parser)
    parser = _add_training_args(parser)
    parser = _add_initialization_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_mixed_precision_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_data_args(parser)
    parser = _add_autoresume_args(parser)
    parser = _add_biencoder_args(parser)
    parser = _add_vision_args(parser)
    parser = _add_logging_args(parser)
    parser = _add_inference_args(parser)
    parser = _add_transformer_engine_args(parser)
    parser = _add_retro_args(parser)
    parser = _add_glm_args(parser)
    parser = _add_chatglm_args(parser)
    parser = _add_blocklm_args(parser)
    parser = _add_finetune_config_args(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    # Skip train iterations
    if args.skip_train_iteration_range is not None:
        args.skip_train_iteration_range = [
            list(map(int, range_.split("-"))) for range_ in args.skip_train_iteration_range
        ]
        args.skip_train_iteration_range.sort()
        skip_train_iteration_range = collections.deque()
        for range_ in args.skip_train_iteration_range:
            if len(range_) == 2:
                start, end = range_
                assert end >= start, \
                "end of skip range cannot be smaller than start of skip range"
                # merge overlapping intervals (e.g. 1-5 2-6 -> 1-6)
                if not skip_train_iteration_range:
                    skip_train_iteration_range.append([start, end])
                elif skip_train_iteration_range[-1][1] >= start:
                    skip_train_iteration_range[-1][1] = max(end, skip_train_iteration_range[-1][1])
                else:
                    skip_train_iteration_range.append([start, end])
            else:
                raise ValueError(
                    "skip train iterations should be specified as two numbers, i.e. start-end"
                )
        if len(skip_train_iteration_range) > 0 and args.rampup_batch_size is not None:
            def get_rampup_end_iteration(start_batch_size, batch_size_increment, ramup_samples, global_batch_size):
                num_increments = (global_batch_size - start_batch_size) // batch_size_increment
                rampup_samples_per_increment = ramup_samples / num_increments
                consumed_samples = 0
                iteration = 0
                current_global_batch_size = 0
                while consumed_samples <= ramup_samples:
                    steps = int(consumed_samples / rampup_samples_per_increment)
                    current_global_batch_size = start_batch_size + steps * batch_size_increment
                    consumed_samples += current_global_batch_size
                    iteration += 1
                return iteration

            start_batch_size = int(args.rampup_batch_size[0])
            batch_size_increment = int(args.rampup_batch_size[1])
            ramup_samples = int(args.rampup_batch_size[2])
            global_batch_size = args.global_batch_size
            assert skip_train_iteration_range[0][0] \
                > get_rampup_end_iteration(start_batch_size, batch_size_increment, ramup_samples, global_batch_size), \
                "train iterations can only be skipped after the rampup has been performed"

        args.skip_train_iteration_range = skip_train_iteration_range

    # Args from environment
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    # Parse group args into dict
    args = parse_group_args_to_dict(args)

    return args


def validate_args(args, defaults={}):
    """validate_args"""
    # Tensor model parallel size.
    args.tensor_model_parallel_size = min(
        args.tensor_model_parallel_size, args.world_size)
    assert args.world_size % args.tensor_model_parallel_size == 0, 'world size'\
        ' ({}) is not divisible by tensor model parallel size ({})'.format(
            args.world_size, args.tensor_model_parallel_size)
    # Pipeline model parallel size.
    args.pipeline_model_parallel_size = min(
        args.pipeline_model_parallel_size,
        (args.world_size // args.tensor_model_parallel_size))
    args.transformer_pipeline_model_parallel_size = (
        args.pipeline_model_parallel_size - 1
        if args.standalone_embedding_stage else
        args.pipeline_model_parallel_size
    )
    # Checks.
    model_parallel_size = args.pipeline_model_parallel_size * \
                          args.tensor_model_parallel_size
    assert args.world_size % model_parallel_size == 0, 'world size({}) is not'\
        ' divisible by tensor parallel size ({}) times pipeline parallel ' \
        'size ({})'.format(args.world_size, args.tensor_model_parallel_size,
                           args.pipeline_model_parallel_size)
    args.data_parallel_size = args.world_size // model_parallel_size
    if args.rank == 0:
        print('using world size: {}, data-parallel-size: {}, '
              'tensor-model-parallel size: {}, '
              'pipeline-model-parallel size: {},'.format(
                  args.world_size, args.data_parallel_size,
                  args.tensor_model_parallel_size,
                  args.pipeline_model_parallel_size), flush=True)
    if args.pipeline_model_parallel_size > 1:
        if args.pipeline_model_parallel_split_rank is not None:
            assert args.pipeline_model_parallel_split_rank < \
                    args.pipeline_model_parallel_size, 'split rank needs'\
                    ' to be less than pipeline model parallel size ({})'.format(
                            args.pipeline_model_parallel_size)

    # Deprecated arguments
    if args.timing_log_level > 0:
        if args.rank == 0:
            print('WARNING: --timing-log-level will be deprecated, use --detail-log-interval instead')
    assert args.batch_size is None, '--batch-size argument is no longer ' \
        'valid, use --micro-batch-size instead'
    del args.batch_size
    assert args.warmup is None, '--warmup argument is no longer valid, use ' \
        '--lr-warmup-fraction instead'
    del args.warmup
    assert args.model_parallel_size is None, '--model-parallel-size is no ' \
        'longer valid, use --tensor-model-parallel-size instead'
    del args.model_parallel_size
    if args.tp_gemm_parallel:
        assert args.tensor_model_parallel_size > 1, 'when using --tp-gemm-parallel option,'\
            ' must be setting --tensor-model-parallel-size > 1 '
        assert args.tp_kernel_split_num >= 1, 'when using --tp-gemm-parallel option,'\
            ' must be setting --tp-kernel-split-num >= 1'
        assert not args.async_tensor_model_parallel_allreduce, 'when using --tp-gemm-parallel option,'\
            ' must be setting --no-async-tensor-model-parallel-allreduce '

    if args.pipeline_model_parallel_partition_balanced:
        args.pipeline_model_parallel_partition_method = 'balanced'
        if args.rank == 0:
            print('--pipeline-model-parallel-partition-balanced is no longer valid, '
                  'use --pipeline-model-parallel-partition-method instead. '
                  'Defaulting to pipeline-model-parallel-partition-method=balanced.')
    del args.pipeline_model_parallel_partition_balanced

    if args.checkpoint_activations:
        if args.rank == 0:
            print('--checkpoint-activations is no longer valid, use --recompute-activations, '
                  'or, for more control, --recompute-granularity and --recompute-method.')
        exit()
    del args.checkpoint_activations

    if args.recompute_activations:
        args.recompute_granularity = 'selective'
    del args.recompute_activations

    if args.recompute_granularity == 'hybrid':
        args.recompute_method = 'block'
        if args.rank == 0:
            print('set recompute-method=block since recompute-granularity=hybrid')

    if args.multi_query_attention and not args.group_query_attention:
        args.group_query_attention = True
        args.num_query_groups = args.multi_query_group_num
    del args.multi_query_attention
    del args.multi_query_group_num

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])

    # Batch size.
    assert args.micro_batch_size is not None
    assert args.micro_batch_size > 0
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    assert args.global_batch_size > 0
    if args.num_layers_per_virtual_pipeline_stage is not None:
        assert args.pipeline_model_parallel_size > 2, \
            'pipeline-model-parallel size should be greater than 2 with ' \
            'interleaved schedule'
        assert args.num_layers % args.num_layers_per_virtual_pipeline_stage == 0, \
            'number of layers is not divisible by number of layers per virtual ' \
            'pipeline stage'
        assert args.pipeline_model_parallel_partition_method is None, \
            'pipeline-model-parallel-partition-method is not allowed for interleaved schedule'
        args.virtual_pipeline_model_parallel_size = \
            (args.num_layers // args.transformer_pipeline_model_parallel_size) // \
            args.num_layers_per_virtual_pipeline_stage
    else:
        args.virtual_pipeline_model_parallel_size = None

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        assert not args.bf16
        args.params_dtype = torch.half
    if args.bf16:
        assert not args.fp16
        args.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.
        if not args.accumulate_allreduce_grads_in_fp32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for '
                      'bfloat16 data type.', flush=True)

    if args.rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype),
              flush=True)

    # If we do accumulation and all-reduces in fp32, we need to have local DDP
    # and we should make sure use-contiguous-buffers-in-local-ddp is not off.
    if args.accumulate_allreduce_grads_in_fp32:
        assert args.DDP_impl == 'local'
        assert args.use_contiguous_buffers_in_local_ddp

    # If we use the distributed optimizer, we need to have local DDP
    # and we should make sure use-contiguous-buffers-in-local-ddp is on.
    if args.use_distributed_optimizer:
        assert args.DDP_impl == 'local'
        assert args.use_contiguous_buffers_in_local_ddp

    if args.use_chunk:
        assert args.use_distributed_optimizer, \
            "use-distributed-optimizer is needed for use-chunk."

    if args.cpu_offload:
        assert args.use_chunk, \
            "use-chunk is needed for cpu-offload."

    if args.use_zero_optimizer:
        # if we use the zero optimizer, we should make sure use_distributed_optimizer is off.
        assert not args.use_distributed_optimizer, \
            "can only specify one of use-distributed-optimizer and use-zero-optimizer"
        args.use_contiguous_buffers_in_local_ddp = False
        if args.rank == 0:
            print('Disable contiguous grad buffer in DDP, since the optimizer would handle it.', flush=True)

        args.gradient_accumulation_fusion = False
        if args.rank == 0:
            print('Disable gradient accumulation fusion, since the optimizer would handle it.', flush=True)


    # For torch DDP, we do not use contiguous buffer
    if args.DDP_impl == 'torch':
        args.use_contiguous_buffers_in_local_ddp = False

    if args.dataloader_type is None:
        args.dataloader_type = 'single'

    # Consumed tokens.
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0

    # Support for variable sequence lengths across batches/microbatches.
    # set it if the dataloader supports generation of variable sequence lengths
    # across batches/microbatches. Due to additional communication overhead
    # during pipeline parallelism, it should not be set if sequence length
    # is constant during training.
    # args.variable_seq_lengths = False

    # Iteration-based training.
    if args.train_iters:
        # If we use iteration-based training, make sure the
        # sample-based options are off.
        assert args.train_samples is None, \
            'expected iteration-based training'
        assert args.lr_decay_samples is None, \
            'expected iteration-based learning rate decay'
        assert args.lr_warmup_samples == 0, \
            'expected iteration-based learning rate warmup'
        assert args.rampup_batch_size is None, \
            'expected no batch-size rampup for iteration-based training'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_iters == 0, \
                'can only specify one of lr-warmup-fraction and lr-warmup-iters'

    # Sample-based training.
    if args.train_samples:
        # If we use sample-based training, make sure the
        # iteration-based options are off.
        assert args.train_iters is None, \
            'expected sample-based training'
        assert args.lr_decay_iters is None, \
            'expected sample-based learning rate decay'
        assert args.lr_warmup_iters == 0, \
            'expected sample-based learnig rate warmup'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_samples == 0, \
                'can only specify one of lr-warmup-fraction ' \
                'and lr-warmup-samples'

    if args.num_layers is not None:
        assert args.encoder_num_layers is None, \
            'cannot have both num-layers and encoder-num-layers specified'
        args.encoder_num_layers = args.num_layers
    else:
        assert args.encoder_num_layers is not None, \
            'either num-layers or encoder-num-layers should be specified'
        args.num_layers = args.encoder_num_layers

    # Check required arguments.
    required_args = ['num_layers', 'hidden_size', 'num_attention_heads']
    for req_arg in required_args:
        _check_arg_is_not_none(args, req_arg)

    # Checks.
    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = 4 * args.hidden_size
        # Custom dim factor multiplier
        if args.ffn_dim_multiplier is not None:
            args.ffn_hidden_size = int(args.ffn_dim_multiplier * args.ffn_hidden_size)
        args.ffn_hidden_size = args.multiple_of * \
                ((args.ffn_hidden_size + args.multiple_of - 1) // args.multiple_of)

    # Reset activation function for compatibility
    if args.relu:
        args.activation_func = 'relu'
    elif args.swiglu:
        args.activation_func = 'swiglu'
    elif args.geglu:
        args.activation_func = 'geglu'
    elif args.openai_gelu:
        args.activation_func = 'openai_gelu'
    elif args.onnx_safe:
        args.activation_func = 'erf_glu'
    elif args.squared_relu:
        args.activation_func = 'squared_relu'

    if args.activation_func in ['swiglu', 'geglu']:
        # reduce the dimnesion for MLP since projections happens on
        # two linear layers. this keeps the number of paramters in
        # the same ballpark as the counterpart with 4*h size
        # we keep it a multiple of 64, which means the actual tensor size
        # will be a multiple of 64 / tp_size
        if args.ffn_hidden_size is None:
            args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64

    # FIXME: verify the correcteness
    if not args.use_rotary_position_embeddings_with_position_ids:
        assert args.max_position_embeddings is not None
        if args.seq_length is not None:
            assert args.max_position_embeddings >= args.seq_length
        if args.decoder_seq_length is not None:
            assert args.max_position_embeddings >= args.decoder_seq_length
    else:
        assert args.max_position_embeddings is None

    # FIXME: verify the correctness
    # if args.use_alibi_position_embeddings:
    #     assert not args.use_rotary_position_embeddings \
    #             and args.use_rotary_position_embeddings_with_position_ids
    if args.use_alibi_position_embeddings:
        assert not args.use_flash_attn, 'Alibi is not supported by FlashAttention for now.'

    if args.kv_channels is None:
        assert args.hidden_size % args.num_attention_heads == 0
        args.kv_channels = args.hidden_size // args.num_attention_heads

    if args.seq_length is not None:
        assert args.encoder_seq_length is None
        args.encoder_seq_length = args.seq_length
    else:
        assert args.encoder_seq_length is not None
        args.seq_length = args.encoder_seq_length

    if args.lr is not None:
        assert args.min_lr <= args.lr
    if args.save is not None:
        assert args.save_interval is not None
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        assert args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.'
    if args.fp32_residual_connection:
        assert args.fp16 or args.bf16, \
            'residual connection in fp32 only supported when using fp16 or bf16.'

    if args.weight_decay_incr_style == 'constant':
        assert args.start_weight_decay is None
        assert args.end_weight_decay is None
        args.start_weight_decay = args.weight_decay
        args.end_weight_decay = args.weight_decay
    else:
        assert args.start_weight_decay is not None
        assert args.end_weight_decay is not None

    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    # Persistent fused layer norm.
    if TORCH_MAJOR < 1 or (TORCH_MAJOR == 1 and TORCH_MINOR < 11):
        args.no_persist_layer_norm = True
        if args.rank == 0:
            print('Persistent fused layer norm kernel is supported from '
                  'pytorch v1.11 (nvidia pytorch container paired with v1.11). '
                  'Defaulting to no_persist_layer_norm=True')

    # Activation recomputing.
    if args.distribute_saved_activations:
        assert args.tensor_model_parallel_size > 1, 'can distribute ' \
            'recomputed activations only across tensor model ' \
            'parallel groups'
        assert args.recompute_granularity in ['full', 'hybrid'], \
            'distributed recompute activations is only '\
            'application to full or hybrid recompute granularity'
        assert args.recompute_method is not None, \
            'for distributed recompute activations to work you '\
            'need to use a recompute method '
        assert TORCH_MAJOR >= 1 and TORCH_MINOR >= 10, \
            'distributed recompute activations are supported for pytorch ' \
            'v1.10 and above (Nvidia Pytorch container >= 21.07). Current ' \
            'pytorch version is v%s.%s.' % (TORCH_MAJOR, TORCH_MINOR)

    # Tranformer-Engine/FP8 related checking
    if args.fp8_e4m3 or args.fp8_hybrid:
        assert args.transformer_impl == 'transformer_engine', \
            'transformer-engine required for fp8 training and inference'

    assert not (args.fp8_e4m3 and args.fp8_hybrid), \
        'cannot train with both fp8 e4m3 and hybrid formatting'

    if args.fp16:
        assert args.transformer_impl == 'local', \
            'transformer-engine not yet approved for fp16 training and inference'

    # disable sequence parallelism when tp=1
    # to avoid change in numerics when
    # sequence_parallelism is enabled.
    if args.tensor_model_parallel_size == 1:
        args.sequence_parallel = False

    # disable async_tensor_model_parallel_allreduce when
    # model parallel memory optimization is enabled
    if args.sequence_parallel:
        args.async_tensor_model_parallel_allreduce = False
        args.tp_gemm_parallel = False

    if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1" and \
        get_accelerator().device_name() == "cuda" and \
        not args.use_zero_optimizer:
        if args.sequence_parallel:
            raise RuntimeError(
                "Using sequence parallelism requires setting the environment variable "
                "CUDA_DEVICE_MAX_CONNECTIONS to 1. "
                "Other accelerator does not support sequence parallel for the moment.")
        if args.async_tensor_model_parallel_allreduce:
            raise RuntimeError(
                "Using async gradient all reduce requires setting the environment "
                "variable CUDA_DEVICE_MAX_CONNECTIONS to 1. "
                "Other accelerator does not support async gradient allreduce for the moment.")

    # Disable bias gelu fusion if we are disabling bias altogether
    if not args.add_bias_linear:
        args.bias_gelu_fusion = False

    # Load retro args.
    if args.retro_workdir:
        retro_args_path = get_retro_args_path(args.retro_workdir)
        if os.path.exists(retro_args_path):
            with open(retro_args_path) as f:
                retro_args = types.SimpleNamespace(**json.load(f))
                retro_args.retro_return_doc_ids = args.retro_return_doc_ids
                retro_args.retro_gpt_retrieved_length = \
                    args.retro_num_retrieved_chunks * \
                    retro_args.retro_gpt_chunk_length
                set_retro_args(retro_args)

    # DeepNorm need to be used with post-layernorm residual connection
    if args.deepnorm:
        args.apply_residual_connection_post_layernorm = True

    # RoPE embedding kernel is not compatible with learnable RoPE
    if args.apply_rotary_positional_embedding_kernel:
        assert not args.learnable_rotary_embedding, \
                'RoPE embedding kernel is not compatible with learnable RoPE'

    # Print arguments.
    _print_args("arguments", args)
    retro_args = get_retro_args()
    if retro_args and args != retro_args:
        _print_args("retro arguments", types.SimpleNamespace(**{k:v for k,v in vars(retro_args).items() if k.startswith("retro")}, rank=args.rank))

    return args


def _print_args(title, args):
    """Print arguments."""
    if args.rank == 0:
        print(f'------------------------ {title} ------------------------',
              flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(f'-------------------- end of {title} ---------------------',
              flush=True)


def _check_arg_is_not_none(args, arg):
    """_check_arg_is_not_none"""
    assert getattr(args, arg) is not None, '{} argument is None'.format(arg)


def _add_transformer_engine_args(parser):
    """_add_transformer_engine_args"""
    group = parser.add_argument_group(title='Transformer-Engine')

    group.add_argument('--fp8-e4m3', action='store_true',
                        help='E4M3 TransformerLayer', dest='fp8_e4m3')
    group.add_argument('--fp8-hybrid', action='store_true',
                        help='Hybrid FP8 TransformerLayer', dest='fp8_hybrid')
    group.add_argument('--no-fp8-wgrad', action='store_false',
                        help='Execute wgrad in higher precision even for FP8 runs', dest='fp8_wgrad')
    group.add_argument('--fp8-margin', type=int, default=0,
                        help='Scaling margin for fp8', dest='fp8_margin')
    group.add_argument('--fp8-interval', type=int, default=1,
                        help='Scaling update interval for fp8', dest='fp8_interval')
    group.add_argument('--transformer-impl', default='local',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.',
                       dest='transformer_impl')
    group.add_argument('--fp8-amax-history-len', type=int, default=1,
                        help='Number of steps for which amax history is recorded per tensor',
                        dest='fp8_amax_history_len')
    group.add_argument('--fp8-amax-compute-algo', default='most_recent',
                       choices=['most_recent', 'max'],
                       help='Algorithm for computing amax from history',
                       dest='fp8_amax_compute_algo')

    return parser

def _add_inference_args(parser):
    """_add_inference_args"""
    group = parser.add_argument_group(title='inference')

    group.add_argument('--inference-batch-times-seqlen-threshold',
                       type=int, default=512,
                       help='During inference, if batch-size times '
                       'sequence-length is smaller than this threshold '
                       'then we will not use pipelining, otherwise we will.')
    group.add_argument('--max-tokens-to-oom',
                       type=int, default=12000,
                       help='Maximum number of tokens during inference'
                       'tokens here is # in prompt + # to generate'
                       'Allows us to throw an error before OOM crashes server')
    group.add_argument('--output-bert-embeddings', action='store_true',
                       help='Output Bert embeddings (via mean pooling) from '
                       'model, rather than its binary head output or entire '
                       'hidden batch.')
    group.add_argument('--bert-embedder-type', default="megatron",
                       choices=["megatron", "huggingface"],
                       help='Select either Megatron or Huggingface as the '
                       'Bert embedder.')

    return parser


def _add_retro_args(parser):
    """_add_retro_args"""
    group = parser.add_argument_group(title='retro')

    group.add_argument('--retro-workdir', default=None,
                       help='Retro working directory, which contains the '
                       'preprocessed data for for pretraining. This directory '
                       'is built during preprocessing (see '
                       'tools/retro/README.md), and contains subdirectories '
                       'for the chunk database and pretraining neighbors.')
    group.add_argument('--retro-add-retriever',
                       action='store_true', default=False,
                       help='Add a retriever to the transformer, for use in '
                       'pretraining a Retro model.')
    group.add_argument('--retro-cyclic-train-iters', type=int, default=None,
                       help='Set number of training iterations for cyclic '
                       'Retro training.')
    group.add_argument('--retro-encoder-layers', type=int, default=2,
                       help='Number of layers to use for the retrieval '
                       'encoder.')
    group.add_argument('--retro-encoder-hidden-dropout',
                       type=float, default=0.1, help='Hidden dropout for '
                       'retrieval encoder.')
    group.add_argument('--retro-encoder-attention-dropout',
                       type=float, default=0.1, help='Attention dropout for '
                       'retrieval encoder.')
    group.add_argument("--retro-num-neighbors", type=int, default=2,
                       help='Number of neighbors to retrieve during '
                       'pretraining.')
    group.add_argument("--retro-num-retrieved-chunks", type=int, default=2,
                       help='Number of chunks to retrieve from the retrieval '
                       'database.')
    group.add_argument("--retro-return-doc-ids", action="store_true",
                       help="Turn this on when preprocessing retro data.")

    # Enforce argument naming convention.
    for action in group._group_actions:
        prefix = action.dest.split("_")[0]
        assert prefix == "retro", \
            "Retro args must be prefixed with '--retro-*', for consistent " \
            "styling. Please fix '%s'." % ", ".join(action.option_strings)

    return parser


def _add_network_size_args(parser):
    """_add_network_size_args"""
    group = parser.add_argument_group(title='network size')

    group.add_argument('--num-layers', type=int, default=None,
                       help='Number of transformer layers.')
    group.add_argument('--encoder-num-layers', type=int, default=None,
                       help='Number of encoder transformer layers.')
    group.add_argument('--decoder-num-layers', type=int, default=None,
                       help='Number of decoder transformer layers.')
    group.add_argument('--hidden-size', type=int, default=None,
                       help='Tansformer hidden size.')
    group.add_argument('--ffn-hidden-size', type=int, default=None,
                       help='Transformer Feed-Forward Network hidden size. '
                       'This is set to 4*hidden-size if not provided')
    group.add_argument('--num-attention-heads', type=int, default=None,
                       help='Number of transformer attention heads.')
    group.add_argument('--kv-channels', type=int, default=None,
                       help='Projection weights dimension in multi-head '
                       'attention. This is set to '
                       '   args.hidden_size // args.num_attention_heads '
                       'if not provided.')
    group.add_argument('--group-query-attention', action='store_true',
                          help='Use group-query attention.')
    group.add_argument('--num-query-groups', type=int, default=1)
    group.add_argument('--max-position-embeddings', type=int, default=None,
                       help='Maximum number of position embeddings to use. '
                       'This is the size of position embedding.')
    group.add_argument('--use-rotary-position-embeddings', action='store_true',
                       help='Use rotary positional embeddings or not.')
    group.add_argument('--use-rotary-position-embeddings-with-position-ids', action='store_true',
                       help='Use rotary positional embeddings with position ids or not.')
    group.add_argument('--rotary-percent', type=float, default=1.0,
                       help='Percent of rotary dimension to use, default 100%')
    group.add_argument('--rotary-seq-len-interpolation-factor', type=int, default=1,
                       help='Sequence length interpolation factor for rotary embeddings.')
    group.add_argument('--rotary-embedding-2d', action='store_true',
                       help='If set, use 2D rotary embedding.')
    group.add_argument('--rotary-embedding-2d-to-1d', action='store_true',
                       help='If set, use 2D rotary embedding and 2dto1d.')
    group.add_argument('--apply-rotary-positional-embedding-kernel', action='store_true',
                       help='Use custom kernel for rotary positional embedding.')
    group.add_argument('--learnable-rotary-embedding', action='store_true',
                       help='If set, make rotary embedding learnable.')
    group.add_argument('--use-learned-position-embeddings', action='store_true',
                       help='Use learned positional embeddings for opt/galactica '
                       'model training')
    group.add_argument('--use-alibi-position-embeddings', action='store_true',
                       help='Use alibi positional embeddings or not.')
    group.add_argument('--no-position-embedding',
                       action='store_false',
                       help='Disable position embedding.',
                       dest='add_position_embedding')
    group.add_argument('--positional-encoding-2d', action='store_true',
                       help='Use positional encoding 2d or not.')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    group.add_argument('--pad-vocab-size-to', type=int, default=None,
                       help='Pad the vocab size to this value.'
                       'This value must be greater than the initial size of the tokenizer'
                       ', needs to be divisible by TP size and `make-vocab-size-divisible-by`.')
    group.add_argument('--layernorm-epsilon', type=float, default=1e-5,
                       help='Layer norm epsilon.')
    group.add_argument('--apply-layernorm-1p', action='store_true',
                       help='Adjust LayerNorm weights such that they are centered '
                       'around zero. This improves numerical stability.')
    group.add_argument('--apply-residual-connection-post-layernorm',
                       action='store_true',
                       help='If set, use original BERT residula connection '
                       'ordering.')
    group.add_argument('--deepnorm', action='store_true',
                       help='use DeepNorm.')
    group.add_argument('--fused-rmsnorm', action='store_true',
                       help='Use fused rmsnorm instead of default rmsnorm.')
    group.add_argument('--rmsnorm-epsilon', type=float, default=1e-6,
                       help='RMS norm epsilon.')
    group.add_argument('--add-embed-layernorm', action='store_true',
                       help='Enable embedding layernorm.')
    group.add_argument('--activation-func', type=str, required=False, default='gelu',
                       choices=['relu', 'gelu', 'geglu', 'liglu', 'reglu', 'swiglu',
                                'openai_gelu', 'erf_gelu', 'squared_relu'],
                       help='Activation function for MLP layer.')
    group.add_argument('--apply-activation-func-fusion', action='store_true',
                       help='If set, use torch.jit.script to fuse activation function.')
    group.add_argument('--bert-no-binary-head', action='store_false',
                       help='Disable BERT binary head.',
                       dest='bert_binary_head')
    group.add_argument('--num-experts', type=int, default=None,
                       help='Number of Experts in Switch Transformer (None means no Switch).')
    group.add_argument('--untie-embeddings-and-output-weights', action='store_true',
                       help='Untie embeddings and output weights.')
    group.add_argument('--multi-query-attention', action='store_true',
                       help='If set, use multi-query attention.')
    group.add_argument('--multi-query-group-num', type=int, default=1,
                       help='Number of multi-query groups (or kv_heads). '
                       'min = tensor_model_parallel_size, '
                       'max = num-attention-heads // tensor_model_parallel_size)')
    group.add_argument('--ffn-dim-multiplier', type=float, default=None,
                       help='Dimension multiplier for Transformer Feed-Forward Network.')
    group.add_argument('--multiple-of', type=int, default=256,
                       help='Make SwiGLU hidden layer size multiple of large power of 2.')
    return parser


def _add_logging_args(parser):
    """_add_logging_args"""
    group = parser.add_argument_group(title='logging')

    group.add_argument('--log-params-norm', action='store_true',
                       help='If set, calculate and log parameters norm.')
    group.add_argument('--log-num-zeros-in-grad', action='store_true',
                       help='If set, calculate and log the number of zeros in gradient.')
    group.add_argument('--no-detail-log',
                       action='store_false',
                       help='If set, the detail-log-interval will no longer take effect.',
                       dest='log_detail')
    group.add_argument('--detail-log-interval', type=int, default=20,
                       help='Report timing interval.'
                            ' detail-log-interval will only take effect when the'
                            ' timing-log-level is set to 0')
    # deprecated
    group.add_argument('--timing-log-level', type=int,
                       default=0, choices=range(0,3),
                       help='Granularity level to measure and report timing. '
                       '   0: report only iteration time and make sure timing '
                       '      does not introduce extra overhead.'
                       '   1: report timing for operations that are executed '
                       '      very limited times (basically once) during '
                       '      each iteration (such as gradient all-reduce) '
                       '   2: report timing for operations that migh be '
                       '      executed numerous times during each iteration. '
                       'Note that setting the level to 1 or 2 might '
                       'cause increase in iteration time.')
    group.add_argument('--no-barrier-with-level-1-timing', action='store_false',
                       help='If not set, use barrier with level 1 time '
                       'measurements. Note that this is up to the user '
                       'to make sure calling barrier with their timers '
                       'will not result in hangs. This can happen if for '
                       'example the user adds a level 1 timer that is not '
                       'called by all ranks.',
                       dest='barrier_with_L1_time')
    group.add_argument('--timing-log-option', type=str, default='minmax',
                       choices=['max', 'minmax', 'all'],
                       help='Options for logging timing:'
                       '  max: report the max timing across all ranks'
                       '  minmax: report min and max timings across all ranks'
                       '  all: report timings of all ranks.')
    group.add_argument('--tensorboard-log-interval', type=int, default=1,
                       help='Report to tensorboard interval.')
    group.add_argument('--tensorboard-queue-size', type=int, default=1000,
                       help='Size of the tensorboard queue for pending events '
                       'and summaries before one of the ‘add’ calls forces a '
                       'flush to disk.')
    group.add_argument('--log-timers-to-tensorboard', action='store_true',
                       help='If set, write timers to tensorboard.')
    group.add_argument('--log-batch-size-to-tensorboard', action='store_true',
                       help='If set, write batch-size to tensorboard.')
    group.add_argument('--no-log-learnig-rate-to-tensorboard',
                       action='store_false',
                       help='Disable learning rate logging to tensorboard.',
                       dest='log_learning_rate_to_tensorboard')
    group.add_argument('--no-log-loss-scale-to-tensorboard',
                       action='store_false',
                       help='Disable loss-scale logging to tensorboard.',
                       dest='log_loss_scale_to_tensorboard')
    group.add_argument('--log-validation-ppl-to-tensorboard',
                       action='store_true',
                       help='If set, write validation perplexity to '
                       'tensorboard.')
    group.add_argument('--log-memory-to-tensorboard',
                       action='store_true',
                       help='Enable memory logging to tensorboard.')
    group.add_argument('--log-world-size-to-tensorboard',
                       action='store_true',
                       help='Enable world size logging to tensorboard.')

    return parser


def _add_regularization_args(parser):
    """_add_regularization_args"""
    group = parser.add_argument_group(title='regularization')

    group.add_argument('--attention-dropout', type=float, default=0.1,
                       help='Post attention dropout probability.')
    group.add_argument('--hidden-dropout', type=float, default=0.1,
                       help='Dropout probability for hidden state transformer.')
    group.add_argument('--embedding-dropout', type=float, default=0.1,
                       help='Dropout probability for word embeddings.')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay coefficient for L2 regularization.')
    group.add_argument('--start-weight-decay', type=float,
                       help='Initial weight decay coefficient for L2 regularization.')
    group.add_argument('--end-weight-decay', type=float,
                       help='End of run weight decay coefficient for L2 regularization.')
    group.add_argument('--weight-decay-incr-style', type=str, default='constant',
                       choices=['constant', 'linear', 'cosine'],
                       help='Weight decay increment function.')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='Gradient clipping based on global L2 norm.')
    group.add_argument('--adam-beta1', type=float, default=0.9,
                       help='First coefficient for computing running averages '
                       'of gradient and its square')
    group.add_argument('--adam-beta2', type=float, default=0.999,
                       help='Second coefficient for computing running averages '
                       'of gradient and its square')
    group.add_argument('--adam-eps', type=float, default=1e-08,
                       help='Term added to the denominator to improve'
                       'numerical stability')
    group.add_argument('--sgd-momentum', type=float, default=0.9,
                       help='Momentum factor for sgd')
    group.add_argument('--shrink-embedding-gradient-alpha', type=float, default=1.0,
                       help='Shrink embedding gradient for alpha')
    group.add_argument('--shrink-word-embedding', action='store_true')
    group.add_argument('--shrink-logit-embedding-gradient', action='store_true')
    group.add_argument('--shrink-embedding-gradient-steps', nargs='*', default=None,
                       help='--shrink-embedding-gradient-steps <x1> <x2>'
                            'Shrink embedding gradient alpha for x1 steps,'
                            'then warm it up to 1.0 with x2 steps')

    return parser


def _add_training_args(parser):
    """_add_training_args"""
    group = parser.add_argument_group(title='training')

    group.add_argument('--micro-batch-size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.')
    group.add_argument('--batch-size', type=int, default=None,
                       help='Old batch size parameter, do not use. '
                       'Use --micro-batch-size instead')
    group.add_argument('--global-batch-size', type=int, default=None,
                       help='Training batch size. If set, it should be a '
                       'multiple of micro-batch-size times data-parallel-size. '
                       'If this value is None, then '
                       'use micro-batch-size * data-parallel-size as the '
                       'global batch size. This choice will result in 1 for '
                       'number of micro-batches.')
    group.add_argument('--rampup-batch-size', nargs='*', default=None,
                       help='Batch size ramp up with the following values:'
                       '  --rampup-batch-size <start batch size> '
                       '                      <batch size incerement> '
                       '                      <ramp-up samples> '
                       'For example:'
                       '   --rampup-batch-size 16 8 300000 \ '
                       '   --global-batch-size 1024'
                       'will start with global batch size 16 and over '
                       ' (1024 - 16) / 8 = 126 intervals will increase'
                       'the batch size linearly to 1024. In each interval'
                       'we will use approximately 300000 / 126 = 2380 samples.')
    group.add_argument('--recompute-activations', action='store_true',
                       help='recompute activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--recompute-granularity', type=str, default=None,
                       choices=['full', 'selective', 'hybrid'],
                       help='Checkpoint activations to allow for training '
                       'with larger models, sequences, and batch sizes. '
                       'It is supported at three granularities 1) full: '
                       'whole transformer layer is recomputed, '
                       '2) selective: core attention part of the transformer '
                       'layer is recomputed, '
                       '3) hybrid: full:block method for recompute-num-layers '
                       'within each pipeline stage, selective method for the reset of layers '
                       'within each pipeline stage')
    group.add_argument('--distribute-saved-activations',
                       action='store_true',
                       help='If set, distribute recomputed activations '
                       'across model parallel group.')
    group.add_argument('--recompute-method', type=str, default=None,
                       choices=['uniform', 'block'],
                       help='1) uniform: uniformly divide the total number of '
                       'Transformer layers and recompute the input activation of '
                       'each divided chunk at specified granularity, '
                       '2) recompute the input activations of only a set number of '
                       'individual Transformer layers per pipeline stage and do the '
                       'rest without any recomputing at specified granularity'
                       'default) do not apply activations recompute to any layers')
    group.add_argument('--recompute-num-layers', type=int, default=1,
                       help='1) uniform: the number of Transformer layers in each '
                       'uniformly divided recompute unit, '
                       '2) block: the number of individual Transformer layers '
                       'to recompute within each pipeline stage.')
    group.add_argument('--profile', action='store_true',
                       help='Enable nsys profiling. When using this option, nsys '
                       'options should be specified in commandline. An example '
                       'nsys commandline is `nsys profile -s none -t nvtx,cuda '
                       '-o <path/to/output_file> --force-overwrite true '
                       '--capture-range=cudaProfilerApi '
                       '--capture-range-end=stop`.')
    group.add_argument('--profile-step-start', type=int, default=10,
                       help='Gloable step to start profiling.')
    group.add_argument('--profile-step-end', type=int, default=12,
                       help='Gloable step to stop profiling.')
    group.add_argument('--profile-ranks', nargs='+', type=int, default=[0],
                       help='Global ranks to profile.')
    group.add_argument('--recompute-view-for-virtual-pipeline-stage', type=str, required=False, default='local',
                       choices=['local', 'global'],
                       help='1) local: apply recompute strategy to each virtual pipeline stage, '
                       '2) global: apply recomptue strategy to a set of virtual pipeline stages '
                       '(model chunks) which compose a pipeline stage.')
    group.add_argument('--pipeline-model-parallel-partition-balanced', action='store_true',
                       help='split pipeline model parallel use partition balanced.')

    # deprecated
    group.add_argument('--checkpoint-activations', action='store_true',
                       help='Checkpoint activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--train-iters', type=int, default=None,
                       help='Total number of iterations to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--train-samples', type=int, default=None,
                       help='Total number of samples to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--label-smoothing', type=float, default=0.0)
    group.add_argument('--log-interval', type=int, default=100,
                       help='Report loss and timing interval.')
    group.add_argument('--exit-interval', type=int, default=None,
                       help='Exit the program after the iteration is divisible '
                       'by this value.')
    group.add_argument('--exit-duration-in-mins', type=int, default=None,
                       help='Exit the program after this many minutes.')
    group.add_argument('--exit-signal-handler', action='store_true',
                       help='Dynamically save the checkpoint and shutdown the '
                       'training if SIGTERM is received')
    group.add_argument('--tensorboard-dir', type=str, default=None,
                       help='Write TensorBoard logs to this directory.')
    group.add_argument('--no-masked-softmax-fusion',
                       action='store_false',
                       help='Disable fusion of query_key_value scaling, '
                       'masking, and softmax.',
                       dest='masked_softmax_fusion')
    group.add_argument('--no-bias-gelu-fusion', action='store_false',
                       help='Disable bias and gelu fusion.',
                       dest='bias_gelu_fusion')
    group.add_argument('--no-bias-dropout-fusion', action='store_false',
                       help='Disable bias and dropout fusion.',
                       dest='bias_dropout_fusion')
    group.add_argument('--use-flash-attn', action='store_true',
                       help='use FlashAttention implementation of attention. '
                       'https://arxiv.org/abs/2205.14135')
    group.add_argument('--attn-mask-pad', action='store_true', help='use attention mask to mask'
                       ' padding of inputs',
                       dest='attn_mask_pad')
    group.add_argument('--disable-bias-linear', action='store_false',
                       help='Disable bias in the linear layers',
                       dest='add_bias_linear')
    group.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'lion'],
                       help='Optimizer function')
    group.add_argument('--dataloader-type', type=str, default=None,
                       choices=['single', 'cyclic'],
                       help='Single pass vs multiple pass data loader')
    group.add_argument('--no-async-tensor-model-parallel-allreduce',
                       action='store_false',
                       help='Disable asynchronous execution of '
                       'tensor-model-parallel all-reduce with weight '
                       'gradient compuation of a column-linear layer.',
                       dest='async_tensor_model_parallel_allreduce')
    group.add_argument('--no-persist-layer-norm', action='store_true',
                       help='Disable using persistent fused layer norm kernel. '
                       'This kernel supports only a set of hidden sizes. Please '
                       'check persist_ln_hidden_sizes if your hidden '
                       'size is supported.')
    group.add_argument('--sequence-parallel', action='store_true',
                       help='Enable sequence parallel optimization.')
    group.add_argument('--no-gradient-accumulation-fusion',
                       action='store_false',
                       help='Disable fusing gradient accumulation to weight '
                       'gradient computation of linear layers',
                       dest='gradient_accumulation_fusion')
    group.add_argument('--dump-model', action='store_true',
                       help='Dump model to ./model.txt. Useful for debugging.')
    group.add_argument('--enable-debug', action='store_true', help='enable debug')
    group.add_argument('--debug-layer-list', type=int, default=[], nargs='+', help='index of debugging layers, e.g., 1 2 3')
    group.add_argument('--debug-tp-list', type=int, default=[], nargs='+', help='debugging tp rank, e.g., 0 1 2')
    group.add_argument('--debug-pp-list', type=int, default=[], nargs='+', help='debugging pp rank, e.g., 0 1 2')
    group.add_argument('--debug-dp-list', type=int, default=[], nargs='+', help='debugging dp rank, e.g., 0 1 2')
    group.add_argument('--debug-save-path', type=str, help='path to save debug info')
    group.add_argument('--skip-train-iteration-range', type=str, nargs='+', default=None,
                       help='Iteration ranges to skip. The values are one or more dash-separated ranges. e.g., 101-200 251-300.')
    group.add_argument('--skipped-samples-path', type=str, default=None, help='Path to save the skipped samples.')
    group.add_argument('--abort-on-unmet-fused-kernel-constraints', action='store_true',
                       help="If set to True, the program will abort if the constraints for loading a fused kernel aren't met")
    group.add_argument('--openai-gelu', action='store_true',
                       help='Use OpenAIs GeLU implementation. This option'
                       'should not be used unless for backward compatibility'
                       'reasons.')
    group.add_argument('--squared-relu', action='store_true',
                       help='Use squared relu activation instead of default gelu')
    group.add_argument('--relu', action='store_true',
                       help='Use relu activation instead of default gelu')
    group.add_argument('--swiglu', action='store_true',
                       help='Use gated linear units and SiLU activation instead of default gelu')
    group.add_argument('--geglu', action='store_true',
                       help='Use GELU activation instead of default gelu')
    group.add_argument('--onnx-safe', type=bool, required=False,
                       help='Use workarounds for known problems with '
                       'Torch ONNX exporter')
    group.add_argument('--memory-saving', type=bool, default=False, required=False,
                       help='Use memory saving strategies for several ops')
    group.add_argument('--findmax-opt', action='store_true', default=False,
                        help='open findmax optimization in fc/mm during training')
    group.add_argument('--use-chunk', action='store_true',
                       help='Use chunk for fp32 param and grad in optiimizer. '
                       'It is only used when use-distributed-optimizer. '
                       'Then argment cpu-offload and cpu-offload-pin-memory are used in Chunk, '
                       'they will be used when creating Tensor in Chunk')
    group.add_argument('--cpu-offload', action='store_true',
                       help='Use cpu offload in optimizer')
    group.add_argument('--cpu-offload-pin-memory', action='store_true',
                       help='Use pinned memory for optimizer')
    group.add_argument('--cpu-offload-percent', type=float, default=1.0,
                       help='Percent of the parameters computed in cpu for optimizer.')
    return parser


def _add_initialization_args(parser):
    """_add_initialization_args"""
    group = parser.add_argument_group(title='initialization')

    group.add_argument('--seed', type=int, default=1234,
                       help='Random seed used for python, numpy, '
                       'pytorch, and accelerator.')
    group.add_argument('--data-parallel-random-init', action='store_true',
                       help='Enable random initialization of params '
                       'across data parallel ranks')
    group.add_argument('--init-method-std', type=float, default=0.02,
                       help='Standard deviation of the zero mean normal '
                       'distribution used for weight initialization.')
    group.add_argument('--init-method-xavier-uniform', action='store_true',
                       help='Enable Xavier uniform parameter initialization')
    group.add_argument('--data-shuffle-seed', type=int, default=None,
                       help='Random seed used for shuffling GLM data')

    return parser


def _add_learning_rate_args(parser):
    """_add_learning_rate_args"""
    group = parser.add_argument_group(title='learning rate')

    group.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learing rate at each '
                       'iteration would be different.')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'inverse-square-root'],
                       help='Learning rate decay function.')
    group.add_argument('--lr-decay-iters', type=int, default=None,
                       help='number of iterations to decay learning rate over,'
                       ' If None defaults to `--train-iters`')
    group.add_argument('--lr-decay-samples', type=int, default=None,
                       help='number of samples to decay learning rate over,'
                       ' If None defaults to `--train-samples`')
    group.add_argument('--lr-warmup-fraction', type=float, default=None,
                       help='fraction of lr-warmup-(iters/samples) to use '
                       'for warmup (as a float)')
    group.add_argument('--lr-warmup-iters', type=int, default=0,
                       help='number of iterations to linearly warmup '
                       'learning rate over.')
    group.add_argument('--lr-warmup-samples', type=int, default=0,
                       help='number of samples to linearly warmup '
                       'learning rate over.')
    group.add_argument('--warmup', type=int, default=None,
                       help='Old lr warmup argument, do not use. Use one of the'
                       '--lr-warmup-* arguments above')
    group.add_argument('--min-lr', type=float, default=0.0,
                       help='Minumum value for learning rate. The scheduler'
                       'clip values below this threshold.')
    group.add_argument('--override-opt_param-scheduler', action='store_true',
                       help='Reset the values of the scheduler (learning rate,'
                       'warmup iterations, minimum learning rate, maximum '
                       'number of iterations, and decay style from input '
                       'arguments and ignore values from checkpoints. Note'
                       'that all the above values will be reset.')
    group.add_argument('--use-checkpoint-opt_param-scheduler', action='store_true',
                       help='Use checkpoint to set the values of the scheduler '
                       '(learning rate, warmup iterations, minimum learning '
                       'rate, maximum number of iterations, and decay style '
                       'from checkpoint and ignore input arguments.')

    return parser


def _add_checkpointing_args(parser):
    """_add_checkpointing_args"""
    group = parser.add_argument_group(title='checkpointing')

    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-interval', type=int, default=None,
                       help='Number of iterations between checkpoint saves.')
    group.add_argument('--no-save-optim', action='store_true', default=None,
                       help='Do not save current optimizer.')
    group.add_argument('--no-save-rng', action='store_true', default=None,
                       help='Do not save current rng state.')
    group.add_argument('--load', type=str, default=None,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--no-load-optim', action='store_true', default=None,
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--no-load-rng', action='store_true', default=None,
                       help='Do not load rng state when loading checkpoint.')
    group.add_argument('--finetune', action='store_true',
                       help='Load model for finetuning. Do not load optimizer '
                       'or rng state from checkpoint and set iteration to 0. '
                       'Assumed when loading a release checkpoint.')
    group.add_argument('--no-initialization', action='store_false',
                       help='Do not perform initialization when building model, '
                       'can reduce startup time when definitely loading from a '
                       'checkpoint',
                       dest='perform_initialization')
    group.add_argument('--use-checkpoint-args', action='store_true',
                       help='Override any command line arguments with arguments '
                       'from the checkpoint')
    group.add_argument('--exit-on-missing-checkpoint', action='store_true',
                       help="If '--load' is set, but checkpoint is not found "
                       "(e.g., path typo), then exit instead of random "
                       "initialization.")
    group.add_argument('--enable-accelerate-checkpoint', action='store_true', help='enable asynchronous checkpoint,'
                       'save checkpoint to memory, backend thread will save it '
                       'to storage asynchronously.' )

    return parser


def _add_mixed_precision_args(parser):
    """_add_mixed_precision_args"""
    group = parser.add_argument_group(title='mixed precision')

    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode.')
    group.add_argument('--bf16', action='store_true',
                       help='Run model in bfloat16 mode.')
    group.add_argument('--loss-scale', type=float, default=None,
                       help='Static loss scaling, positive power of 2 '
                       'values can improve fp16 convergence. If None, dynamic'
                       'loss scaling is used.')
    group.add_argument('--initial-loss-scale', type=float, default=2**32,
                       help='Initial loss-scale for dynamic loss scaling.')
    group.add_argument('--min-loss-scale', type=float, default=1.0,
                       help='Minimum loss scale for dynamic loss scale.')
    group.add_argument('--loss-scale-window', type=float, default=1000,
                       help='Window over which to raise/lower dynamic scale.')
    group.add_argument('--hysteresis', type=int, default=2,
                       help='hysteresis for dynamic loss scaling')
    group.add_argument('--fp32-residual-connection', action='store_true',
                       help='Move residual connections to fp32.')
    group.add_argument('--no-query-key-layer-scaling', action='store_false',
                       help='Do not scale Q * K^T by 1 / layer-number.',
                       dest='apply_query_key_layer_scaling')
    group.add_argument('--attention-softmax-in-fp32', action='store_true',
                       help='Run attention masking and softmax in fp32. '
                       'This flag is ignored unless '
                       '--no-query-key-layer-scaling is specified.')
    group.add_argument('--accumulate-allreduce-grads-in-fp32',
                       action='store_true',
                       help='Gradient accumulation and all-reduce in fp32.')
    group.add_argument('--fp16-lm-cross-entropy', action='store_true',
                       help='Move the cross entropy unreduced loss calculation'
                       'for lm head to fp16.')

    return parser


def _add_distributed_args(parser):
    """_add_distributed_args"""
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--tensor-model-parallel-size', type=int, default=1,
                       help='Degree of tensor model parallelism.')
    group.add_argument('--tp-gemm-parallel', action='store_true',
                       help='when using tensor model parallelism, make gemm-allreduce overlap.')
    group.add_argument('--tp-kernel-split-num', type=int, default=2,
                       help='set attn and mlp layer gemm operation to cut into gemm-kernel number.')
    group.add_argument('--pipeline-model-parallel-size', type=int, default=1,
                       help='Degree of pipeline model parallelism.')
    group.add_argument('--pipeline-model-parallel-split-rank',
                       type=int, default=None,
                       help='Rank where encoder and decoder should be split.')
    group.add_argument('--zero-secondary-parallel-size', type=int, default=None,
                       help='Degree of ZeRO secondary parallelism.')
    group.add_argument('--model-parallel-size', type=int, default=None,
                       help='Old model parallel argument, do not use. Use '
                       '--tensor-model-parallel-size instead.')
    group.add_argument('--pipeline-model-parallel-partition-method', default=None,
                       choices=['balanced', 'imbalanced'],
                       help='split pipeline model parallel use partition method.')
    group.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                       help='Number of layers per virtual pipeline stage')
    group.add_argument('--overlap-p2p-communication',
                       action='store_true',
                       help='overlap pipeline parallel communication with forward and backward chunks',
                       dest='overlap_p2p_comm')
    group.add_argument('--no-batch-p2p-communication',
                       action='store_false',
                       help='If set, do not use batch p2p communication for pipeline parallel',
                       dest='batch_p2p_comm')
    group.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--distributed-timeout-minutes', type=int, default=10,
                       help='Timeout minutes for torch.distributed.')
    group.add_argument('--DDP-impl', default='local',
                       choices=['local', 'torch'],
                       help='which DistributedDataParallel implementation '
                       'to use.')
    group.add_argument('--no-contiguous-buffers-in-local-ddp',
                       action='store_false', help='If set, dont use '
                       'contiguous buffer in local DDP.',
                       dest='use_contiguous_buffers_in_local_ddp')
    group.add_argument('--no-scatter-gather-tensors-in-pipeline', action='store_false',
                       help='Use scatter/gather to optimize communication of tensors in pipeline',
                       dest='scatter_gather_tensors_in_pipeline')
    group.add_argument('--use-ring-exchange-p2p', action='store_true',
                       default=False, help='If set, use custom-built ring exchange '
                       'for p2p communications. Note that this option will require '
                       'a custom built image that support ring-exchange p2p.')
    group.add_argument('--local_rank', '--local-rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')
    group.add_argument('--lazy-mpu-init', type=bool, required=False,
                       help='If set to True, initialize_megatron() '
                       'skips DDP initialization and returns function to '
                       'complete it instead.Also turns on '
                       '--use-cpu-initialization flag. This is for '
                       'external DDP manager.' )
    group.add_argument('--use-cpu-initialization', action='store_true',
                       default=None, help='If set, affine parallel weights '
                       'initialization uses CPU' )
    group.add_argument('--empty-unused-memory-level', default=0, type=int,
                       choices=[0, 1, 2],
                       help='Call get_accelerator().empty_cache() each iteration '
                       '(training and eval), to reduce fragmentation.'
                       '0=off, 1=moderate, 2=aggressive.')
    group.add_argument('--standalone-embedding-stage', action='store_true',
                       default=False, help='If set, *input* embedding layer '
                       'is placed on its own pipeline stage, without any '
                       'transformer layers. (For T5, this flag currently only '
                       'affects the encoder embedding.)')
    group.add_argument('--use-distributed-optimizer', action='store_true',
                       help='Supports ZeRO-1, mutually exclusive with --use-zero-optimizer.')
    group.add_argument('--use-zero-optimizer', action='store_true',
                       default=False, help='Supports ZeRO-1, ZeRO-2 and DP gradient overlapping,'
                       'mutually exclusive with --use-distributed-optimizer.')
    group.add_argument('--zero-stage',  default=1, type=int,
                       choices=[1, 2],
                       help='1=ZeRO-1, 2=ZeRO-2 refer to optimizer state partitioning, '
                       'and optimizer+gradient state partitioning in ZeroOptimizer.')
    group.add_argument('--reduce-bucket-size', default=5e4, type=int,
                       help='Bucket size for reduce in ZeroOptimizer, the unit is the number of elements')
    group.add_argument('--last-bucket-split-count', type=int, default=None, help='Number of \
                       splits to split the last bucket in ZeroOptimizer.')
    group.add_argument('--overlap-param', action='store_true',
                       default=False, help='Supports ZeRO-1, ZeRO-2 and DP param overlapping,'
                       'work with --use-zero-optimizer.')
    group.add_argument('--use-chunked-param-groups', action='store_true',
                       default=False, help='Supports ZeRO-1, ZeRO-2 and DP overlapping '
                       'with interleaved pipeline scheduler, work with --use-zero-optimizer.')

    return parser


def _add_validation_args(parser):
    """_add_validation_args"""
    group = parser.add_argument_group(title='validation')

    group.add_argument('--eval-iters', type=int, default=100,
                       help='Number of iterations to run for evaluation'
                       'validation/test for.')
    group.add_argument('--eval-interval', type=int, default=1000,
                       help='Interval between running evaluation on '
                       'validation set.')
    group.add_argument('--verify-grad-order', action='store_true',
                       default=False, help='Verify the order of grad in each reduced bucket.')

    return parser


@extract_group_args("data_args")
def _add_data_args(parser):
    """_add_data_args"""
    group = parser.add_argument_group(title='data and dataloader')

    group.add_argument('--data-name', type=str, default=None,
                       choices=['wudao', 'stories', 'pile', 'cc-news', 'bert-large',
                                'bert-base', 'wikibook', 'baike', 'zhidao', 'zhihu',
                                'openwebtext', 'wikipedia-key', 'wikipedia'],
                                help='Which data to use.')
    group.add_argument('--data-path', nargs='*', default=None,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ... It is used with --split when a '
                       'single dataset used for all three: train, valid '
                       'and test. It is exclusive to the other '
                       '--*-data-path args')
    group.add_argument('--split', type=str, default='969, 30, 1',
                       help='Comma-separated list of proportions for training,'
                       ' validation, and test split. For example the split '
                       '`90,5,5` will use 90%% of data for training, 5%% for '
                       'validation and 5%% for test.')
    group.add_argument('--train-data-path', nargs='*', default=None,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--valid-data-path', nargs='*', default=None,
                       help='Path to the validation dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--test-data-path', nargs='*', default=None,
                       help='Path to the test dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--vocab-size', type=int, default=None,
                       help='Size of vocab before EOD or padding.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file.')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file.')
    group.add_argument('--vocab-extra-ids', type=int, default=0,
                       help='Number of additional vocabulary tokens. '
                            'They are used for span masking in the T5 model')
    group.add_argument('--seq-length', type=int, default=None,
                       help='Maximum sequence length to process.')
    group.add_argument('--variable-seq-lengths', action='store_true',
                       help='Support for variable sequence lengths across microbatches')
    group.add_argument('--encoder-seq-length', type=int, default=None,
                       help='Maximum encoder sequence length to process.'
                       'This should be exclusive of --seq-length')
    group.add_argument('--decoder-seq-length', type=int, default=None,
                       help="Maximum decoder sequence length to process.")
    group.add_argument('--retriever-seq-length', type=int, default=256,
                       help='Maximum sequence length for the biencoder model '
                       'for retriever')
    group.add_argument('--sample-rate', type=float, default=1.0,
                       help='sample rate for training data. Supposed to be 0 '
                            ' < sample_rate < 1')
    group.add_argument('--mask-prob', type=float, default=0.15,
                       help='Probability of replacing a token with mask.')
    group.add_argument('--short-seq-prob', type=float, default=0.1,
                       help='Probability of producing a short sequence.')
    group.add_argument('--mmap-warmup', action='store_true',
                       help='Warm up mmap files.')
    group.add_argument('--num-workers', type=int, default=2,
                       help="Dataloader number of workers.")
    group.add_argument('--tokenizer-type', type=str,
                       default=None,
                       choices=['BertWordPieceLowerCase',
                                'BertWordPieceCase',
                                'GPT2BPETokenizer',
                                'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer',
                                'LLaMASentencePieceTokenizer',
                                'BloomHFTokenizer',
                                'GalacticaHFTokenizer',
                                'IceTokenizer',
                                'ChatGLMTokenizer',
                                'GLMChineseTokenizer',
                                'OptTokenizer',
                                'ChatGLM2Tokenizer',
                                'LLaMASFTTokenizerHF',
                                'NullTokenizer'],

                       help='What type of tokenizer to use.')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='Sentencepiece tokenizer model.')
    group.add_argument('--hf-tokenizer-path', type=str, default=None,
                       help='HuggingFace tokenizer path: '
                       '1) A string, the *model id* of a predefined tokenizer '
                       'hosted inside a model repo on huggingface.co'
                       '2) A path to a *directory* containing vocabulary '
                       'files required by the tokenizer')
    group.add_argument('--data-impl', type=str, default='infer',
                       choices=['lazy', 'cached', 'mmap', 'infer'],
                       help='Implementation of indexed datasets.')
    group.add_argument('--reset-position-ids', action='store_true',
                       help='Reset posistion ids after end-of-document token.')
    group.add_argument('--reset-attention-mask', action='store_true',
                       help='Reset self attention maske after '
                       'end-of-document token.')
    group.add_argument('--eod-mask-loss', action='store_true',
                       help='Mask loss for the end of document tokens.')

    group.add_argument('--seq2seq-data-collator', action='store_true',
                       help='Use Data Collator For Seq2Seq')

    group.add_argument('--sft-collator', action='store_true', help='Use data collector for sft')
    group.add_argument('--enable-prep-data', action='store_true', help='enable to load preprocess sft data')
    group.add_argument('--sft-data-template', type=str, default='alpaca', help='sft data template')
    group.add_argument('--sft-dataset-name', type=str, default='alpaca_en', help='sft dataset name')
    group.add_argument('--custom-collator', action='store_true', help='Use data collector for sft')

    return parser


def _add_autoresume_args(parser):
    """_add_autoresume_args"""
    group = parser.add_argument_group(title='autoresume')

    group.add_argument('--adlr-autoresume', action='store_true',
                       help='Enable autoresume on adlr cluster.')
    group.add_argument('--adlr-autoresume-interval', type=int, default=1000,
                       help='Intervals over which check for autoresume'
                       'termination signal')

    return parser


def _add_biencoder_args(parser):
    """_add_biencoder_args"""
    group = parser.add_argument_group(title='biencoder')

    # network size
    group.add_argument('--ict-head-size', type=int, default=None,
                       help='Size of block embeddings to be used in ICT and '
                        'REALM (paper default: 128)')
    group.add_argument('--biencoder-projection-dim', type=int, default=0,
                       help='Size of projection head used in biencoder (paper'
                        ' default: 128)')
    group.add_argument('--biencoder-shared-query-context-model', action='store_true',
                        help='Whether to share the parameters of the query '
                        'and context models or not')

    # checkpointing
    group.add_argument('--ict-load', type=str, default=None,
                       help='Directory containing an ICTBertModel checkpoint')
    group.add_argument('--bert-load', type=str, default=None,
                       help='Directory containing an BertModel checkpoint '
                       '(needed to start ICT and REALM)')

    # data
    group.add_argument('--titles-data-path', type=str, default=None,
                       help='Path to titles dataset used for ICT')
    group.add_argument('--query-in-block-prob', type=float, default=0.1,
                       help='Probability of keeping query in block for '
                       'ICT dataset')
    group.add_argument('--use-one-sent-docs', action='store_true',
                       help='Whether to use one sentence documents in ICT')
    group.add_argument('--evidence-data-path', type=str, default=None,
                       help='Path to Wikipedia Evidence frm DPR paper')

    # training
    group.add_argument('--retriever-report-topk-accuracies', nargs='+', type=int,
                        default=[], help="Which top-k accuracies to report "
                        "(e.g. '1 5 20')")
    group.add_argument('--retriever-score-scaling', action='store_true',
                       help='Whether to scale retriever scores by inverse '
                        'square root of hidden size')

    # faiss index
    group.add_argument('--block-data-path', type=str, default=None,
                       help='Where to save/load BlockData to/from')
    group.add_argument('--embedding-path', type=str, default=None,
                       help='Where to save/load Open-Retrieval Embedding'
                        ' data to/from')

    # indexer
    group.add_argument('--indexer-batch-size', type=int, default=128,
                       help='How large of batches to use when doing indexing '
                       'jobs')
    group.add_argument('--indexer-log-interval', type=int, default=1000,
                       help='After how many batches should the indexer '
                       'report progress')
    return parser


def _add_vision_args(parser):
    """_add_vision_args"""
    group = parser.add_argument_group(title="vision")

    # general vision arguements
    group.add_argument('--num-classes', type=int, default=1000,
                       help='num of classes in vision classificaiton task')
    group.add_argument('--img-h', type=int, default=224,
                       help='Image height for vision classification task')
    group.add_argument('--img-w', type=int, default=224,
                       help='Image height for vision classification task')
    group.add_argument('--num-channels', type=int, default=3,
                       help='Number of channels in input image data')
    group.add_argument('--patch-dim', type=int, default=16,
                       help='patch dimension')
    group.add_argument('--classes-fraction', type=float, default=1.0,
                       help='training with fraction of classes.')
    group.add_argument('--data-per-class-fraction', type=float, default=1.0,
                       help='training with fraction of data per class.')
    group.add_argument('--no-data-sharding', action='store_false',
                       help='Disable data sharding.',
                       dest='data_sharding')
    group.add_argument('--head-lr-mult', type=float, default=1.0,
                       help='learning rate multiplier for head during finetuning')

    # pretraining type and backbone selection`
    group.add_argument('--vision-pretraining', action='store_true',
                       help='flag to indicate vision pretraining')
    group.add_argument('--vision-pretraining-type', type=str, default='classify',
                       choices=['classify', 'inpaint', 'dino'],
                       help='pretraining objectives')
    group.add_argument('--vision-backbone-type', type=str, default='vit',
                       choices=['vit', 'mit', 'swin'],
                       help='backbone types types')
    group.add_argument('--swin-backbone-type', type=str, default='tiny',
                       choices=['tiny', 'base', 'h3'],
                       help='pretraining objectives')

    # inpainting arguments
    group.add_argument('--mask-type', type=str, default='random',
                       choices=['random', 'row'],
                       help='mask types')
    group.add_argument('--mask-factor', type=float, default=1.0,
                       help='mask size scaling parameter')

    # dino arguments
    group.add_argument('--iter-per-epoch', type=int, default=1250,
                       help='iterations per epoch')
    group.add_argument('--dino-local-img-size', type=int, default=96,
                       help='Image size for vision classification task')
    group.add_argument('--dino-local-crops-number', type=int, default=10,
                       help='Number of local crops')
    group.add_argument('--dino-head-hidden-size', type=int, default=2048,
                       help='Hidden dimension size in dino head')
    group.add_argument('--dino-bottleneck-size', type=int, default=256,
                       help='Bottle neck dimension in dino head ')
    group.add_argument('--dino-freeze-last-layer', type=float, default=1,
                       help='Freezing last layer weights')
    group.add_argument('--dino-norm-last-layer', action='store_true',
                       help='Disable Norm in last layer.')
    group.add_argument('--dino-warmup-teacher-temp', type=float, default=0.04,
                       help='warump teacher temperature')
    group.add_argument('--dino-teacher-temp', type=float, default=0.07,
                       help='teacher temperature')
    group.add_argument('--dino-warmup-teacher-temp-epochs', type=int, default=30,
                       help='warmup teacher temperaure epochs')
    group.add_argument('--model-type-custom', type=str, required=False, default=None,
                        choices=['chatglm-6b', 'glm-10b', 'glm-130b', 'chatglm2-6b'],
                       help='custom model type such as chatglm-6b, glm-10b, glm-130b')

    return parser

def _add_glm_args(parser):
    """_add_glm_args"""
    group = parser.add_argument_group("GLM", "GLM configurations")
    group.add_argument("--gpt-prob", type=float, default=0.0)
    # group.add_argument("--mask-ratio", type=float, default=0.15)
    group.add_argument("--average-block-length", type=int, default=3)
    group.add_argument("--min-gmask-ratio", type=float, default=0.2)
    group.add_argument(
        "--sandwich-ln",
        action="store_true",
        help="Use Sandwich-LN",
    )
    group.add_argument(
        "--apply-pb-relax",
        action="store_true",
        help="Use pb-relax",
    )
    group.add_argument(
        "--pb-relax-alpha",
        type=int,
        default=32,
        help="Alpha value in pb-relax",
    )
    group.add_argument("--length-per-sample", type=int, default=None)
    group.add_argument("--aggregated-samples-per-sequence", type=int, default=1)
    group.add_argument('--aggregate-gpt-sample', action='store_true')
    group.add_argument('--multitask-data-transform-steps', nargs='*', default=None,
                       help='--multitask-data-transform-steps <x1> <x2>'
                            'multitask ds1 -> ds2 in [x1, x2)')
    group.add_argument('--fast-decode', action='store_true')
    return parser

def _add_blocklm_args(parser):
    """_add_blocklm_args"""
    group = parser.add_argument_group("GLM", "block lm configurations")
    group.add_argument('--block-lm', action='store_true', help="whether use the BlockLM pre-training")
    group.add_argument('--masked-lm', action='store_true', help='whether to use the mlm objective')
    group.add_argument('--bert-prob', type=float, default=0.5)
    group.add_argument('--gpt-infill-prob', type=float, default=0.5)
    group.add_argument('--gpt-min-ratio', type=float, default=0.5)
    group.add_argument('--gap-sentence-prob', type=float, default=0.0)
    group.add_argument('--gap-sentence-ratio', type=float, default=0.15)
    group.add_argument('--single-span-prob', type=float, default=0.0)
    group.add_argument('--task-mask', action='store_true', help="Use different mask for generation and blank filling")
    group.add_argument('--no-shuffle-block', action='store_true', help="not shuffle the blocks when filling the blank")
    group.add_argument('--no-block-position', action='store_true',
                       help='Use (rough) absolute positions instead of block positions')
    group.add_argument('--sentinel-token', action='store_true',
                       help="Use sentinel (mask) tokens to replace 2d position encoding")
    group.add_argument('--block-mask-prob', type=float, default=0.0)
    group.add_argument('--context-mask-ratio', type=float, default=0.0)
    group.add_argument('--random-position', action='store_true',
                       help="Use random start position to cover all the position embeddings")
    group.add_argument('--encoder-decoder', action='store_true',
                       help="use the encoder-decoder architecture for blocklm")
    group.add_argument('--continuous-prompt', action='store_true', help="Use continuous prompt for PET")
    group.add_argument('--prefix-prompt', type=int, default=0)
    group.add_argument('--fix-command-token', action='store_true')
    group.add_argument('--avg-block-length', type=int, default=3)
    return parser

def _add_chatglm_args(parser):
    """_add_chatglm_args"""
    group = parser.add_argument_group("CHATGLM", "Chat-GLM configurations")
    group.add_argument("--prompt-resource-name", type=str, default="content",
                       help='dataset file question key in per line')
    group.add_argument("--prompt-target-name", type=str, default="summary",
                       help='dataset file answer in per line ')
    group.add_argument("--prompt-history-name", type=str, default=None)
    group.add_argument('--preprocessing-num-workers', type=int, default=2,
                       help='preprocessing num workers token trans string to int')
    group.add_argument('--add-qkv-bias', type=bool, required=False,
                       help='add-qkv-bias')
    group.add_argument('--rmsnorm-weight-empty', type=bool, default=False, required=False,
                       help='rmsnorm-weight-empty')
    return parser

def _add_finetune_config_args(parser):
    """_add_finetune_config_args"""
    group = parser.add_argument_group('finetune', 'finetune configurations')
    group.add_argument('--task', type=str, help='Task name.')
    group.add_argument('--load-pretrained', type=str, help="Load pretrained model", default=None)
    group.add_argument('--pool-token', type=str, choices=['start', 'pad', 'cls'],
                       help='The token to pool the sequence representation', default='cls')
    group.add_argument('--cloze-eval', action='store_true', help='Evaluation dataset with cloze task')
    group.add_argument('--multi-token', action='store_true', help='Use multi token for cloze evaluation')
    group.add_argument('--segment-length', type=int, default=0, help="The maximum segment length for cloze evaluation")
    group.add_argument('--loss-func', type=str, choices=["cross_entropy", "hinge", "generative", "mix"],
                       default="cross_entropy")
    group.add_argument('--block-lm-ratio', type=float, default=0.0)
    group.add_argument('--adapet', action='store_true', help="Use the decoupled cross entropy loss in AdaPET")
    group.add_argument('--pattern-id', type=int, default=0)
    group.add_argument('--few-superglue', action='store_true')
    group.add_argument('--eval-valid', action='store_true', help="Whether evaluate on the valid set")
    group.add_argument('--validation-metric', type=str, default=None)
    group.add_argument('--unidirectional', action='store_true', help="Use the left to right language model")
    group.add_argument('--src-seq-length', type=int, default=None)
    group.add_argument('--tgt-seq-length', type=int, default=None)
    group.add_argument('--wsc-negative', action='store_true')
    group.add_argument('--overwrite', action='store_true')
    group.add_argument('--no-validation', action='store_true')
    # Continuous prompt arguments
    group.add_argument('--num-prompt-tokens', type=int, default=0)
    group.add_argument('--prompt-func', default='lstm', choices=["lstm", "mlp", "none"])
    group.add_argument('--freeze-transformer', action='store_true', default=False)
    group.add_argument('--tune-prefix-layers', type=int, default=None)
    group.add_argument('--prompt-init', action='store_true', default=False)
    group.add_argument('--mask-pad-token', action='store_true')
    return parser
