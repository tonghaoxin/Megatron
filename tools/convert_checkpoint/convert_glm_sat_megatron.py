# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import types
import torch
import argparse
import collections

from tqdm import tqdm

#=============================
# global variables
#=============================

name_map = {
    "input_layernorm": "input_layernorm",
    "attention.query_key_value": "self_attention.query_key_value",
    "attention.dense": "self_attention.dense",
    "post_attention_layernorm": "post_attention_layernorm",
    "mlp.dense_h_to_4h": "mlp.dense_h_to_4h",
    "mlp.dense_4h_to_h": "mlp.dense_4h_to_h",
    "attention.rotary_emb.inv_freq": "self_attention.rotary_emb.inv_freq",
    "final_layernorm": "final_layernorm",
}

LAYER_CONCAT_DIM = {
    "attention.dense.weight": 1,
    "mlp.dense_4h_to_h.weight": 1,
}

SEQUENTIAL_LAYERS = [
    "input_layernorm.weight",
    "input_layernorm.bias",
    "attention.dense.bias",
    "post_attention_layernorm.weight",
    "post_attention_layernorm.bias",
    "mlp.dense_4h_to_h.bias",
    "final_layernorm.weight",
    "final_layernorm.bias",
    "attention.rotary_emb.inv_freq",
]

# GLU_LAYERS = [
#     "mlp.dense_h_to_4h.weight",
#     "mlp.dense_h_to_4h.bias",
# ]


def parse_arguments():
    parser = argparse.ArgumentParser()

    add_checkpointing_args(parser)
    add_megatron_checkpoint_args(parser)

    return parser.parse_args()


def add_checkpointing_args(parser):
    parser.add_argument("--megatron-path", type=str, default=None, help="Base directory of Megatron repository")
    parser.add_argument(
        "--convert_checkpoint_from_megatron_to_sat",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a SAT checkpoint. "
            "If False, convert a SAT checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--load_path",
        type=str,
        required=True,
        help="Path to the checkpoint to convert.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to the converted checkpoint.",
    )
    parser.add_argument(
        "--print_checkpoint_structure",
        action="store_true",
        help="print the converted checkpointing structure")


def add_megatron_checkpoint_args(parser):
    parser.add_argument(
        "--target_tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The tensor model parallel size of the converted checkpoint. "
            "Only used when converting a SAT checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_pipeline_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The pipeline model parallel size of the converted checkpoint. "
            "Only used when converting a SAT checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_data_parallel_size",
        type=int,
        default=1,
        help=(
            "The data parallel size of the converted checkpoint. "
            "Only used when converting a SAT checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_params_dtype",
        type=str,
        default="fp32",
        help=(
            "The dtype of the converted checkpoint. "
            "Only used when converting a SAT checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--make_vocab_size_divisible_by",
        type=int,
        default=128,
        help=(
            "Pad the vocab size to be divisible by this value. "
            "This is added for computational efficieny reasons. "
            "Only used when converting a SAT checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--use_distributed_optimizer",
        action="store_true",
        help=(
            "If True, use the distributed optimizer. "
            "Only used when converting a SAT checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--pipeline_model_parallel_partition_balanced",
        action='store_true',
        help=(
            "split pipeline model parallel use partition balanced."
        ),
    )
    parser.add_argument(
        "--model_type",
        default='glm-130b',
        choices=['glm-130b', 'glm-10b-chinese', 'chatglm-6b'],
        help=(
            "The GLM model type of the converted checkpoint."
        ),
    )
    return parser


def recursive_print(name, val, spaces=0):
    """
    Recursively print the structure of a checkpoint. This function is taken from `convert_megatron_gpt2_checkpoint.py`

    Args:
        name (str): the name of the current tensor parameter
        val (Tuple(int)): the shape of the current tensor parameter
        spaces (int): the number of spaces to print before the output for a nested structure
    """
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size(), ":", val.type())
    else:
        print(msg, ":", val)


def merge_sat_sharded_states(args):
    """
    Merge shareded checkpoints from SAT into a single checkpoint.

    Args:
        args: the input arguments
    """
    state_dict = {}
    merged_state_dict = {}
    for pt_file in tqdm(pt_files := sorted(os.listdir(args.load_path))):
        pt_file_cols = pt_file.split('_')
        tp_rank = pt_file_cols[2]

        pt_file_content = torch.load(os.path.join(args.load_path, pt_file), map_location='cpu')
        layer_dict = {}
        for layer_name, layer in pt_file_content['module'].items():
            layer_dict[layer_name] = layer
        state_dict[int(tp_rank)] = layer_dict

    for key in state_dict[0].keys():
        if key.startswith("word_embeddings"):
            col_id = 0
        elif key.startswith("transformer.layers"):
            col_id = 3
        else:
            col_id = 1
        name = ".".join(key.split(".")[col_id:])
        print(f"name: {name}; params.shape: {state_dict[0][key].shape}")
        if name in SEQUENTIAL_LAYERS:
            merged_state_dict[key] = state_dict[0][key]
        else:
            merged_state_dict[key] = torch.cat([sd[key] for _, sd in state_dict.items()],
                                               dim=LAYER_CONCAT_DIM.get(name, 0))
        print(f"after concat name: {name}; params.shape: {merged_state_dict[key].shape}")

    if args.print_checkpoint_structure:
        recursive_print(None, merged_state_dict)

    return merged_state_dict


def merge_transformers_sharded_states(path, num_checkpoints):
    """
    Merge sharded checkpoints from transformers into a single checkpoint.

    Args:
        path (str): the path to the sharded checkpoints
        num_checkpoints (int): the number of checkpoints to merge
    """
    state_dict = {}
    for i in range(1, num_checkpoints + 1):
        checkpoint_path = os.path.join(path, f"pytorch_model-{i:05d}-of-{num_checkpoints:05d}.bin")
        current_chunk = torch.load(checkpoint_path, map_location="cpu")
        state_dict.update(current_chunk)
    return state_dict


def get_element_from_dict_by_path(d, path):
    """
    Get element from dictionary by path. If element is not present, recursively add empty dictionaries.

    Args:
        d (dict): the dictionary to get the element from
        path (list): the path to the element which is delimited by "."
    """
    path = path.split(".")
    for k in path:
        if k not in d:
            d[k] = {}
        d = d[k]
    return d


def get_num_layers_per_pipeline(args, margs):
    total_layers = getattr(margs, 'num_layers', 0)

    try:
        from megatron.core.pipeline_parallel import utils
    except ModuleNotFoundError:
        print("Unable to import megatron.core.pipeline_parallel.utils, please verify the Megatron version that you specified. Exiting.")
        exit(1)

    if not args.pipeline_model_parallel_partition_balanced:
        assert total_layers % args.target_pipeline_model_parallel_size == 0, \
                f"Total number of layers ({total_layers}) must be divisible by number of tensor parallelism {args.target_pipeline_model_parallel_size}"

        num_layers_per_pipeline = total_layers // args.target_pipeline_model_parallel_size
        layer_list = [num_layers_per_pipeline] * args.target_pipeline_model_parallel_size
        layer_index = list(range(0, total_layers+1, num_layers_per_pipeline))
    else:
        layer_list, layer_index = utils.partition_balanced(total_layers, args.target_pipeline_model_parallel_size)

    return layer_list, layer_index


def preprocess(args):
    if args.model_type == 'glm-130b':
        # load SAT model state dict (merged)
        state_dict = merge_sat_sharded_states(args)

        # disable embedding padding workflow
        args.add_embedding_padding = False

        # disable positional embedding
        args.add_position_embedding = False

        # use rotary positional embeddings
        args.use_rotary_position_embeddings = True

        # do not use rotary embedding 2d
        args.use_rotary_embedding_2d = False

        # set megatron args
        megatron_args = {
            "orig_vocab_size": 150009,
            "max_position_embeddings": None,
            "hidden_size": 12288,
            "num_layers": 70,
            "num_attention_heads": 96,
            "ffn_hidden_size": 32768,
            "tensor_model_parallel_size": args.target_tensor_model_parallel_size,
            "pipeline_model_parallel_size": args.target_pipeline_model_parallel_size,
            "data_parallel_size": args.target_data_parallel_size,
            "make_vocab_size_divisible_by": args.make_vocab_size_divisible_by,
            "rank": 0,
            "tokenizer_type": "IceTokenizer",
        }

    elif args.model_type == 'glm-10b-chinese':
        # load SAT model state dict (merged)
        state_dict = merge_sat_sharded_states(args)

        # disable embedding padding workflow
        args.add_embedding_padding = False

        # enable positional embedding
        args.add_position_embedding = True

        # do not use rotary positional embeddings
        args.use_rotary_position_embeddings = False

        # do not use rotary embedding 2d
        args.use_rotary_embedding_2d = False

        # set megatron args
        megatron_args = {
            "orig_vocab_size": 50048,
            "max_position_embeddings": 1024,
            "hidden_size": 4096,
            "num_layers": 48,
            "num_attention_heads": 64,
            "ffn_hidden_size": 16384,
            "tensor_model_parallel_size": args.target_tensor_model_parallel_size,
            "pipeline_model_parallel_size": args.target_pipeline_model_parallel_size,
            "data_parallel_size": args.target_data_parallel_size,
            "make_vocab_size_divisible_by": args.make_vocab_size_divisible_by,
            "rank": 0,
            "tokenizer_type": "ChineseSPTokenizer",
        }

    elif args.model_type == 'chatglm-6b':
        # load transformer model state dict
        sub_dirs = [x for x in os.listdir(args.load_path) if x.startswith("pytorch_model")]
        if len(sub_dirs) == 1:
            checkpoint_name = "pytorch_model.bin"
            state_dict = torch.load(os.path.join(args.load_path, checkpoint_name), map_location="cpu")
        else:
            num_checkpoints = len(sub_dirs) - 1
            state_dict = merge_transformers_sharded_states(args.load_path, num_checkpoints)

        # enable embedding padding workflow
        args.add_embedding_padding = True

        # disable positional embedding
        args.add_position_embedding = False

        # use rotary positional embeddings
        args.use_rotary_position_embeddings = True

        # use rotary embedding 2d
        args.use_rotary_embedding_2d = True

        # set megatron args
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.load_path,
                                            trust_remote_code=True,
                                            max_sequence_length=1024,
                                            position_encoding_2d=args.use_rotary_embedding_2d)
        print(f"--------- Model hf config: ----------- \n {config}")

        megatron_args = {
            "orig_vocab_size": config.vocab_size,
            "max_position_embeddings": None,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_attention_heads": config.num_attention_heads,
            "ffn_hidden_size": config.inner_hidden_size,
            "tensor_model_parallel_size": args.target_tensor_model_parallel_size,
            "pipeline_model_parallel_size": args.target_pipeline_model_parallel_size,
            "data_parallel_size": args.target_data_parallel_size,
            "make_vocab_size_divisible_by": args.make_vocab_size_divisible_by,
            "rank": 0,
            "tokenizer_type": "ChatGLMTokenizer",
        }


    # set up margs
    margs = types.SimpleNamespace()
    for k, v in megatron_args.items():
        setattr(margs, k, v)

    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    setattr(margs, "params_dtype", dtype)

    return args, margs, state_dict


# TODO: Now not support glm model
def convert_checkpoint_from_megatron_to_sat(args):
    pass


def convert_checkpoint_from_sat_to_megatron(args, margs, state_dict):
    """
    Convert a checkpoint from SAT to Megatron-LM. This allows converted checkpoints with variable
    tensor parallelism and pipeline parallelism sizes. It takes as input a checkpoint from SAT
    which can have multiple shards.

    Args:
        args (argparse.Namespace): the arguments to the script
        margs (types.SimpleNamespace): the megatron-related arguments for checkpoint conversion
        state_dict (dict): the merged model state dict loaded from load_path

    """
    os.makedirs(args.save_path, exist_ok=True)
    # Search in directory above this
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    if args.add_embedding_padding:
        try:
            from megatron.tokenizer.tokenizer import _vocab_size_with_padding
        except ModuleNotFoundError:
            print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
            exit(1)

    # Saving the tracker file
    tracker_filepath = os.path.join(args.save_path, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as f:
        f.write("release")

    # create `release` dir in args.load_path
    release_dir = os.path.join(args.save_path, "release")
    os.makedirs(release_dir, exist_ok=True)

    # save dummy optim state dict
    dummy_optim_state_dict = {}
    dummy_optim_state_dict["optimizer"] = {
        "step": 0,
        "param_groups": [
            {
                "lr": 0.0,
                "beta1": 0.0,
                "beta2": 0.0,
                "eps": 0.0,
                "weight_decay": 0.0,
                "correct_bias": False,
                "params": [],
            }
        ],
    }

    if args.use_distributed_optimizer:
        for _pp_rank in range(args.target_pipeline_model_parallel_size):
            for _tp_rank in range(args.target_tensor_model_parallel_size):
                for _dp_rank in range(args.target_data_parallel_size):
                    if args.target_pipeline_model_parallel_size == 1:
                        checkpoint_dir = f"mp_rank_{_tp_rank:02d}_{_dp_rank:03d}"
                    else:
                        checkpoint_dir = f"mp_rank_{_tp_rank:02d}_{_pp_rank:03d}_{_dp_rank:03d}"

                    checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(
                        dummy_optim_state_dict,
                        os.path.join(checkpoint_dir, "optim.pt"),
                    )

    # Convert.
    print("Converting")
    output_state_dict = []
    for i in range(args.target_tensor_model_parallel_size):
        output_state_dict.append({})

    # get params_dtype
    dtype = getattr(margs, 'params_dtype', torch.float32)

    # Embedding layer
    print("converting embedding layer")
    if args.model_type in ['glm-130b', 'chatglm-6b']:
        word_embedding = state_dict["transformer.word_embeddings.weight"].to(dtype)
    elif args.model_type == 'glm-10b-chinese':
        word_embedding = state_dict["word_embeddings.weight"].to(dtype)

    if args.add_position_embedding:
        position_embedding = state_dict["transformer.position_embeddings.weight"].to(dtype)
        block_position_embedding = state_dict["transformer.block_position_embeddings.weight"].to(dtype)
        out_position_embed = torch.chunk(position_embedding, args.target_tensor_model_parallel_size, dim=0)
        out_block_position_embed = torch.chunk(block_position_embedding, args.target_tensor_model_parallel_size, dim=0)

    if args.add_embedding_padding:
        orig_vocab_size = getattr(margs, 'orig_vocab_size', 0)
        padded_vocab_size = _vocab_size_with_padding(orig_vocab_size, margs)
        print(f"Embedding padded vocab size: {padded_vocab_size}, ori vocab size: {orig_vocab_size}")
        setattr(margs, "padded_vocab_size", padded_vocab_size)

        # Cut out extra padding we don't need
        if orig_vocab_size > padded_vocab_size:
            full_word_embed = word_embedding[0:padded_vocab_size, :]
        # Expanding embedding to larger size by replicating final entry
        elif orig_vocab_size < padded_vocab_size:
            padding_size = padded_vocab_size - orig_vocab_size
            full_word_embed = torch.cat((word_embedding, word_embedding[-1].unsqueeze(0).expand(padding_size, -1)))
        # Same size!
        else:
            full_word_embed = word_embedding

        # Split into new tensor model parallel sizes
        out_word_embed = torch.chunk(full_word_embed, args.target_tensor_model_parallel_size, dim=0)
    else:
        out_word_embed = torch.chunk(word_embedding, args.target_tensor_model_parallel_size, dim=0)

    for i in range(args.target_tensor_model_parallel_size):
        # embeddings
        word_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.language_model.embedding.word_embeddings"
        )
        word_emb_dict["weight"] = out_word_embed[i].clone()

        if args.add_position_embedding:
            # positional embeddings
            position_embeddings_dict = get_element_from_dict_by_path(
                output_state_dict[i], "model.language_model.embedding.position_embeddings"
            )
            position_embeddings_dict["weight"] = out_position_embed[i].clone()

            # block positional embeddings
            block_position_embeddings_dict = get_element_from_dict_by_path(
                output_state_dict[i], "model.language_model.embedding.block_position_embeddings"
            )
            block_position_embeddings_dict["weight"] = out_block_position_embed[i].clone()

    # Transformer layers
    print("converting transformer layers")

    # The hidden size per head
    hidden_size = getattr(margs, 'hidden_size', 0)
    heads = getattr(margs, 'num_attention_heads', 1)
    hidden_size_per_head = hidden_size // heads

    # Rotary positional embedding
    if args.use_rotary_position_embeddings:
        try:
            from megatron.model.rotary_pos_embedding import GLMRotaryEmbedding
        except ModuleNotFoundError:
            print("Unable to import megatron.model.rotary_pos_embedding, please verify the Megatron version that you specified. Exiting.")
            exit(1)

        rotary_emb = GLMRotaryEmbedding(
                hidden_size_per_head // (2 if args.use_rotary_embedding_2d else 1),
                base=10000,
                precision=dtype)

    # Obtain num_layers for each pipeline stage
    layer_list, layer_index = get_num_layers_per_pipeline(args, margs)

    layer_re = re.compile("transformer.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    for pp_rank in range(args.target_pipeline_model_parallel_size):
        num_layers = layer_list[pp_rank]
        layer_offset = layer_index[pp_rank]
        print("num_layers:", num_layers)
        print("layer_offset:", layer_offset)
        if pp_rank > 0:
            output_state_dict = []
            for i in range(args.target_tensor_model_parallel_size):
                output_state_dict.append({})

        find_rope_invfreq = False

        for layer in range(num_layers):
            pp_layer_id = layer + layer_offset
            layers_to_copy = [
                layer_name
                for layer_name in state_dict.keys()
                if layer_name.startswith(f"transformer.layers.{pp_layer_id}.")
            ]
            print(layers_to_copy)

            for layer_name in layers_to_copy:

                print(f"======= start convert layer: {layer_name} =======")

                m = layer_re.match(layer_name)
                # Stop if that's not a layer
                if m is None:
                    print(f"Warning -- not match layer for {layer_name}")
                    break

                # The name of the operation.
                op_name = m.group(2)
                # Is it a weight or a bias?
                weight_or_bias = m.group(3)

                params = state_dict[layer_name].to(dtype)

                # obtain the mapped layer name
                if op_name == 'attention.rotary_emb':
                    assert args.use_rotary_position_embeddings == True, \
                f"args.use_rotary_position_embeddings is required to be set to True since we capture the rotary_emb op"
                    op_name = 'attention.rotary_emb.inv_freq'
                    find_rope_invfreq = True

                mapped_op_name = name_map[op_name]

                if mapped_op_name == 'self_attention.rotary_emb.inv_freq':
                    layer_name = f"layers.{layer}.{mapped_op_name}"
                else:
                    layer_name = f"layers.{layer}.{mapped_op_name}.{weight_or_bias}"

                print(f"==== find {op_name}, to {layer_name} ====")

                # count nonzero for bias
                if weight_or_bias != "weight":
                    print("torch.count_nonzero(params):",torch.count_nonzero(params))

                # chunk non-sequential layers according to target tp degree
                if op_name != 'attention.rotary_emb.inv_freq':
                    op_name = op_name + "." + weight_or_bias

                if op_name not in SEQUENTIAL_LAYERS:
                    print(f"op_name: {op_name}; params.shape: {params.shape}")
                    params = torch.chunk(params,
                                         args.target_tensor_model_parallel_size,
                                         dim=LAYER_CONCAT_DIM.get(op_name, 0))
                    print(f"after chunk op_name: {op_name}; params.shape: {params[0].shape}")

                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.encoder")
                    params_dict[layer_name] = (
                        params[i].clone() if (op_name not in SEQUENTIAL_LAYERS) else params
                    )

            # manually create rotary positional embedding inv_freq
            if args.use_rotary_position_embeddings and not find_rope_invfreq:
                op_name = "attention.rotary_emb.inv_freq"
                mapped_op_name = name_map[op_name]
                layer_name = f"layers.{layer}.{mapped_op_name}"
                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.encoder")
                    params_dict[layer_name] = rotary_emb.inv_freq.to(dtype)

        if pp_rank == args.target_pipeline_model_parallel_size - 1:
            # handle final layernorm
            # weight_or_bias = "weight"
            for weight_or_bias in ["weight", "bias"]:
                params = state_dict[f"transformer.final_layernorm.{weight_or_bias}"].to(dtype)
                layer_name = f"final_layernorm.{weight_or_bias}"
                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.encoder")
                    params_dict[layer_name] = params

            # add the LM head
            if state_dict.get("lm_head.weight", None) is None:
                # full_head_embed = full_word_embed
                head_embed = word_embedding
            else:
                head_embed = state_dict[f"lm_head.weight"].to(dtype)

                if args.add_embedding_padding:
                    orig_vocab_size = getattr(margs, 'orig_vocab_size', 0)
                    padded_vocab_size = _vocab_size_with_padding(orig_vocab_size, margs)

                    # Cut out extra padding we don't need
                    if orig_vocab_size > padded_vocab_size:
                        full_head_embed = head_embed[0:padded_vocab_size, :]
                    # Expanding embedding to larger size by replicating final entry
                    elif orig_vocab_size < padded_vocab_size:
                        padding_size = padded_vocab_size - orig_vocab_size
                        full_head_embed = torch.cat((head_embed, head_embed[-1].unsqueeze(0).expand(padding_size, -1)))
                    # Same size!
                    else:
                        full_head_embed = head_embed
                    head_embed = full_head_embed

            head_out_embed = torch.chunk(head_embed, args.target_tensor_model_parallel_size, dim=0)
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.word_embeddings_for_head")
                params_dict["weight"] = head_out_embed[i].clone()

        # saving the state dict as per the tp_rank and pp_rank
        for tp_rank in range(args.target_tensor_model_parallel_size):
            output_state_dict[tp_rank]["checkpoint_version"] = 3.0
            output_state_dict[tp_rank]["args"] = margs
            checkpoint_dir = (
                f"mp_rank_{tp_rank:02d}"
                if args.target_pipeline_model_parallel_size == 1
                else f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
            )
            if args.use_distributed_optimizer:
                checkpoint_name = "model_rng.pt"
            else:
                checkpoint_name = "model_optim_rng.pt"
                output_state_dict[tp_rank]["optimizer"] = dummy_optim_state_dict["optimizer"]
            checkpoint_dir = os.path.join(release_dir, checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            if args.print_checkpoint_structure:
                print(
                    f"Checkpoint structure of model state dict shard belonging to TP rank {tp_rank} and PP rank"
                    f" {pp_rank}:"
                )
                recursive_print(None, output_state_dict[tp_rank])
            torch.save(output_state_dict[tp_rank], checkpoint_path)


def main():
    args = parse_arguments()
    args, margs, state_dict = preprocess(args)

    if args.convert_checkpoint_from_megatron_to_sat:
        convert_checkpoint_from_megatron_to_sat(args)
    else:
        convert_checkpoint_from_sat_to_megatron(args, margs, state_dict)


if __name__ == "__main__":
    main()
