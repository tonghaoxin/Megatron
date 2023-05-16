# !/usr/bin/env python3

import argparse
import json
import os
import re
import sys
import types

import torch

from transformers import OPTConfig, OPTForCausalLM, PreTrainedTokenizerFast
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint


def add_checkpointing_args(parser):
    parser.add_argument("--megatron-path", type=str, default=None, help="Base directory of Megatron repository")
    parser.add_argument(
        "--convert_checkpoint_from_megatron_to_transformers",
        action="store_true",
        help=(
            "If True, convert a Megatron checkpoint to a Transformers checkpoint. "
            "If False, convert a Transformers checkpoint to a Megatron checkpoint."
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
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    return parser


def add_megatron_checkpoint_args(parser):
    parser.add_argument(
        "--target_tensor_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The tensor model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_pipeline_model_parallel_size",
        type=int,
        default=1,
        help=(
            "The pipeline model parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_data_parallel_size",
        type=int,
        default=1,
        help=(
            "The data parallel size of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--target_params_dtype",
        type=str,
        default="fp32",
        help=(
            "The dtype of the converted checkpoint. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--make_vocab_size_divisible_by",
        type=int,
        default=128,
        help=(
            "Pad the vocab size to be divisible by this value. "
            "This is added for computational efficieny reasons. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    parser.add_argument(
        "--use_distributed_optimizer",
        action="store_true",
        help=(
            "If True, use the distributed optimizer. "
            "Only used when converting a Transformers checkpoint to a Megatron checkpoint."
        ),
    )
    
    parser.add_argument(
        "--untie_embeddings_and_output_weights",
        type=bool,
        default=False,
        help=(
            "Untie embeddings and output weights. galactica need set to False "
        ),
    )
    return parser


def add_transformers_checkpoint_args(parser):
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help=(
            "The name of the pre-trained tokenizer to save. "
            "If not None, the tokenizer will be saved. "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="10GB",
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )

    return parser


# The simple map of names for "automated" rules.
megatron_to_transformers = {
    # "attention.dense": ".attn.c_proj.",
    "self_attention.dense": ".self_attn.o_proj", #".attn.c_proj.",
    "mlp.dense_h_to_4h": ".mlp.c_fc.",
    "mlp.dense_4h_to_h": ".mlp.c_proj.",
}
transformers_to_megatron = {v[1:-1]: k for k, v in megatron_to_transformers.items()}

tensor_parallel_params = [
    # transformers layers to split across tp ranks
    "self_attn.q_proj.weight",
    #"self_attn.q_proj.bias",
    "self_attn.out_proj.weight",
    #"self_attn.out_proj.bias",
    "fc1.weight",
    #"fc1.bias",
    "fc2.weight",
    #"fc2.bias",
]


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


def get_megatron_sharded_states(args, tp_size, pp_size, pp_rank):
    """
    Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor parallel size, pipeline
    parallel size and pipeline parallel rank.

    Args:
        args (argparse.Namespace): the arguments to the script
        tp_size (int): the tensor parallel size
        pp_size (int): the pipeline parallel size
        pp_rank (int): the pipeline parallel rank
    """
    tp_state_dicts = []
    for i in range(tp_size):
        sub_dir_name = f"mp_rank_{i:02d}" if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}"
        checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir_name))[0]
        checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dicts.append(state_dict)
    return tp_state_dicts


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


def transformers_to_megatron_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    # Input is [num_splits * num_heads * hidden_size, :]
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


# TODO: Now not support llama model
def convert_checkpoint_from_megatron_to_transformers(args):
    pass


def convert_checkpoint_from_transformers_to_megatron(args):
    """
    Convert a checkpoint from HuggingFace Transformers to Megatron-LM. This allows converted checkpoints with variable
    tensor parallelism and pipeline parallelism sizes. It takes as input a checkpoint from HuggingFace Transformers
    which can have multiple shards.

    Args:
        args (argparse.Namespace): the arguments to the script

    """
    os.makedirs(args.save_path, exist_ok=True)
    # Search in directory above this
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        from megatron.tokenizer.tokenizer import _vocab_size_with_padding
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron-path. Exiting.")
        exit(1)

    # load the transformers model state dict and config
    sub_dirs = [x for x in os.listdir(args.load_path) if x.startswith("pytorch_model")]
    if len(sub_dirs) == 1:
        checkpoint_name = "pytorch_model.bin"
        state_dict = torch.load(os.path.join(args.load_path, checkpoint_name), map_location="cpu")
    else:
        num_checkpoints = len(sub_dirs) - 1
        state_dict = merge_transformers_sharded_states(args.load_path, num_checkpoints)

    config = OPTConfig.from_pretrained(args.load_path)
    print(f"--------- Model hf config: ----------- \n {config}")
    
    # Saving the tracker file
    tracker_filepath = os.path.join(args.save_path, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as f:
        f.write("release")

    # create `release` dir in args.load_path
    release_dir = os.path.join(args.save_path, "release")
    os.makedirs(release_dir, exist_ok=True)

    # megatron args
    megatron_args = {
        "orig_vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "ffn_hidden_size": config.ffn_dim,
        "tensor_model_parallel_size": args.target_tensor_model_parallel_size,
        "pipeline_model_parallel_size": args.target_pipeline_model_parallel_size,
        "data_parallel_size": args.target_data_parallel_size,
        "make_vocab_size_divisible_by": args.make_vocab_size_divisible_by,
        "rank": 0,
        "tokenizer_type": "GalacticaHFTokenizer",
    }

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

    # Embedding layer
    print("converting embedding layer")
    print("converting embed_tokens.weight")
    #print("state_dict:", state_dict.keys())
    word_embedding = state_dict["model.decoder.embed_tokens.weight"].to(dtype)
    embed_positions = state_dict["model.decoder.embed_positions.weight"].to(dtype)
    print(f"embed_positions size: {embed_positions.shape}")

    orig_vocab_size = config.vocab_size
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
    for i in range(args.target_tensor_model_parallel_size):
        word_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.language_model.embedding.word_embeddings"
        )
        word_emb_dict["weight"] = out_word_embed[i]

        # embed_positions
        position_embeddings_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.language_model.embedding.position_embeddings"
        )
        position_embeddings_dict["weight"] = embed_positions

    #print("word_emb_dict:", word_emb_dict)
    #print("word_emb_dict[weight] size:", word_emb_dict["weight"].shape)

    # Transformer layers
    print("converting transformer layers")
    if config.num_hidden_layers % args.target_tensor_model_parallel_size != 0:
        raise ValueError(
            f"Number of layers ({config.num_hidden_layers}) must be divisible by number of tensor parallelism"
            f" ({args.target_tensor_model_parallel_size})"
        )
    num_layers = config.num_hidden_layers // args.target_pipeline_model_parallel_size
    print("num_layers:", num_layers)

    layer_re = re.compile("model.decoder.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    # The number of heads.
    heads = config.num_attention_heads

    # The hidden_size per head.
    hidden_size_per_head = config.hidden_size // config.num_attention_heads

    for pp_rank in range(args.target_pipeline_model_parallel_size):
        layer_offset = pp_rank * num_layers
        print("layer_offset:", layer_offset)
        if pp_rank > 0:
            output_state_dict = []
            for i in range(args.target_tensor_model_parallel_size):
                output_state_dict.append({})

        for layer in range(num_layers):
            pp_layer_id = layer + layer_offset
            layers_to_copy = [
                layer_name
                for layer_name in state_dict.keys()
                if layer_name.startswith(f"model.decoder.layers.{pp_layer_id}.")
            ]

            for layer_name in layers_to_copy:
                
                print(f"======= start convert layer: {layer_name} =======")
                
                m = layer_re.match(layer_name)
                # Stop if that's not a layer
                if m is None:
                    print(f"Warning -- not match layer for {layer_name}")
                    break

                # The index of the layer.
                layer_index = int(m.group(1))
                # The name of the operation.
                op_name = m.group(2)
                # Is it a weight or a bias?
                weight_or_bias = m.group(3)

                params = state_dict[layer_name].to(dtype)
                # handle layernorm
                if op_name == "self_attn_layer_norm":
                    layer_name = f"layers.{layer}.input_layernorm.{weight_or_bias}"
                    print(f"==== find {op_name}, to {layer_name} ====")

                elif op_name == "final_layer_norm":
                    layer_name = f"layers.{layer}.post_attention_layernorm.{weight_or_bias}"
                    print(f"==== find {op_name}, to {layer_name} ====")

                elif op_name.startswith("self_attn.q_proj") and weight_or_bias == "weight":
                    # handle k_proj, v_proj
                    layer_name = f"layers.{layer}.self_attention.query_key_value.{weight_or_bias}"
                    
                    k_hf_layer = f"model.decoder.layers.{layer_index}.self_attn.k_proj.{weight_or_bias}"
                    v_hf_layer = f"model.decoder.layers.{layer_index}.self_attn.v_proj.{weight_or_bias}"
                    
                    k_params = state_dict[k_hf_layer].to(dtype)
                    v_params = state_dict[v_hf_layer].to(dtype)
                    
                    params = torch.cat([params, k_params, v_params], dim=0)
                    
                    params = transformers_to_megatron_fix_query_key_value_ordering(
                        params,
                        3.0,
                        3,
                        heads,
                        hidden_size_per_head,
                    )
                    
                    print(f"==== find {op_name}, k {k_hf_layer}, v {v_hf_layer} wob {weight_or_bias} to {layer_name} ====")

                elif op_name.startswith("fc1") and weight_or_bias == "weight":
                    layer_name = f"layers.{layer}.mlp.dense_h_to_4h.{weight_or_bias}"
                    print(f"==== find {op_name}, {weight_or_bias} to {layer_name} ====")

                elif op_name.startswith("fc2") and weight_or_bias == "weight":
                    layer_name = f"layers.{layer}.mlp.dense_4h_to_h.{weight_or_bias}"
                    print(f"==== find {op_name}, {weight_or_bias} to {layer_name} ====")

                elif op_name.startswith("self_attn.out_proj") and weight_or_bias == "weight":
                    layer_name = f"layers.{layer}.attention.dense.{weight_or_bias}"
                    print(f"==== find {op_name}, {weight_or_bias} to {layer_name} ====")

                # skip
                else:
                    continue
    
                if weight_or_bias != "weight":
                    print("torch.count_nonzero(params):",torch.count_nonzero(params))

                if op_name + "." + weight_or_bias in tensor_parallel_params:
                    dim = 1 if op_name in ["self_attn.out_proj", "fc2"] and weight_or_bias == "weight" else 0
                    #print("params.shape:", params.shape)
                    #print("dim:", dim)
                    params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=dim)
                    #print("after chunk params.shape:", params[0].shape)

                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.encoder")
                    params_dict[layer_name] = (
                        params[i] if (op_name + "." + weight_or_bias in tensor_parallel_params) else params
                    )

        if pp_rank == args.target_pipeline_model_parallel_size - 1:
            # handle final layernorm
            # weight_or_bias = "weight"
            for weight_or_bias in ["weight", "bias"]:
                params = state_dict[f"model.decoder.final_layer_norm.{weight_or_bias}"].to(dtype)
                layer_name = f"final_layernorm.{weight_or_bias}"
                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.encoder")
                    params_dict[layer_name] = params

            # add the LM head
            head_embed = state_dict[f"lm_head.weight"].to(dtype)

            orig_vocab_size = config.vocab_size
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

            head_out_embed = torch.chunk(full_head_embed, args.target_tensor_model_parallel_size, dim=0)
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.word_embeddings_for_head")
                params_dict["weight"] = head_out_embed[i]

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
    parser = argparse.ArgumentParser()
    parser = add_checkpointing_args(parser)
    parser = add_megatron_checkpoint_args(parser)
    parser = add_transformers_checkpoint_args(parser)
    args = parser.parse_args()
    if args.convert_checkpoint_from_megatron_to_transformers:
        convert_checkpoint_from_megatron_to_transformers(args)
    else:
        convert_checkpoint_from_transformers_to_megatron(args)


if __name__ == "__main__":
    main()
