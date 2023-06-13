# !/usr/bin/env python3

import argparse
import json
import os
import re
import sys
import types

import torch

from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint


def _validate_args(args):
    assert args.untie_embeddings_and_output_weights, 'Untie embeddings and output weights is required for LLaMA'
    #TODO: implement the load/save support for rotary positional embedding
    # assert args.use_rotary_position_embeddings, 'Rotary positional embedding is required for LLaMA'


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser = add_checkpointing_args(parser)
    parser = add_megatron_checkpoint_args(parser)
    parser = add_transformers_checkpoint_args(parser)

    args = parser.parse_args()
    _validate_args(args)

    return args


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
    parser.add_argument(
        "--untie_embeddings_and_output_weights", # FIXME
        action="store_true",
        help="Untie embeddings and output weights. Required for LLaMA.",
    )
    #TODO: implement the load/save support for rotary positional embedding
    # parser.add_argument(
    #     "--use-rotary-position-embeddings",
    #     action="store_true",
    #     help="Use rotary positional embeddings. Required for LLaMA",
    # )
    parser.add_argument(
        "--print-checkpoint-structure",
        action="store_true",
        help="Print checkpoint structure recursively.",
    )
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
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to LLaMA tokenizer.model.",
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
    # megatron-lm layers to merge across tp ranks
    "self_attention.query_key_value.weight",
    "self_attention.query_key_value.bias",
    "self_attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight",
    # deprecated
    "attention.query_key_value.weight",
    "attention.query_key_value.bias",
    "attention.dense.weight",
    # transformers layers to split across tp ranks
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
    # huggingface llama
    "self_attn.q_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.down_proj.weight",
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


def megatron_to_transformers_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :] for compatibility with later versions
    of NVIDIA Megatron-LM. The inverse operation is performed inside Megatron-LM to read checkpoints:
    https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209 If param is the weight tensor of the
    self-attention block, the returned tensor will have to be transposed one more time to be read by HuggingFace GPT2.
    This function is taken from `convert_megatron_gpt2_checkpoint.py`

    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (int): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


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


def convert_checkpoint_from_megatron_to_transformers(args):
    """
    Convert a checkpoint from Megatron-LM to HuggingFace Transformers. This handles Megatron checkpoints
    with different tensor parallelism and pipeline parallelism sizes. It saves the converted checkpoint into shards
    using HuggingFace Transformers checkpoint sharding functionality.

    Args:
        args (argparse.Namespace): the arguments to the script
    """
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    # Search in directory above this
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    # Search in directory above this
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    # Load Megatron-LM checkpoint arguments from the state dict
    sub_dirs = os.listdir(args.load_path)
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000"]
    rank0_checkpoint_name = None
    rank0_checkpoint_path = None
    for sub_dir in possible_sub_dirs:
        if sub_dir in sub_dirs:
            rank0_checkpoint_name = os.listdir(os.path.join(args.load_path, sub_dir))[0]
            rank0_checkpoint_path = os.path.join(args.load_path, sub_dir, rank0_checkpoint_name)
            break
    print(f"Loading Megatron-LM checkpoint arguments from: {rank0_checkpoint_path}")
    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")
    megatron_args = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint insead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    # params dtype
    if args.target_params_dtype == "fp16":
        dtype = torch.float16
        dtype_str = "float16"
    elif args.target_params_dtype == "bf16":
        dtype = torch.bfloat16
        dtype_str = "bfloat16"
    else:
        dtype = torch.float32
        dtype_str = "float32"

    # Create Transformers LLaMA config from Megatron-LM arguments
    vocab_size = (
        megatron_args.padded_vocab_size
        if getattr(megatron_args, "orig_vocab_size", None) is None
        else megatron_args.orig_vocab_size
    )
    print(vocab_size)

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=megatron_args.hidden_size,
        intermediate_size=megatron_args.ffn_hidden_size,
        num_hidden_layers=megatron_args.num_layers,
        num_attention_heads=megatron_args.num_attention_heads,
        hidden_act="silu",
        initializer_range=0.02,
        max_sequence_length=2048,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=-1,
        bos_token_id=0,
        eos_token_id=1,
        torch_dtype=dtype_str,
        model_type="llama",
        architectures=["LLaMAForCausalLM"]
    )

    output_state_dict = {}

    checkpoint_version = state_dict.get("checkpoint_version", 0.0)
    tp_size = megatron_args.tensor_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size

    # The regex to extract layer names.
    layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # Convert.
    print("Converting")
    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, 0)

    # Embeddings
    print("Converting embeddings")

    # Convert and store the word embeddings.
    word_embeddings = torch.cat(
        [
            get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], "model.language_model.embedding.word_embeddings.weight"
            )
            for tp_rank in range(tp_size)
        ],
        dim=0,
    )
    word_embeddings = word_embeddings[:vocab_size].to(dtype)
    output_state_dict["model.embed_tokens.weight"] = word_embeddings

    # Transformer Layers
    print("Converting transformer layers")
    # The number of heads.
    heads = config.num_attention_heads
    # The hidden_size per head.
    hidden_size_per_head = config.hidden_size // config.num_attention_heads
    num_layers = config.num_hidden_layers // pp_size

    # Rotary positonal embedding
    try:
        from megatron.model.rotary_pos_embedding import RotaryEmbedding
    except ModuleNotFoundError:
        print("Unable to import megatron.model.rotary_pos_embedding, please verify the Megatron version that you specified. Exiting.")
        exit(1)

    rotary_emb = RotaryEmbedding(hidden_size_per_head)
    rope_invfreq_layer_idx = 0

    for pp_rank in range(pp_size):
        if pp_size > 0:
            print(f"Converting pipeline parallel rank {pp_rank}")
            tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, pp_rank)

        # The transformer.
        path = (
            "model.language_model.transformer"
            if "transformer" in get_element_from_dict_by_path(tp_state_dicts[0], "model.language_model").keys()
            else "model.language_model.encoder"
        )
        # Extract the layers.
        for key, val in get_element_from_dict_by_path(tp_state_dicts[0], path).items():
            # Match the name.
            m = layer_re.match(key)
            # Stop if that's not a layer
            if m is None:
                break

            # The index of the layer.
            layer_idx = int(m.group(1)) + pp_rank * num_layers
            # The name of the operation.
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)

            # The name of the layer.
            layer_name = f"model.layers.{layer_idx}"

            if op_name + "." + weight_or_bias not in tensor_parallel_params:
                params = val.to(dtype)
            else:
                dim = 1 if op_name in ["self_attention.dense", "mlp.dense_4h_to_h"] else 0
                params = torch.cat(
                    [val]
                    + [
                        get_element_from_dict_by_path(tp_state_dicts[tp_rank], f"{path}")[key]
                        for tp_rank in range(1, tp_size)
                    ],
                    dim=dim,
                ).to(dtype)

            # For layernorm(s), simply store the layer norm.
            if op_name.endswith("layernorm"):
                ln_name = "input_layernorm" if op_name.startswith("input") else "post_attention_layernorm"
                output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = params

            # Transpose the QKV matrix.
            elif op_name == "self_attention.query_key_value" \
                    and weight_or_bias == "weight":
                out_val = megatron_to_transformers_fix_query_key_value_ordering(
                    params,
                    checkpoint_version,
                    3,
                    heads,
                    hidden_size_per_head,
                )
                out_val = torch.chunk(out_val, 3, dim=0)
                # Store.
                output_state_dict[layer_name + ".self_attn.q_proj.weight"] = out_val[0]
                output_state_dict[layer_name + ".self_attn.k_proj.weight"] = out_val[1]
                output_state_dict[layer_name + ".self_attn.v_proj.weight"] = out_val[2]

            # Transpose the self_attention.dense matrix
            elif op_name == "self_attention.dense" and weight_or_bias == "weight":
                output_state_dict[layer_name + ".self_attn.o_proj.weight"] = params

            # Transpose the mlp.dense_h_to_4h matrix
            elif op_name == "mlp.dense_h_to_4h" and weight_or_bias == "weight":
                _shape = params.shape
                params = params.view(tp_size,
                                     2,
                                     _shape[0] // (2 * tp_size),
                                     _shape[1])
                params = params.transpose(0, 1).contiguous()
                params = params.view(*_shape)
                params = torch.chunk(params, 2, dim=0)
                output_state_dict[layer_name + ".mlp.gate_proj.weight"] = params[0]
                output_state_dict[layer_name + ".mlp.up_proj.weight"] = params[1]

            # Transpose the mlp.dense_4h_to_h matrix
            elif op_name == "mlp.dense_4h_to_h" and weight_or_bias == "weight":
                output_state_dict[layer_name + ".mlp.down_proj.weight"] = params

            # Manually add rotary positional embedding inv_freq
            if rope_invfreq_layer_idx == layer_idx:
                output_state_dict[layer_name + ".self_attn.rotary_emb.inv_freq"] = params
                rope_invfreq_layer_idx += 1

    if config.num_hidden_layers != (layer_idx + 1):
        raise ValueError(f"Expected {config.num_hidden_layers} layers but found {layer_idx + 1}")

    # The final layernorm.
    print("Converting final layernorm")
    params = get_element_from_dict_by_path(tp_state_dicts[0], str(path))
    output_state_dict["model.norm.weight"] = params["final_layernorm.weight"].to(dtype)
    if params.get("final_layernorm.bias") is not None:
        output_state_dict["model.norm.bias"] = params["final_layernorm.bias"].to(dtype)

    print("Converting LM head")
    layer_name = "model.language_model.output_layer.weight" if args.untie_embeddings_and_output_weights \
            else "model.word_embeddings_for_head.weight"
    lm_head_embeddings = torch.cat(
        [
            get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], layer_name
            )
            for tp_rank in range(tp_size)
        ],
        dim=0,
    )
    output_state_dict["lm_head.weight"] = lm_head_embeddings

    # It should be done!
    print("Conversion from Megatron-LM to Transformers is done!")

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # Add tokenizer class info to config
    # see https://github.com/huggingface/transformers/issues/13906)

    if args.tokenizer_name is None:
        tokenizer_name = "decapoda-research/llama-7b-hf"
    else:
        tokenizer_name = args.tokenizer_name

    tokenizer = LlamaTokenizer(args.tokenizer_path)
    config.tokenizer_class = "LLaMATokenizer"

    # Store the config to file.
    print("Saving config")
    config.save_pretrained(args.save_path)

    # Save tokenizer based on args
    if args.tokenizer_name is not None:
        print(f"Adding {config.tokenizer_class} tokenizer files")
        tokenizer.save_pretrained(args.save_path)

    # Store the state_dict to file.
    max_shard_size = int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)

    # Save the model
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(args.save_path, shard_file))

    if index is None:
        print(f"Model weights saved in {os.path.join(args.save_path, WEIGHTS_NAME)}")
    else:
        save_index_file = os.path.join(args.save_path, WEIGHTS_INDEX_NAME)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        print(
            f"The model is bigger than the maximum size per checkpoint ({args.max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )
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
    # print(f"----sub_dirs: {sub_dirs}, len: {len(sub_dirs)}")
    if len(sub_dirs) == 1:
        checkpoint_name = "pytorch_model.bin"
        state_dict = torch.load(os.path.join(args.load_path, checkpoint_name), map_location="cpu")
    else:
        num_checkpoints = len(sub_dirs) - 1
        state_dict = merge_transformers_sharded_states(args.load_path, num_checkpoints)

    # config = GPT2Config.from_pretrained(args.load_path)

    config = LlamaConfig.from_pretrained(args.load_path)

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
        "ffn_hidden_size": config.intermediate_size,
        "tensor_model_parallel_size": args.target_tensor_model_parallel_size,
        "pipeline_model_parallel_size": args.target_pipeline_model_parallel_size,
        "data_parallel_size": args.target_data_parallel_size,
        "make_vocab_size_divisible_by": args.make_vocab_size_divisible_by,
        "rank": 0,
        "tokenizer_type": "LLaMASentencePieceTokenizer",
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

    word_embedding = state_dict["model.embed_tokens.weight"].to(dtype)

    orig_vocab_size = config.vocab_size
    padded_vocab_size = _vocab_size_with_padding(orig_vocab_size, margs)
    # padded_vocab_size = 32256 # just for megatron
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
        word_emb_dict["weight"] = out_word_embed[i].clone()

    # Transformer layers
    print("converting transformer layers")
    if config.num_hidden_layers % args.target_tensor_model_parallel_size != 0:
        raise ValueError(
            f"Number of layers ({config.num_hidden_layers}) must be divisible by number of tensor parallelism"
            f" ({args.target_tensor_model_parallel_size})"
        )
    num_layers = config.num_hidden_layers // args.target_pipeline_model_parallel_size

    layer_re = re.compile("model.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    # The number of heads.
    heads = config.num_attention_heads

    # The hidden_size per head.
    hidden_size_per_head = config.hidden_size // config.num_attention_heads

    for pp_rank in range(args.target_pipeline_model_parallel_size):
        layer_offset = pp_rank * num_layers
        if pp_rank > 0:
            output_state_dict = []
            for i in range(args.target_tensor_model_parallel_size):
                output_state_dict.append({})

        for layer in range(num_layers):
            pp_layer_id = layer + layer_offset
            layers_to_copy = [
                layer_name
                for layer_name in state_dict.keys()
                if layer_name.startswith(f"model.layers.{pp_layer_id}.")
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
                if op_name == "input_layernorm" or op_name == "post_attention_layernorm":
                    layer_name = f"layers.{layer}.{op_name}.{weight_or_bias}"
                    print(f"==== find {op_name}, to {layer_name} ====")

                elif op_name.startswith("self_attn.q_proj") and weight_or_bias == "weight":
                    # handle k_proj, v_proj
                    layer_name = f"layers.{layer}.self_attention.query_key_value.{weight_or_bias}"

                    k_hf_layer = f"model.layers.{layer_index}.self_attn.k_proj.weight"
                    v_hf_layer = f"model.layers.{layer_index}.self_attn.v_proj.weight"

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

                    print(f"==== find {op_name}, k {k_hf_layer}, v {v_hf_layer} to {layer_name} ====")

                elif op_name.startswith("self_attn.o_proj") and weight_or_bias == "weight":
                    layer_name = f"layers.{layer}.self_attention.dense.{weight_or_bias}"
                    print(f"==== find {op_name}, to {layer_name} ====")

                elif op_name.startswith("mlp.gate_proj") and weight_or_bias == "weight":
                    layer_name = f"layers.{layer}.mlp.dense_h_to_4h.{weight_or_bias}"

                    up_hf_layer = f"model.layers.{layer_index}.mlp.up_proj.weight"
                    up_params = state_dict[up_hf_layer].to(dtype)
                    params = torch.cat([params, up_params], dim=0)

                    _shape = params.shape
                    params = params.view(2,
                                         args.target_tensor_model_parallel_size,
                                         _shape[0] // (2 * args.target_tensor_model_parallel_size),
                                         _shape[1])
                    params = params.transpose(0, 1).contiguous()
                    params = params.view(*_shape)

                    print(f"==== find {op_name}, up {up_hf_layer}, to {layer_name}, shape {_shape} ====")

                elif op_name.startswith("mlp.down_proj") and weight_or_bias == "weight":
                    layer_name = f"layers.{layer}.mlp.dense_4h_to_h.{weight_or_bias}"

                    print(f"==== find {op_name}, to {layer_name} ====")

                # skip
                else:
                    continue

                if op_name + "." + weight_or_bias in tensor_parallel_params:
                    dim = 1 if op_name in ["attn.c_proj", "mlp.c_proj", "self_attn.o_proj", "mlp.down_proj"] else 0
                    params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=dim)

                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.encoder")
                    params_dict[layer_name] = (
                        params[i].clone() if (op_name + "." + weight_or_bias in tensor_parallel_params) else params
                    )

        if pp_rank == args.target_pipeline_model_parallel_size - 1:
            # handle final layernorm
            # for weight_or_bias in ["weight", "bias"]:
            weight_or_bias = "weight"
            params = state_dict[f"model.norm.{weight_or_bias}"].to(dtype)
            layer_name = f"final_layernorm.{weight_or_bias}"
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.encoder")
                params_dict[layer_name] = params

            # add the LM head
            head_embed = state_dict[f"lm_head.weight"].to(dtype)

            orig_vocab_size = config.vocab_size
            padded_vocab_size = _vocab_size_with_padding(orig_vocab_size, margs)
            # padded_vocab_size = 32256 # just for megatron

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
                if not args.untie_embeddings_and_output_weights:
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.word_embeddings_for_head")
                else:
                    # only weight
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.output_layer")

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
    if args.convert_checkpoint_from_megatron_to_transformers:
        convert_checkpoint_from_megatron_to_transformers(args)
    else:
        convert_checkpoint_from_transformers_to_megatron(args)


if __name__ == "__main__":
    main()
