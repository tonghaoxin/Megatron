import argparse
import os
import sys
import torch
import unittest

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, "../../../"))

from megatron.accelerator.custom_ops.cuda_ops import load
from megatron.model.rotary_pos_embedding import RotaryEmbeddingWithPositionIds
from megatron.model.rotary_pos_embedding import apply_rotary_pos_emb_index, apply_rotary_pos_emb_index_fused


class TestRotaryPositionalEmbedding(unittest.TestCase):

    def test_apply_rotary_pos_emb_index(self):
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.rank = 0
        args.masked_softmax_fusion = False
        args.apply_rotary_positional_embedding_kernel = True
        load(args)

        apply_rotary_fn_1 = apply_rotary_pos_emb_index
        apply_rotary_fn_2 = apply_rotary_pos_emb_index_fused

        seq_length = 2048
        hidden_size = 2048
        num_heads = 32
        hidden_size_per_head = hidden_size // num_heads
        rotary_emb = RotaryEmbeddingWithPositionIds(hidden_size_per_head, precision=torch.float16)

        query_layer = torch.rand(seq_length, 1, num_heads, hidden_size_per_head, dtype=torch.float16)
        key_layer = torch.rand(seq_length, 1, num_heads, hidden_size_per_head, dtype=torch.float16)
        value_layer = torch.rand(seq_length, 1, num_heads, hidden_size_per_head, dtype=torch.float16)
        position_ids = torch.arange(0, seq_length, dtype=torch.int64).unsqueeze(dim=0).transpose(0, 1)
        rotary_pos_emb = rotary_emb(value_layer, seq_len=position_ids.max() + 1)
        q_pos_emb, k_pos_emb = rotary_pos_emb
        # ensure that q_pos_emb and k_pos_emb are of data type torch.float16
        q_pos_emb = q_pos_emb.to(torch.float16)
        k_pos_emb = k_pos_emb.to(torch.float16)

        query_layer_1, key_layer_1 = apply_rotary_fn_1(query_layer,
                                                       key_layer,
                                                       q_pos_emb,
                                                       k_pos_emb,
                                                       position_ids)

        query_layer_2, key_layer_2 = apply_rotary_fn_2(query_layer.cuda(),
                                                       key_layer.cuda(),
                                                       q_pos_emb.cuda(),
                                                       k_pos_emb.cuda(),
                                                       position_ids.cuda())

        self.assertTrue(torch.equal(query_layer_1.cuda(), query_layer_2))
        self.assertTrue(torch.equal(key_layer_1.cuda(), key_layer_2))


if __name__ == "__main__":
    unittest.main()
