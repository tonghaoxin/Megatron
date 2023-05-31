import argparse
import os
import sys
import torch
import unittest

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, "../../../"))

from megatron.data.glm_data import build_mask_matrix, build_single_mask_matrix

class TestBuildMaskMatrix(unittest.TestCase):

    def test_build_mask_matrix_1(self):
        seq_len = 9
        separator1 = torch.tensor([[6]], dtype=torch.int)
        separator2 = torch.tensor([[1, 0, 2]], dtype=torch.int)
        separator3 = separator1.expand(-1, 3).clone()
        separator3[:,separator1.shape[-1]:] = separator3[:,separator1.shape[-1]:].fill_(-1)
        separator4 = torch.cat((separator3, separator2), dim=0)

        m1 = build_mask_matrix(separator1, batch_size=1, seq_length=seq_len)
        m2 = build_mask_matrix(separator2, batch_size=1, seq_length=seq_len)

        m  = build_mask_matrix(separator4, batch_size=2, seq_length=seq_len)
        m  = torch.chunk(m, 2, dim=0)
        m1_, m2_ = m[0], m[1]

        self.assertTrue(torch.equal(m1, m1_))
        self.assertTrue(torch.equal(m2, m2_))

    def test_build_mask_matrix_2(self):
        seq_len = 2048
        separator1 = torch.tensor([[1032]], dtype=torch.int)
        separator2 = torch.tensor([[456, 457, 458, 459]], dtype=torch.int)
        separator3 = separator1.expand(-1, 4).clone()
        separator3[:,separator1.shape[-1]:] = separator3[:,separator1.shape[-1]:].fill_(-1)
        separator4 = torch.cat((separator3, separator2), dim=0)

        m1 = build_mask_matrix(separator1, batch_size=1, seq_length=seq_len)
        m2 = build_mask_matrix(separator2, batch_size=1, seq_length=seq_len)

        m  = build_mask_matrix(separator4, batch_size=2, seq_length=seq_len)
        m  = torch.chunk(m, 2, dim=0)
        m1_, m2_ = m[0], m[1]

        self.assertTrue(torch.equal(m1, m1_))
        self.assertTrue(torch.equal(m2, m2_))

    def test_build_mask_matrix_3(self):
        seq_len = 2048
        separator1 = torch.tensor([[1032]], dtype=torch.int)
        separator2 = separator1.expand(-1, 4).clone()
        separator2[:,separator1.shape[-1]:] = separator2[:,separator1.shape[-1]:].fill_(-1)

        m1 = build_mask_matrix(separator1, batch_size=1, seq_length=seq_len)
        m2 = build_mask_matrix(separator2, batch_size=1, seq_length=seq_len)

        self.assertTrue(torch.equal(m1, m2))

    def test_build_mask_matrix_4(self):
        seq_len = 2048
        separator1 = torch.tensor([[456, 457, 458, 459]], dtype=torch.int)
        separator2 = separator1.expand(-1, 4).clone()
        separator2[:,separator1.shape[-1]:] = separator2[:,separator1.shape[-1]:].fill_(-1)

        m1 = build_mask_matrix(separator1, batch_size=1, seq_length=seq_len)
        m2 = build_mask_matrix(separator2, batch_size=1, seq_length=seq_len)

        self.assertTrue(torch.equal(m1, m2))


if __name__ == "__main__":
    unittest.main()
