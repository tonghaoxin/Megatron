import argparse
import os
import sys
import torch
import unittest

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, "../../../../"))

from megatron.core.pipeline_parallel.utils import partition_balanced, partition_imbalanced

class TestPartition(unittest.TestCase):

    def test_partition_balanced(self):
        num_layers = 70
        num_parts = 6
        layer_list, layer_index = partition_balanced(num_layers, num_parts)

        print(layer_list)
        print(layer_index)

        self.assertTrue([12, 12, 12, 12, 12, 10] == layer_list)

    def test_partition_imbalanced(self):
        num_layers = 70
        num_parts = 6
        layer_list, layer_index = partition_imbalanced(num_layers, num_parts)

        print(layer_list)
        print(layer_index)

        self.assertTrue([11, 12, 12, 12, 12, 11] == layer_list)


if __name__ == "__main__":
    unittest.main()
