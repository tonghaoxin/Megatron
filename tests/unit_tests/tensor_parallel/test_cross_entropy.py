from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.accelerator import get_accelerator
import torch
from tests.unit_tests.test_utilities import Utils
import numpy as np

def test_vocab_parallel_cross_entropy():
    Utils.initialize_model_parallel(4,2)
    vocab_parallel_logits = get_accelerator().h2d_copy(torch.range(0,7).repeat(16,4))
    target = get_accelerator().h2d_copy(torch.arange(0,32,2))
    output = vocab_parallel_cross_entropy(vocab_parallel_logits, target)
    expected_output = get_accelerator().h2d_copy(
            torch.tensor([10.2309, 8.2309, 6.2309, 4.2309, 10.2309, 8.2309, 6.2309, 4.2309, 10.2309, 8.2309, 6.2309, 4.2309, 10.2309, 8.2309, 6.2309, 4.2309])
    )
    assert(torch.equal(torch.round(expected_output), torch.round(output)))
    Utils.destroy_model_parallel()
