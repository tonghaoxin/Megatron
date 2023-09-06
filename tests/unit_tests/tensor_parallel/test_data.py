from megatron.core.tensor_parallel.data import broadcast_data
from megatron.accelerator import get_accelerator
import torch
from tests.unit_tests.test_utilities import Utils

def test_broadcast_data():
    Utils.initialize_model_parallel(2,4)
    input_data = {
        0 : get_accelerator().h2d_copy(torch.ones((8,8))) * 0.0,
        1 : get_accelerator().h2d_copy(torch.ones((8,8))) * 1.0,
        2 : get_accelerator().h2d_copy(torch.ones((8,8))) * 2.0,
        3 : get_accelerator().h2d_copy(torch.ones((8,8))) * 3.0,
        4 : get_accelerator().h2d_copy(torch.ones((8,8))) * 4.0,
        5 : get_accelerator().h2d_copy(torch.ones((8,8))) * 5.0,
        6 : get_accelerator().h2d_copy(torch.ones((8,8))) * 6.0,
        7 : get_accelerator().h2d_copy(torch.ones((8,8))) * 7.0
        }
    dtype = torch.float32
    actual_output = broadcast_data([0,1],input_data, dtype)
    assert(torch.equal(actual_output[0], input_data[0]))
    assert(torch.equal(actual_output[1], input_data[1]))
    Utils.destroy_model_parallel()
