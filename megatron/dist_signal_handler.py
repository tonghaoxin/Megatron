import signal
import torch
from megatron.accelerator import get_accelerator


def get_world_size():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    return world_size


def get_device(local_rank=None):
    return get_accelerator().device(local_rank)


def all_gather_item(item, dtype, group=None, async_op=False, local_rank=None):
    """all_gather_item"""
    if not torch.distributed.is_available() or \
       not torch.distributed.is_initialized():
        return [item]

    device = get_device(local_rank)

    if group is not None:
        group_size = group.size()
    else:
        group_size = get_world_size()

    tensor = torch.tensor([item], device=device, dtype=dtype)
    output_tensors = [
        torch.zeros(1, dtype=tensor.dtype, device=tensor.device)
        for _ in range(group_size)
    ]
    if torch.distributed.get_backend(group=group) == "gloo":
        tensor = get_accelerator().d2h_copy(tensor)
        output_tensors = get_accelerator().d2h_copy(output_tensors)
    torch.distributed.all_gather(output_tensors, tensor, group, async_op)
    if torch.distributed.get_backend(group=group) == "gloo":
        tensor = get_accelerator().h2d_copy(tensor)
        output_tensors = get_accelerator().h2d_copy(output_tensors)
    output = [elem.item() for elem in output_tensors]
    return output


class DistributedSignalHandler:
    def __init__(self, sig=signal.SIGTERM):
        self.sig = sig

    def signals_received(self):
        all_received = all_gather_item(
            self._signal_received, dtype=torch.int32
        )
        return all_received

    def __enter__(self):
        self._signal_received = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self._signal_received = True

        signal.signal(self.sig, handler)

        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True
