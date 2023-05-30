""" chatglm sampler and dataloader """
#!/usr/bin/env python3
from torch.utils.data import RandomSampler
import torch
from transformers.utils import (
    logging,
)

from transformers.trainer_utils import (
    seed_worker,
    has_length,
)

from torch.utils.data.distributed import DistributedSampler
from typing import Optional

from megatron import get_args
from megatron.core import mpu

logger = logging.get_logger(__name__)

def get_train_sampler(train_dataset) -> Optional[torch.utils.data.Sampler]:
    if train_dataset is None or not has_length(train_dataset):
        return None
    args = get_args()
    generator = None
    if args.world_size <= 1:
        generator = torch.Generator()
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator.manual_seed(seed)
    
    if args.seed is None:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
    else:
        seed = args.seed

    if args.world_size <= 1:
        return RandomSampler(train_dataset, generator=generator)
    else:
        group = mpu.get_data_parallel_group()
        rank = torch.distributed.get_rank(group=group)
        return DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=rank,
            seed=seed,
        )
