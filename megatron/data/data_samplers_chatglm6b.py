""" chatglm sampler and dataloader """
#!/usr/bin/env python3
from megatron.global_vars import get_tokenizer
from torch.utils.data import DataLoader, RandomSampler
import torch
from transformers.utils import (
    logging,
)

from transformers.trainer_utils import (
    seed_worker,
    has_length,
)

from transformers import (
    DataCollatorForSeq2Seq
)

from torch.utils.data.distributed import DistributedSampler
from typing import Optional

from megatron import get_args

logger = logging.get_logger(__name__)

def _get_train_sampler(train_dataset, args) -> Optional[torch.utils.data.Sampler]:
    if train_dataset is None or not has_length(train_dataset):
        return None

    generator = None
    if args.world_size <= 1:
        generator = torch.Generator()
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator.manual_seed(seed)
    
    if args.seed is None:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
    else:
        seed = args.seed

    # else:
    if args.world_size <= 1:
        return RandomSampler(train_dataset, generator=generator)
    else:
        return DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=None,
            #rank=args.process_index,
            seed=seed,
        )


def build_pretraining_data_loader(train_dataset, consumed_samples) -> DataLoader:
    """
    Returns the training [`~torch.utils.data.DataLoader`].

    Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
    training if necessary) otherwise.

    Subclass and override this method if you want to inject some custom behavior.
    """
    if train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    args = get_args()

    tokenizer = get_tokenizer()
    data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=None,
            label_pad_token_id=-100,
            pad_to_multiple_of=None,
            padding=False
        )

    # 构造sampler, 再构造DataLoader
    train_sampler = _get_train_sampler(train_dataset, args)

    return DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
