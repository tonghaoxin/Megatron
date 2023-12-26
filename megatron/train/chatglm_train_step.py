""" chatGLM train step """
import torch
from functools import partial
from megatron import (
    get_args,
    get_timers,
    get_tensorboard_writer,
)
from megatron.core import tensor_parallel
from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
)
from megatron.utils import (
    average_losses_across_data_parallel_group,
)
from .abstract_train_step import AbstractTrainStep

def get_masks(input_ids):
    """ get masks for attention """
    batch_size, seq_length = input_ids.shape
    full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
    full_attention_mask.tril_()
    full_attention_mask = (full_attention_mask < 0.5).bool()
    full_attention_mask.unsqueeze_(1)
    return full_attention_mask

def get_position_ids(input_ids, device):
    """ get position ids for attention """
    batch_size, seq_length = input_ids.shape
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
    return position_ids

class ChatGLMTrainStep(AbstractTrainStep):
    """
    ChatGLM train step class.

    Args:
        args (`argparse.Namespace`): Megatron-LM arguments.
    """

    def __init__(self, args):
        super().__init__(self.__class__.__name__)
        self.get_batch = self.get_batch_func(args.megatron_dataset_flag)
        self.loss_func = self.get_loss_func()
        self.forward_step = self.get_forward_step_func()
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            self.model_output_class = Seq2SeqLMOutput

    def get_batch_func(self, megatron_dataset_flag):
        def get_batch(data_iterator):
            """Generate a batch"""
            args = get_args()
            datatype = torch.int64
            # data = next(data_iterator)
            if data_iterator is not None:
                data = next(data_iterator)
                keys = data.keys()
            else:
                data = None
                keys = None
            data_b = tensor_parallel.broadcast_data(keys, data, datatype)
            # Unpack.
            tokens = data_b['input_ids'].long()
            labels = data_b['labels'].long()
            attention_mask = get_masks(tokens)
            position_ids = get_position_ids(tokens, tokens.device)
            # position_ids = data['position_ids'].long()
            loss_mask = (labels != -100).long()

            labels = torch.cat((labels[:, 1:], labels[:, :1]), dim=1)
            loss_mask[:, -1] = 0

            return (tokens, labels, loss_mask), (attention_mask, position_ids)

        return get_batch

    def get_loss_func(self):
        def loss_func(loss_mask, output_tensor):
            """Loss func."""
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])
            return loss, {'lm loss': averaged_loss[0]}
        return loss_func

    def get_forward_step_func(self):
        def forward_step(data_iterator, model):
            """Forward step."""
            args = get_args()
            timers = get_timers()

            # Get the batch.
            timers('batch-generator').start()
            (tokens, labels, loss_mask), (attention_mask, position_ids) = self.get_batch(data_iterator)
            timers('batch-generator').stop()

            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

            return output_tensor[0], partial(self.loss_func, loss_mask)

        return forward_step