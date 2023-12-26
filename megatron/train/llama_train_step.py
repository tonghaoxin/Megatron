""" llama train step class. """
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

class LlamaTrainStep(AbstractTrainStep):
    """
    Llama train step class.

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
            data = next(data_iterator)
            input_ids = data['input_ids'].long()
            labels = data['labels'].long()
            loss_mask = labels.ne(-100).long()

            if 'attention_mask' in data:
                attention_mask = data['attention_mask'].long()
            else:
                attention_mask = torch.ones(
                    input_ids.shape, dtype=torch.int64, device=input_ids.device
                )
            
            if "position_ids" in data:
                position_ids  = data['position_ids'].long()
            else:
                position_ids = torch.arange(
                    input_ids.shape[1], dtype=torch.long, device=input_ids.device
                )
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

            labels = torch.cat((labels[:, 1:], labels[:, :1]), dim=1)
            loss_mask[:, -1] = 0
            return [input_ids, labels, loss_mask, attention_mask, position_ids]

        return get_batch

    def get_loss_func(self):
        def loss_func(loss_mask, output_tensor):
            """Loss func."""
            # print(f"output_tensor {output_tensor}")
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
            timers = get_timers()

            # Get the batch.
            timers('batch-generator', log_level=2).start()
            tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(
                data_iterator)
            timers('batch-generator').stop()

            output_tensor = model(tokens, position_ids, attention_mask,
                                labels=labels)

            return output_tensor, partial(self.loss_func, loss_mask)

        return forward_step