""" Baichuan train step class. """
import torch
from functools import partial
from megatron import get_timers
from megatron import pretrain_baichuan
from .abstract_train_step import AbstractTrainStep

class BaichuanTrainStep(AbstractTrainStep):
    """
    Baichuan train step class.

    Args:
        args (`argparse.Namespace`): Megatron-LM arguments.
    """

    def __init__(self, args):
        super().__init__(self.__class__.__name__)
        self.get_batch = self.get_batch_func(args.megatron_dataset_flag)
        self.loss_func = self.get_loss_func()
        self.forward_step = self.get_forward_step_func()
        if args.model_return_dict:
            from transformers.modeling_outputs import Seq2SeqLMOutput
            self.model_output_class = Seq2SeqLMOutput

    def get_batch_func(self, megatron_dataset_flag):
        """ get_batch_func """
        def get_batch(data_iterator):
            """Generate a batch"""
            data = next(data_iterator)
            input_ids = data['input_ids'].long()
            labels = data['labels'].long()

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
            loss_mask = labels.ne(-100).long()
            loss_mask[:, -1] = 0
            return [input_ids, labels, loss_mask, attention_mask, position_ids]

        if megatron_dataset_flag:
            return pretrain_baichuan.get_batch
        else:
            return get_batch

    def get_loss_func(self):
        """ get_loss_func """
        return pretrain_baichuan.loss_func

    def get_forward_step_func(self):
        """ get_forward_step_func """
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