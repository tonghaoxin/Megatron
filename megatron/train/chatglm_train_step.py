""" chatGLM train step """
import torch
from functools import partial
from megatron import get_timers
from megatron.core import tensor_parallel
from megatron import pretrain_chatglm2
from .abstract_train_step import AbstractTrainStep

def get_masks(input_ids):
    """ get masks for attention """
    batch_size, seq_length = input_ids.shape
    full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
    full_attention_mask.tril_()
    full_attention_mask = (full_attention_mask < 0.5).bool()
    full_attention_mask.unsqueeze_(1)
    return full_attention_mask

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
        if args.model_return_dict:
            from transformers.modeling_outputs import Seq2SeqLMOutput
            self.model_output_class = Seq2SeqLMOutput

    def get_batch_func(self, megatron_dataset_flag):
        def get_batch(data_iterator):
            """Generate a batch"""
            data = next(data_iterator)
            # Unpack.
            tokens = data['input_ids'].long()
            labels = data['labels'].long()
            attention_mask = get_masks(tokens)
            position_ids = pretrain_chatglm2.get_position_ids(tokens, tokens.device)

            labels = torch.cat((labels[:, 1:], labels[:, :1]), dim=1)
            loss_mask = (labels != -100).long()
            loss_mask[:, -1] = 0

            return (tokens, labels, loss_mask), (attention_mask, position_ids)
        
        if megatron_dataset_flag:
            return pretrain_chatglm2.get_batch
        else:
            return get_batch

    def get_loss_func(self):
        return pretrain_chatglm2.loss_func

    def get_forward_step_func(self):
        def forward_step(data_iterator, model):
            """Forward step."""
            timers = get_timers()

            # Get the batch.
            timers('batch-generator').start()
            (tokens, labels, loss_mask), (attention_mask, position_ids) = self.get_batch(data_iterator)
            timers('batch-generator').stop()

            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

            return output_tensor[0], partial(self.loss_func, loss_mask)

        return forward_step