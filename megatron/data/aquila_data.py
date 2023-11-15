# !/usr/bin/env python3
""" aquila dataset """
import cyg_conversation as conversation_lib
from torch.utils.data import Dataset
import numpy as np
import jsonlines
import os
import torch

max_seq_len = 2048


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    unknown_role = "unknown"  # use default unknown role
    roles = {
        "human": conversation_lib.default_conversation.roles[0],  # human role
        "gpt": conversation_lib.default_conversation.roles[1],  # gpt role
    }
    if "instruction" in source and source["instruction"] is not None and len(source["instruction"]) > 0:
        source["instruction"] = (
            BEGIN_SIGNAL
            + conversation_lib.default_conversation.roles[2]
            + ": "
            + source["instruction"]
            + END_SIGNAL
        )
        if get_conversation:
            conversation += source["instruction"]
    for sentence in source["conversations"]:
        sentence_from = sentence["from"].lower()
        sentence["value"] = (
            BEGIN_SIGNAL
            + roles.get(sentence_from, unknown_role)
            + ": "
            + sentence["value"]
            + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    return conversation


class ConversationDatasetV2(Dataset):
    """ConversationDatasetV2"""

    def __init__(self, conversations, tokenizer, maxlen=512):
        """__init__"""
        super(ConversationDatasetV2, self).__init__()
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __getitem__(self, i):
        """__getitem__"""
        header = f"{conversation_lib.default_conversation.system}\n\n"
        source = self.conversations[i]
        _add_speaker_and_signal(header, source)
        source["chat_desc"] = header
        chat_desc = source['chat_desc']
        instruction = source['instruction']
        conversations = source['conversations']
        # chat_desc
        example = self.tokenizer.encode_plus(f"{chat_desc}", None, max_length=None)['input_ids']
        EOS_TOKEN = example[-1]
        example = example[:-1] # remove eos
        # instruction
        instruction = self.tokenizer.encode_plus(f"{instruction}", None, max_length=None)['input_ids']
        instruction = instruction[1:-1] # remove bos & eos
        example += instruction
        import copy
        labels = copy.deepcopy(example)
        for conversation in conversations:
            role = conversation['from']
            content = conversation['value']
            content = self.tokenizer.encode_plus(f"{content}", None, max_length=None)['input_ids']
            content = content[1:-1] # remove bos & eos
            example += content
            if role == 'gpt':
                role_labels = copy.deepcopy(content)
            else:
                # masking
                #role_labels = [env_args.IGNORE_INDEX] * len(content)
                role_labels = [-100] * len(content)
            labels += role_labels
        example.append(EOS_TOKEN)
        labels.append(EOS_TOKEN)
        ## delete bos & eos
        #example = example[1:-1]
        #labels = labels[1:-1]
        ## maxlen
        if len(example) > self.maxlen:
            example = example[:self.maxlen]
        else:
            #pad_batch
            example = example + [0] * (self.maxlen - len(example))
            #example = np.concatenate((example, np.zeros(self.maxlen - len(example), dtype=np.int)))
        if len(labels) > self.maxlen:
            labels = labels[:self.maxlen]
        else:
            labels = labels +  [-100] * (self.maxlen - len(labels))
            #labels = np.concatenate((labels, np.zeros(self.maxlen - len(labels), dtype=np.int)))
        loss_mask = [(0 if k == -100 else 1) for k in labels ]
        #loss_masks = np.concatenate((loss_masks, np.zeros(self.maxlen - len(loss_masks), dtype=np.int)))
        output = {
            "input_ids": np.array(example, dtype=np.int64),
            "labels": np.array(labels, dtype=np.int64),
            "loss_mask": np.array(loss_mask, dtype=np.int64),
        }
        return output

    def __len__(self):
        """__len__"""
        return len(self.conversations)

    @staticmethod
    def collate_fn(batch): #TODO mod by zh
        """collate_fn"""

        def padding(indice, max_length, pad_idx=0):
            """padding"""
            pad_indice = [
                item + [pad_idx] * max(0, max_length - len(item)) for item in indice
            ]
            return torch.tensor(pad_indice)

        input_ids = [data["input_ids"] for data in batch]
        labels = [data["labels"] for data in batch]
        max_length = max_seq_len
        input_ids = padding(input_ids, max_length)[:, :max_length]
        #labels = padding(labels, max_length, pad_idx=env_args.IGNORE_INDEX)[:,:max_length]
        labels = padding(labels, max_length, pad_idx=-100)[:, :max_length]
        data = {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "labels": np.array(labels, dtype=np.int64),
        }
        return data


def read_file(jsonl_file):
    """read_file"""
    conversations = []
    with jsonlines.open(jsonl_file) as reader:
        for line in reader:
            conversations.append(line)
    return conversations


def build_train_valid_test_datasets(
    data_prefix, splits_string, train_valid_test_num_samples, seq_length,
    length_per_sample, aggregated_samples_per_sequence, args
):
    """build_train_valid_test_datasets"""
    jsonl_data = data_prefix[0]
    conversations = read_file(jsonl_data)
    data_len = len(conversations)
    train_size = data_len
    train_conversations = conversations[:train_size]

    from megatron import get_tokenizer
    tokenizer = get_tokenizer()
    print('*' * 20, "tokenizer", tokenizer)
    train_dataset = ConversationDatasetV2(train_conversations,
                                            tokenizer=tokenizer,
                                            maxlen=max_seq_len)
    test_dataset = None
    valid_dataset = None
    return train_dataset, valid_dataset, test_dataset
