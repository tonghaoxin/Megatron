# !/usr/bin/env python3
import random
import copy
import numpy as np
from scipy.stats import poisson
from scipy.linalg import block_diag
from megatron.core import mpu
from megatron import print_rank_0
import torch
import torch.utils.data


class GLMPreprocessor:
    def __init__(
            self,
            eod_id,
            mask_id,
            gmask_id,
            sop_id,
            eop_id,
            max_seq_length,
            aggregated_samples_per_sequence,
            gpt_prob,
            short_seq_prob,
            single_span_prob,
            mask_ratio,
            average_block_length,
            min_gmask_ratio,
            relative_pos_encoding,
            no_2d_encoding,
            aggregate_gpt_sample,
            adaptive_multitask_encoding,
            adaptive_multitask_encoding_length,
            unified_multitask_encoding,
            rank,
            device_num,
    ):
        self.eod_id = eod_id
        self.mask_id = mask_id
        self.gmask_id = gmask_id
        self.sop_id = sop_id
        self.eop_id = eop_id
        self.max_seq_length = max_seq_length
        self.aggregated_samples_per_sequence = aggregated_samples_per_sequence
        self.gpt_prob = gpt_prob
        self.bert_prob = 1 - gpt_prob
        self.short_seq_prob = short_seq_prob
        self.single_span_prob = single_span_prob
        self.mask_ratio = mask_ratio
        self.average_block_length = average_block_length
        self.min_gmask_ratio = min_gmask_ratio
        self.block_length_distribution = [
            poisson.pmf(i, average_block_length) for i in range(1, 40)
        ]
        self.relative_pos_encoding = relative_pos_encoding
        self.no_2d_encoding = no_2d_encoding
        self.aggregate_gpt_sample = aggregate_gpt_sample
        self.adaptive_multitask_encoding = adaptive_multitask_encoding
        self.adaptive_length_distribution = 1 - np.array([
            poisson.cdf(i, adaptive_multitask_encoding_length) for i in range(1, 40)
        ])
        self.unified_multitask_encoding = unified_multitask_encoding
        self.count = 0
        self.rank = rank
        self.device_num = device_num

    def truncate_input(self, input_ids, rng):
        target_length = rng.randrange(32, len(input_ids))
        return input_ids[:target_length]

    @staticmethod
    def build_mask_matrix(separator, seq_length, memory_length=0):
        dtype = np.int64
        m = np.ones((seq_length, seq_length), dtype=dtype)
        m = np.tril(m)
        m[:, :separator] = 1
        if memory_length > 0:
            m = np.concatenate(
                (np.ones((seq_length, memory_length), dtype=dtype), m), dim=2
            )
        m = m[np.newaxis, :, :]
        return m

    @staticmethod
    def sample_spans(span_lengths, total_length, rng, offset=0):
        blank_length = total_length - sum(span_lengths)
        m = blank_length - len(span_lengths) + 1
        places = [rng.randrange(m + 1) for _ in range(len(span_lengths))]
        places.sort()
        spans = []
        for place, span_length in zip(places, span_lengths):
            start = offset + place
            end = offset + place + span_length
            spans.append((start, end))
            offset += span_length + 1
        return spans

    def make_block_data(self, input_ids, block_spans, rng, task="bert"):
        position_ids = np.ones(len(input_ids), dtype=np.int)
        for start, end in block_spans:
            position_ids[start + 1: end] = 0
        position_ids = np.cumsum(position_ids) - 1
        rng.shuffle(block_spans)
        block_spans = [(start, end) for start, end in block_spans]
        (
            target_tokens,
            target_position_ids,
            target_block_position_ids,
            targets,
        ) = ([], [], [], [])
        for start, end in block_spans:
            target_tokens.append([self.sop_id])
            span_tokens = copy.deepcopy(input_ids[start:end])
            target_tokens.append(span_tokens)
            targets.append(input_ids[start:end])
            targets.append([self.eop_id])
            target_position_id = position_ids[start:end]
            target_position_ids.append(target_position_id)
            target_position_ids.append([target_position_id[0]])
            target_block_position_ids.append(
                np.arange(1, end - start + 2, dtype=np.int)
            )
        block_spans.sort(key=lambda x: x[0])
        source_tokens, source_position_ids, local_spans = [], [], []
        last, current_length = 0, 0
        for start, end in block_spans:
            if task == "generation":
                mask_id = self.gmask_id
            else:
                mask_id = self.mask_id
            local_spans.append((current_length, current_length + start - last))
            source_tokens.append(input_ids[last:start])
            source_tokens.append([mask_id])
            source_position_ids.append(position_ids[last:start])
            source_position_ids.append([position_ids[start]])
            current_length += start - last + 1
            last = end
        if last < len(input_ids):
            local_spans.append(
                (current_length, current_length + len(input_ids) - last)
            )
            source_tokens.append(input_ids[last:])
            source_position_ids.append(position_ids[last:])
        source_length = sum(map(len, source_tokens))
        tokens = np.concatenate(source_tokens + target_tokens)
        targets = np.concatenate(source_tokens + targets)
        loss_masks = np.ones(len(tokens), dtype=np.int)
        loss_masks[:source_length] = 0
        position_ids = np.concatenate(source_position_ids + target_position_ids)
        block_position_ids = np.concatenate(
            [np.zeros(source_length, dtype=np.int)] + target_block_position_ids
        )
        position_ids = np.stack([position_ids, block_position_ids], axis=0)
        return tokens, targets, loss_masks, position_ids, source_length

    def generate_blank_data(self, input_ids, masked_lengths, rng, task="bert"):
        rng.shuffle(masked_lengths)
        block_spans = self.sample_spans(masked_lengths, len(input_ids), rng)
        if len(block_spans) < len(masked_lengths):
            return None
        data = self.make_block_data(input_ids, block_spans, rng, task=task)
        return data

    def pad_batch(self, tokens, targets, loss_masks, position_ids, max_seq_length=None):
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        if len(tokens) >= max_seq_length:
            tokens = tokens[: max_seq_length]
            targets = targets[: max_seq_length]
            loss_masks = loss_masks[: max_seq_length]
            position_ids = position_ids[:, : max_seq_length]
        else:
            tokens = np.concatenate(
                (
                    tokens,
                    np.zeros(max_seq_length - len(tokens), dtype=np.int),
                )
            )
            targets = np.concatenate(
                (
                    targets,
                    np.zeros(max_seq_length - len(targets), dtype=np.int),
                )
            )
            loss_masks = np.concatenate(
                (
                    loss_masks,
                    np.zeros(
                        max_seq_length - len(loss_masks), dtype=np.int
                    ),
                )
            )
            position_ids = np.concatenate(
                (
                    position_ids,
                    np.zeros(
                        (2, max_seq_length - position_ids.shape[1]),
                        dtype=np.int,
                    ),
                ),
                axis=1,
            )
        return tokens, targets, loss_masks, position_ids

    def _build_relative_pos_encoding(self, position_ids, division):
        position_ids, block_position_ids = position_ids[0], position_ids[1]
        seq_length = position_ids.shape[0]
        relative_pos = np.zeros((seq_length, seq_length), dtype=np.float16)
        relative_pos[:, :division] = -np.abs(position_ids[:, None] - position_ids[None, :division])
        arange = np.arange(seq_length - division)
        relative_pos[division:, division:] = np.tril(arange[None, :] - arange[:, None])
        return relative_pos

    def _pack_samples(self, sequences):
        tokens, targets, loss_masks, position_ids, division = zip(*sequences)
        tokens = np.concatenate(tokens, axis=-1)
        targets = np.concatenate(targets, axis=-1)
        loss_masks = np.concatenate(loss_masks, axis=-1)
        if self.relative_pos_encoding:
            position_ids = block_diag(*position_ids)
        else:
            position_ids = np.concatenate(position_ids, axis=-1)
        division = np.concatenate(division, axis=-1)
        if len(division) < self.aggregated_samples_per_sequence:
            division = np.append(division, [-1 for i in range(self.aggregated_samples_per_sequence - 1)], axis=-1)
        return tokens, targets, loss_masks, position_ids, division

    def get_input_data(self, input_ids, index=None):
        if index is None:
            rng = random.Random(self.count * self.device_num + self.rank)
        else:
            rng = random.Random(random.Random(index).randint(0, 2 ** 32 - 1))
        self.count += 1
        if rng.random() < self.bert_prob:
            sequences = []
            assert self.max_seq_length % self.aggregated_samples_per_sequence == 0
            assert len(input_ids) % self.aggregated_samples_per_sequence == 0
            input_length = len(input_ids) // self.aggregated_samples_per_sequence
            for i in range(self.aggregated_samples_per_sequence):
                current_input_ids = input_ids[input_length * i: input_length * (i + 1)]
                if rng.random() < self.short_seq_prob:
                    current_input_ids = self.truncate_input(current_input_ids, rng)
                single_span = rng.random() < self.single_span_prob
                if single_span:
                    masked_lengths = [
                        rng.choices(
                            range(1, len(self.block_length_distribution) + 1),
                            weights=self.block_length_distribution,
                        )[0]
                    ]
                else:
                    masked_lengths, masked_count = [], 0
                    while masked_count < int(self.mask_ratio * len(current_input_ids)):
                        block_length = rng.choices(
                            range(1, len(self.block_length_distribution) + 1),
                            weights=self.block_length_distribution,
                        )[0]
                        masked_lengths.append(block_length)
                        masked_count += block_length
                tokens, targets, loss_masks, position_ids, division = self.generate_blank_data(
                    current_input_ids, masked_lengths, rng, task="bert"
                )
                tokens, targets, loss_masks, position_ids = self.pad_batch(
                    tokens, targets, loss_masks, position_ids,
                    max_seq_length=self.max_seq_length // self.aggregated_samples_per_sequence
                )
                if self.relative_pos_encoding:
                    position_ids = self._build_relative_pos_encoding(position_ids, division)
                elif self.no_2d_encoding:
                    position_ids = position_ids[0]
                division = np.array([division], dtype=np.int)
                sequences.append((tokens, targets, loss_masks, position_ids, division))
            return *self._pack_samples(sequences), 0
        else:
            sequences = []
            if self.aggregate_gpt_sample:
                assert self.max_seq_length % self.aggregated_samples_per_sequence == 0
                assert len(input_ids) % self.aggregated_samples_per_sequence == 0
                input_length = len(input_ids) // self.aggregated_samples_per_sequence
                aggregated_samples_per_sequence = self.aggregated_samples_per_sequence
            else:
                input_length = len(input_ids)
                aggregated_samples_per_sequence = 1
            for i in range(aggregated_samples_per_sequence):
                current_input_ids = input_ids[input_length * i: input_length * (i + 1)]
                generation_length = rng.randint(
                    int(self.min_gmask_ratio * len(current_input_ids)), len(current_input_ids)
                )
                division = len(current_input_ids) - generation_length
                source_tokens, target_tokens = (
                    current_input_ids[:division],
                    current_input_ids[division:],
                )
                target_masks = np.ones(len(target_tokens), dtype=np.int)
                tokens = np.concatenate(
                    (
                        source_tokens,
                        [self.gmask_id, self.sop_id],
                        target_tokens[:-1],
                    )
                )
                targets = np.concatenate(
                    (source_tokens, [self.gmask_id], target_tokens)
                )
                loss_masks = np.concatenate(
                    (np.zeros(len(source_tokens) + 1, dtype=np.int), target_masks)
                )
                position_ids = np.arange(
                    len(source_tokens) + len(target_tokens) + 1, dtype=np.int
                )
                position_ids[len(source_tokens) + 1:] = len(source_tokens)
                block_position_ids = np.concatenate(
                    (
                        np.zeros(len(source_tokens), dtype=np.int),
                        np.arange(len(target_tokens) + 1, dtype=np.int),
                    )
                )
                position_ids = np.stack([position_ids, block_position_ids], axis=0)
                division = division + 1
                tokens, targets, loss_masks, position_ids = self.pad_batch(
                    tokens, targets, loss_masks, position_ids,
                    max_seq_length=self.max_seq_length // aggregated_samples_per_sequence
                )
                if self.relative_pos_encoding:
                    position_ids = self._build_relative_pos_encoding(position_ids, division)
                elif self.no_2d_encoding:
                    position_ids = np.arange(len(tokens), dtype=np.int)
                # attention_mask = self.build_mask_matrix(division, self.max_seq_length)
                division = np.array([division], dtype=np.int)
                sequences.append((tokens, targets, loss_masks, position_ids, division))
            return *self._pack_samples(sequences), 1

    def _get_single_multitask_data(self, text, target, max_seq_length):
        if len(text) + len(target) + 2 > max_seq_length:
            text_length = max(int(0.25 * max_seq_length), max_seq_length - len(target) - 2)
            text = text[:text_length]
        if len(text) + len(target) + 2 > max_seq_length:
            target = target[:max_seq_length - len(text) - 2]
        dtype = text.dtype
        if self.mask_id in text:
            assert self.unified_multitask_encoding
            mask_position = np.where(self.mask_id)[0][0]
            tokens = np.concatenate((text, [self.sop_id], target))
            targets = np.concatenate((text, target, [self.eop_id]))
            loss_masks = np.concatenate((np.zeros(len(text), dtype=dtype), np.ones(len(target) + 1, dtype=dtype)))
            position_ids = np.arange(len(tokens), dtype=dtype)
            position_ids[len(text):] = mask_position
            position_ids = np.stack([position_ids, position_ids])
            division = len(text)
            tokens, targets, loss_masks, position_ids = self.pad_batch(tokens, targets, loss_masks, position_ids,
                                                                       max_seq_length=max_seq_length)
            return tokens, targets, loss_masks, position_ids[0], np.array([division], dtype=dtype)
        tokens = np.concatenate((text, [self.mask_id, self.sop_id], target))
        targets = np.concatenate((text, [self.mask_id], target, [self.eop_id]))
        loss_masks = np.concatenate((np.zeros(len(text) + 1, dtype=dtype), np.ones(len(target) + 1, dtype=dtype)))
        position_ids = np.arange(len(tokens), dtype=dtype)
        position_ids[len(text) + 1:] = len(text)
        block_position_ids = np.concatenate((np.zeros(len(text), dtype=dtype), np.arange(len(target) + 2, dtype=dtype)))
        position_ids = np.stack([position_ids, block_position_ids])
        tokens, targets, loss_masks, position_ids = self.pad_batch(tokens, targets, loss_masks, position_ids,
                                                                   max_seq_length=max_seq_length)
        division = len(text) + 1
        if self.relative_pos_encoding:
            position_ids = self._build_relative_pos_encoding(position_ids, division)
        elif self.no_2d_encoding:
            position_ids = np.arange(len(tokens), dtype=dtype)
            if self.adaptive_multitask_encoding:
                rng = random.Random(random.Random(np.sum(tokens) + np.sum(targets)).randint(0, 2 ** 32 - 1))
                if len(target) < len(self.adaptive_length_distribution) \
                        and rng.random() < self.adaptive_length_distribution[len(target)]:
                    position_ids[len(text) + 1:] = len(text)
                else:
                    position_ids = np.concatenate((np.arange(len(text) + 1, dtype=dtype),
                                                   np.arange(len(text), len(text) + len(target) + 1, dtype=dtype)))
                    position_ids = np.concatenate((position_ids,
                                                   np.zeros(max_seq_length - len(position_ids), dtype=dtype)))
            elif self.unified_multitask_encoding:
                position_ids = np.concatenate((np.arange(len(text) + 1, dtype=dtype),
                                         np.arange(len(text), len(text) + len(target) + 1, dtype=dtype)))
                position_ids = np.concatenate((position_ids,
                                        np.zeros(max_seq_length - len(position_ids), dtype=dtype)))
        # attention_mask = self.build_mask_matrix(len(text) + 1, max_seq_length)
        return tokens, targets, loss_masks, position_ids, np.array([division], dtype=dtype)

    def get_multitask_data(self, texts, targets):
        if self.aggregated_samples_per_sequence > 1:
            assert self.max_seq_length % self.aggregated_samples_per_sequence == 0
            sequences = []
            for text, target in zip(texts, targets):
                data = self._get_single_multitask_data(text, target, self.max_seq_length // self.aggregated_samples_per_sequence)
                sequences.append(data)
            return self._pack_samples(sequences)
        else:
            return self._get_single_multitask_data(texts[0], targets[0], self.max_seq_length)

    def get_greedily_aggregated_multitask_data(self, texts, targets):
        sequences, length = [], 0
        for idx, (text, target) in enumerate(zip(texts, targets)):
            cur_length = self.max_seq_length - length if idx + 1 == len(texts) else len(text) + len(target) + 2
            tokens, targets, loss_masks, position_ids, division = \
                self._get_single_multitask_data(text, target, max_seq_length=cur_length)
            division  = np.array([division, [cur_length]], dtype=np.long)
            sequences.append((tokens, targets, loss_masks, position_ids, division))
            length += cur_length
        return self._pack_samples(sequences)


def debug_block_data(data):
    tokens, targets, loss_masks, position_ids, attention_mask = data
    # block_position_ids = position_ids[1]
    # position_ids_ = position_ids[0]
    sep = int(attention_mask[0, 0].sum())
    text, last_segment = "", []
    for i, token_id in enumerate(tokens[:sep].tolist()):
        if token_id == 10001 or token_id == 10002:
            if last_segment:
                text += " ".join(last_segment)
                last_segment = []
            text += f" [{position_ids[i]}, mask]"
        else:
            last_segment.append(str(token_id))
    if last_segment:
        text += " ".join(last_segment)
    print(text)
    last_index = None
    for i in range(sep, tokens.shape[0]):
        if tokens[i] == 10003:
            if last_index is not None:
                print(
                    tokens[last_index:i].tolist(),
                    "|",
                    targets[last_index:i].tolist(),
                    position_ids[last_index:i].tolist(),
                    # position_ids_[last_index:i].tolist(),
                    # block_position_ids[last_index:i].tolist(),
                )
            last_index = i
    if last_index is not None:
        end_index = last_index
        for i in range(last_index, tokens.shape[0]):
            if loss_masks[i] != 0:
                end_index = i
        print(
            tokens[last_index:end_index + 1].tolist(),
            "|",
            targets[last_index:end_index + 1].tolist(),
            position_ids[last_index:end_index + 1].tolist(),
            # position_ids_[last_index:end_index + 1].tolist(),
            # block_position_ids[last_index:end_index + 1].tolist(),
        )


def rindex(lst, val, start=None):
    if start is None:
        start = len(lst) - 1
    for i in range(start, -1, -1):
        if lst[i] == val:
            return i
    return -1


def index_in_list(lst, val, start=None):
    if start is None:
        start = 0
    for i in range(start, len(lst)):
        if lst[i] == val:
            return i
    return -1


class ConstructBlockStrategy:
    def __init__(self, args, tokenizer, max_seq_length, bert_prob=1.0, gap_sentence_prob=0.0, gpt_infill_prob=0.5,
                 gpt_min_ratio=0.5, bert_ratio=0.15, gap_sentence_ratio=0.15, average_block_length=3,
                 max_block_length=40, block_mask_prob=0.0, context_mask_ratio=0.0, context_mask_range=3,
                 short_seq_prob=0.0, single_span_prob=0.0, block_position_encoding=True, encoder_decoder=False,
                 shuffle_blocks=True, sentinel_token=False, task_mask=False, random_position=False, masked_lm=False):
        self.eod_token = tokenizer.get_command('eos').Id
        self.tokenizer = tokenizer
        self.count = 0
        self.max_seq_length = max_seq_length
        self.rank = mpu.get_data_parallel_rank()
        self.world_size = mpu.get_data_parallel_world_size()
        # self.rank = 0
        # self.world_size = 1
        assert 0.0 <= bert_prob <= 1.0
        self.bert_prob = bert_prob
        self.gap_sentence_prob = gap_sentence_prob
        self.gpt_prob = 1 - bert_prob - gap_sentence_prob
        assert self.gpt_prob >= -1e-10
        self.infill_prob = gpt_infill_prob
        self.gpt_min_ratio = gpt_min_ratio
        self.bert_ratio = bert_ratio
        self.gap_sentence_ratio = gap_sentence_ratio
        self.block_length_distribution = [poisson.pmf(i, average_block_length) for i in range(1, max_block_length)]
        self.block_mask_prob = block_mask_prob
        self.context_mask_ratio = context_mask_ratio
        self.context_mask_range = context_mask_range
        self.short_seq_prob = short_seq_prob
        self.single_span_prob = single_span_prob
        self.block_position_encoding = block_position_encoding
        self.encoder_decoder = encoder_decoder
        self.shuffle_blocks = shuffle_blocks
        self.sentinel_token = sentinel_token
        self.generation_mask = 'gMASK' if task_mask else 'MASK'
        self.generation_mask = self.tokenizer.get_command(self.generation_mask).Id
        self.gap_sentence_mask = 'sMASK' if task_mask else 'MASK'
        self.gap_sentence_mask = self.tokenizer.get_command(self.gap_sentence_mask).Id
        self.random_position = random_position
        self.masked_lm = masked_lm
        print_rank_0(
            f"BERT prob {self.bert_prob}, gap sent prob {self.gap_sentence_prob}, GPT prob {self.gpt_prob}, infill prob {self.infill_prob}")
        print_rank_0(
            f"generation min ratio {self.gpt_min_ratio}, block ratio {self.bert_ratio}, gap sent ratio {self.gap_sentence_ratio}")
        print_rank_0(f"block length distribution {self.block_length_distribution}")
        print_rank_0(f"block mask prob {self.block_mask_prob}, context mask ratio {self.context_mask_ratio}")

    def contains_sentence_end(self, tok):
        tok = self.tokenizer.IdToToken(tok)
        if '.' in tok:
            return True
        if '?' in tok:
            return True
        if '!' in tok:
            return True
        if ';' in tok:
            return True
        if ':' in tok:
            return True
        if '。' in tok:
            return True
        if '？' in tok:
            return True
        if '！' in tok:
            return True
        if '；' in tok:
            return True
        if '…' in tok:
            return True
        if '\n' in tok:
            return True
        return False

    @staticmethod
    def sample_spans(span_lengths, total_length, rng, offset=0):
        blank_length = total_length - sum(span_lengths)
        m = blank_length - len(span_lengths) + 1
        places = [rng.randrange(m + 1) for _ in range(len(span_lengths))]
        places.sort()
        spans = []
        for place, span_length in zip(places, span_lengths):
            start = offset + place
            end = offset + place + span_length
            spans.append((start, end))
            offset += span_length + 1
        return spans

    def sample_span_in_document(self, tokens, masked_lengths, rng):
        rng.shuffle(masked_lengths)
        mask_spans = []
        mask_index = 0
        indices = [-1] + np.where(tokens == self.eod_token)[0].tolist()
        last_index = len(tokens)
        documents = []
        for index in reversed(indices):
            start_index = index
            if start_index + 1 < len(tokens) and tokens[start_index + 1] == self.tokenizer.get_command('ENC').Id:
                start_index += 1
            length = last_index - start_index - 1
            if last_index == len(tokens) and length > 0:
                length -= 1
            documents.append((start_index + 1, length))
            last_index = index
        documents.sort(key=lambda x: x[1])
        for i, (offset, length) in enumerate(documents):
            if i == len(documents) - 1:
                current_masked_length, current_count = 0, 0
                while mask_index + current_count < len(masked_lengths) and masked_lengths[
                    mask_index + current_count] + current_masked_length + current_count <= length:
                    current_masked_length += masked_lengths[mask_index + current_count]
                    current_count += 1
                if current_count > 0:
                    spans = self.sample_spans(masked_lengths[mask_index: mask_index + current_count], length, rng,
                                              offset=offset)
                    mask_spans += spans
                if mask_index + current_count < len(masked_lengths) - 1:
                    print(length, masked_lengths[mask_index:], masked_lengths[:mask_index], indices)
            else:
                current_masked_total = int(length * self.bert_ratio)
                current_masked_length, current_count = 0, 0
                while mask_index + current_count < len(masked_lengths) and masked_lengths[
                    mask_index + current_count] + current_masked_length <= current_masked_total:
                    current_masked_length += masked_lengths[mask_index + current_count]
                    current_count += 1
                if current_count > 0:
                    spans = self.sample_spans(masked_lengths[mask_index:mask_index + current_count], length,
                                              rng, offset=offset)
                    mask_spans += spans
                    mask_index += current_count
        return mask_spans

    def make_masked_data(self, tokens, loss_masks, attention_mask, block_spans, rng, task='bert'):
        position_ids = np.arange(len(tokens), dtype=np.long)
        targets = copy.deepcopy(tokens)
        mask_id = self.tokenizer.get_command('MASK').Id
        mlm_masks = np.zeros(len(tokens), dtype=np.long)
        for start, end in block_spans:
            for idx in range(start, end):
                tokens[idx] = mask_id
            mlm_masks[start: end] = 1
        loss_masks = loss_masks * mlm_masks
        return tokens, targets, loss_masks, position_ids

    def make_block_data(self, tokens, loss_masks, attention_mask, block_spans, rng, task='bert'):
        text_length = len(tokens)
        position_ids = np.ones(len(tokens), dtype=np.long)
        for start, end in block_spans:
            position_ids[start + 1: end] = 0
        position_ids = np.cumsum(position_ids) - 1
        if self.random_position and position_ids[-1] < self.max_seq_length - 1:
            position_bias = self.max_seq_length - position_ids[-1]
            position_bias = rng.randrange(0, position_bias)
            position_ids = position_ids + position_bias
        if self.encoder_decoder or not self.shuffle_blocks:
            block_spans.sort(key=lambda x: x[0])
        else:
            rng.shuffle(block_spans)
        if self.sentinel_token:
            block_spans = [(start, end, idx) for idx, (start, end) in enumerate(block_spans)]
        else:
            block_spans = [(start, end, 0) for start, end in block_spans]
        target_tokens, target_position_ids, target_block_position_ids, targets = [], [], [], []
        for start, end, idx in block_spans:
            sop_token = 'sop' if idx == 0 else f"sop{idx}"
            target_tokens.append([self.tokenizer.get_command(sop_token).Id])
            span_tokens = copy.deepcopy(tokens[start: end])
            if self.block_mask_prob > 0.0 and task == 'bert':
                for sub_idx in range(len(span_tokens)):
                    if random.random() < self.block_mask_prob:
                        span_tokens[sub_idx] = self.tokenizer.get_command('dBLOCK').Id
            target_tokens.append(span_tokens)
            targets.append(tokens[start: end])
            targets.append([self.tokenizer.get_command('eop').Id])
            if not self.sentinel_token:
                target_position_id = position_ids[start: end]
                target_position_ids.append(target_position_id)
                target_position_ids.append([target_position_id[0]])
            else:
                target_position_ids.append([self.max_seq_length] * (end - start + 1))
            if self.block_position_encoding:
                target_block_position_ids.append(np.arange(1, end - start + 2, dtype=np.long))
            else:
                target_block_position_ids.append([1] * (end - start + 1))
        block_spans.sort(key=lambda x: x[0])
        source_tokens, source_position_ids, local_spans = [], [], []
        last, current_length = 0, 0
        for start, end, idx in block_spans:
            if task == 'generation':
                mask_id = self.generation_mask
            elif task == 'gap_sentence':
                mask_id = self.gap_sentence_mask
            else:
                mask_token = 'MASK' if idx == 0 else f'MASK{idx}'
                mask_id = self.tokenizer.get_command(mask_token).Id
            local_spans.append((current_length, current_length + start - last))
            source_tokens.append(tokens[last: start])
            source_tokens.append([mask_id])
            source_position_ids.append(position_ids[last: start])
            source_position_ids.append([position_ids[start]])
            current_length += start - last + 1
            last = end
        if last < len(tokens):
            local_spans.append((current_length, current_length + len(tokens) - last))
            source_tokens.append(tokens[last:])
            source_position_ids.append(position_ids[last:])
        source_length = sum(map(len, source_tokens))
        if attention_mask is not None:
            assert source_length == attention_mask
        if target_tokens and self.eod_token in np.concatenate(target_tokens).tolist():
            print("Found EOS in target", self.tokenizer.DecodeIds(tokens))
            raise RuntimeError
        if self.encoder_decoder:
            target_tokens = target_tokens + [self.tokenizer.get_command('eop').Id]
            loss_masks = np.ones(len(target_tokens), dtype=np.long)
            return source_tokens, target_tokens, loss_masks
        else:
            tokens = np.concatenate(source_tokens + target_tokens)
            if task == 'bert' and self.context_mask_ratio > 0:
                mask_candidates = set()
                for start, end in local_spans:
                    if start != 0:
                        local_end = min(end, start + self.context_mask_range)
                        mask_candidates.update(range(start, local_end))
                    if end != 0:
                        local_start = max(start, end - self.context_mask_range)
                        mask_candidates.update(range(local_start, end))
                mask_pos = rng.sample(mask_candidates, int(self.context_mask_ratio * text_length))
                for pos in mask_pos:
                    tokens[pos] = self.tokenizer.get_command('dBLOCK').Id
            targets = np.concatenate(source_tokens + targets)
            loss_masks = np.ones(len(tokens), dtype=np.long)
            loss_masks[:source_length] = 0
            position_ids = np.concatenate(source_position_ids + target_position_ids)
            block_position_ids = np.concatenate(
                [np.zeros(source_length, dtype=np.long)] + target_block_position_ids)
            position_ids = np.stack([position_ids, block_position_ids], axis=0)
            if attention_mask is not None:
                return tokens, targets, loss_masks, position_ids
            else:
                return tokens, targets, loss_masks, position_ids, source_length

    def generate_blank_data(self, sample, masked_lengths, attention_mask, rng, task='bert'):
        rng.shuffle(masked_lengths)
        tokens, loss_masks = sample['text'], sample['loss_mask']
        assert tokens[0] == self.tokenizer.get_command('ENC').Id
        block_spans = self.sample_span_in_document(tokens, masked_lengths, rng)
        if len(block_spans) < len(masked_lengths):
            return None
        if self.masked_lm:
            data = self.make_masked_data(tokens, loss_masks, attention_mask, block_spans, rng)
        else:
            data = self.make_block_data(tokens, loss_masks, attention_mask, block_spans, rng, task=task)
        return data

    def split_samples(self, samples, rng):
        target_length = rng.randrange(32, self.max_seq_length - 1)
        num_splits = (self.max_seq_length - 1) // target_length
        new_samples = []
        cls_id = self.tokenizer.get_command('ENC').Id
        eos_id = self.tokenizer.get_command('eos').Id
        for sample in samples:
            tokens, loss_masks = sample['text'][1:], sample['loss_mask'][1:]
            for _ in range(num_splits):
                if target_length >= len(tokens):
                    new_tokens, new_loss_masks = tokens, loss_masks
                else:
                    random_start = rng.randrange(0, len(tokens) - target_length)
                    while random_start > 0 and (tokens[random_start] == eos_id or not (
                            self.contains_sentence_end(tokens[random_start - 1]) or tokens[
                        random_start - 1] == eos_id)):
                        random_start -= 1
                    random_end = random_start + target_length
                    while random_end > random_start and not (
                            self.contains_sentence_end(tokens[random_end - 1]) or tokens[random_end - 1] == eos_id):
                        random_end -= 1
                    if random_end - random_start < target_length // 2:
                        random_end = random_start + target_length
                    new_tokens, new_loss_masks = tokens[random_start: random_end], loss_masks[random_start: random_end]
                new_tokens = np.concatenate(([cls_id], new_tokens))
                new_loss_masks = np.concatenate(([0], new_loss_masks))
                new_samples.append({'text': new_tokens, 'loss_mask': new_loss_masks})
        return new_samples

    def construct_blocks(self, samples):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
        else:
            worker_id, num_workers = 0, 1
        rng = random.Random((self.count * num_workers + worker_id) * self.world_size + self.rank)
        self.count += 1
        token_batch, target_batch, loss_mask_batch, position_id_batch = [], [], [], []
        source_batch, target_batch = [], []
        if rng.random() < self.short_seq_prob:
            samples = self.split_samples(samples, rng)
        rand = rng.random()
        single_span = rand < self.single_span_prob
        rand = 0.0 if single_span else rng.random()
        attention_mask = []
        if rand < self.bert_prob:
            mode = 'bert'
            for sample in samples:
                if single_span:
                    masked_lengths = [rng.choices(range(1, len(self.block_length_distribution) + 1),
                                                  weights=self.block_length_distribution)[0]]
                    masked_count = masked_lengths[0]
                else:
                    masked_lengths, masked_count = [], 0
                    while masked_count < int(self.bert_ratio * len(sample['text'])):
                        block_length = rng.choices(range(1, len(self.block_length_distribution) + 1),
                                                   weights=self.block_length_distribution)[0]
                        masked_lengths.append(block_length)
                        masked_count += block_length
                if self.masked_lm:
                    sep = len(sample['text'])
                else:
                    sep = len(sample['text']) - masked_count + len(masked_lengths)
                data = self.generate_blank_data(sample, masked_lengths, sep, rng, task='bert')
                if data is not None:
                    if self.encoder_decoder:
                        source_tokens, target_tokens, loss_masks = data
                        source_batch.append(source_tokens)
                        target_batch.append(target_tokens)
                        loss_mask_batch.append(loss_masks)
                    else:
                        tokens, targets, loss_masks, position_ids = data
                        token_batch.append(tokens)
                        target_batch.append(targets)
                        loss_mask_batch.append(loss_masks)
                        position_id_batch.append(position_ids)
                    attention_mask.append(sep)

        elif rand < self.bert_prob + self.gap_sentence_prob:
            mode = 'sentence'
            for sample in samples:
                tokens, loss_masks = sample['text'], sample['loss_mask']
                sentence_spans = []
                last_index = 1 if tokens[0] == self.tokenizer.get_command('ENC').Id else 0
                for i in range(len(tokens)):
                    if self.contains_sentence_end(tokens[i]):
                        if last_index < i + 1:
                            sentence_spans.append((last_index, i + 1))
                        last_index = i + 1
                    elif tokens[i] == self.tokenizer.get_command('eos').Id:
                        last_index = i + 1
                if last_index < len(tokens):
                    sentence_spans.append((last_index, len(tokens)))
                if not sentence_spans and torch.distributed.get_rank() == 0:
                    try:
                        print(self.tokenizer.DecodeIds(tokens[1:]))
                    except IndexError:
                        print(tokens[1:])
                rng.shuffle(sentence_spans)
                block_spans, block_length = [], 0
                for start, end in sentence_spans:
                    block_spans.append((start, end))
                    block_length += end - start
                    if block_length >= int(self.gap_sentence_ratio * len(tokens)):
                        break
                data = self.make_block_data(tokens, loss_masks, None, block_spans, rng, task='gap_sentence')
                tokens, targets, loss_masks, position_ids, sep = data
                token_batch.append(tokens)
                target_batch.append(targets)
                loss_mask_batch.append(loss_masks)
                position_id_batch.append(position_ids)
                attention_mask.append(sep)
        else:
            # start_indices = [index_in_list(sample['loss_mask'], 1) for sample in samples]
            # end_indices = [rindex(sample['loss_mask'], 1) for sample in samples]
            # start_index, end_index = max(start_indices), min(end_indices) - self.min_generation_length
            # if end_index < start_index + 1:
            #     end_index = start_index + 1
            # division = rng.randrange(start_index, end_index)
            mode = 'gpt'
            max_generation_length = rng.randint(int(self.gpt_min_ratio * min(map(lambda x: len(x['text']), samples))),
                                                max(map(lambda x: len(x['text']), samples)) - 2)
            for sample in samples:
                generation_length = min(max_generation_length, len(sample['text']) - 2)
                attention_mask.append(len(sample['text']) - generation_length + 1)
                multiple_doc = index_in_list(sample['text'], self.tokenizer.get_command('eos').Id) not in [-1, len(
                    sample['text']) - 1]
                if multiple_doc or rng.random() < self.infill_prob:
                    division = len(sample['text']) - generation_length
                    tokens, loss_masks = sample['text'], sample['loss_mask']
                    source_tokens, target_tokens = tokens[:division], tokens[division:]
                    target_masks = loss_masks[division:]
                    tokens = np.concatenate((
                        source_tokens, [self.generation_mask, self.tokenizer.get_command('sop').Id],
                        target_tokens[:-1]))
                    targets = np.concatenate((source_tokens, [self.generation_mask], target_tokens))
                    loss_masks = np.concatenate((np.zeros(len(source_tokens) + 1, dtype=np.long), target_masks))
                    token_batch.append(tokens)
                    target_batch.append(targets)
                    loss_mask_batch.append(loss_masks)
                    position_ids = np.arange(len(source_tokens) + len(target_tokens) + 1, dtype=np.long)
                    position_ids[len(source_tokens) + 1:] = len(source_tokens)
                    if self.block_position_encoding:
                        block_position_ids = np.concatenate(
                            (np.zeros(len(source_tokens), dtype=np.long),
                             np.arange(len(target_tokens) + 1, dtype=np.long)))
                    else:
                        block_position_ids = np.concatenate((np.zeros(len(source_tokens) + 1, dtype=np.long),
                                                             np.ones(len(target_tokens) + 1, dtype=np.long)))
                    position_id_batch.append(np.stack([position_ids, block_position_ids], axis=0))
                else:
                    tokens, targets, loss_masks, position_ids = self.generate_blank_data(sample, [generation_length],
                                                                                         attention_mask[-1], rng,
                                                                                         task='generation')
                    token_batch.append(tokens)
                    target_batch.append(targets)
                    loss_mask_batch.append(loss_masks)
                    position_id_batch.append(position_ids)
                    if tokens is None:
                        print(sample, generation_length, multiple_doc)
        if self.encoder_decoder:
            return {
                'text': torch.tensor(source_batch, dtype=torch.long),
                'target': torch.tensor(target_batch, dtype=torch.long),
                'loss_mask': torch.tensor(loss_mask_batch, dtype=torch.long)}
        else:
            token_batch, target_batch, loss_mask_batch, position_id_batch = self.pad_batch(token_batch, target_batch,
                                                                                           loss_mask_batch,
                                                                                           position_id_batch)
            return {'text': torch.tensor(np.array(token_batch), dtype=torch.long),
                    'target': torch.tensor(np.array(target_batch), dtype=torch.long),
                    'loss_mask': torch.tensor(np.array(loss_mask_batch), dtype=torch.long),
                    'position_id': torch.tensor(np.array(position_id_batch), dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                    'mode': mode}

    @staticmethod
    def pad_batch(token_batch, target_batch, loss_mask_batch, position_id_batch):
        seq_lengths = list(map(len, token_batch))
        if seq_lengths.count(seq_lengths[0]) != len(seq_lengths):
            max_length = max(seq_lengths)
            token_batch = [np.concatenate((tokens, np.zeros(max_length - len(tokens), dtype=np.long))) for tokens in
                           token_batch]
            target_batch = [np.concatenate((targets, np.zeros(max_length - len(targets), dtype=np.long))) for
                            targets in
                            target_batch]
            loss_mask_batch = [np.concatenate((loss_masks, np.zeros(max_length - len(loss_masks), dtype=np.long)))
                               for loss_masks in loss_mask_batch]
            position_id_batch = [
                np.concatenate((position_ids, np.zeros((2, max_length - position_ids.shape[1]), dtype=np.long)),
                               axis=1) for position_ids in position_id_batch]
        return token_batch, target_batch, loss_mask_batch, position_id_batch


if __name__ == "__main__":
    aggregated_samples_per_sequence = 4
    max_seq_length = 2304
    single_length = max_seq_length // aggregated_samples_per_sequence
    collator = GLMPreprocessor(
        eod_id=10000,
        mask_id=10001,
        gmask_id=10002,
        sop_id=10003,
        eop_id=10004,
        max_seq_length=max_seq_length,
        aggregated_samples_per_sequence=aggregated_samples_per_sequence,
        gpt_prob=0.5,
        short_seq_prob=0.02,
        single_span_prob=0.02,
        mask_ratio=0.15,
        average_block_length=3,
        min_gmask_ratio=0.2,
        relative_pos_encoding=False,
        no_2d_encoding=True,
        rank=1,
        device_num=2,
        aggregate_gpt_sample=False
    )
    input_ids = np.arange(2048)
    for _ in range(10):
        (
            tokens_,
            targets_,
            loss_masks_,
            position_ids_,
            attention_mask_,
            task_type
        ) = collator.get_input_data(input_ids)
        if len(attention_mask_) > 1:
            for i in range(aggregated_samples_per_sequence):
                debug_block_data(
                    (tokens_[i * single_length: (i + 1) * single_length],
                     targets_[i * single_length: (i + 1) * single_length],
                     loss_masks_[i * single_length: (i + 1) * single_length],
                     position_ids_[i * single_length: (i + 1) * single_length], collator.build_mask_matrix(
                        attention_mask_[i], single_length))
                )
        else:
            debug_block_data(
                (tokens_, targets_, loss_masks_, position_ids_,
                 collator.build_mask_matrix(attention_mask_[0], max_seq_length)))
        print()
    texts, targets = [np.arange(256), np.arange(256, 512), np.arange(512, 768), np.arange(768, 1024)], [
        np.arange(1024, 1024 + 64), np.arange(1024 + 64, 1024 + 128), np.arange(1024 + 128, 1024 + 192),
        np.arange(1024 + 192, 1024 + 256)]
    tokens_, targets_, loss_masks_, position_ids_, attention_mask_ = collator.get_multitask_data(texts, targets)
    for i in range(aggregated_samples_per_sequence):
        debug_block_data((tokens_[i * single_length: (i + 1) * single_length],
                          targets_[i * single_length: (i + 1) * single_length],
                          loss_masks_[i * single_length: (i + 1) * single_length],
                          position_ids_[i * single_length: (i + 1) * single_length],
                          collator.build_mask_matrix(attention_mask_[i], single_length)))
        print()
        breakpoint()
    text, target = np.arange(1024), np.arange(1024, 1024 + 64)
    tokens_, targets_, loss_masks_, position_ids_, attention_mask_ = collator.get_multitask_data(text, target)
    debug_block_data((tokens_, targets_, loss_masks_, position_ids_, attention_mask_))
    print()
    text, target = np.arange(256), np.arange(1024, 1024 + 1024)
    tokens_, targets_, loss_masks_, position_ids_, attention_mask_ = collator.get_multitask_data(text, target)
    debug_block_data((tokens_, targets_, loss_masks_, position_ids_, attention_mask_))
