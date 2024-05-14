# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""

import re
import warnings
import torch

from abc import ABC
from abc import abstractmethod

from .bert_tokenization import FullTokenizer as FullBertTokenizer
from .gpt2_tokenization import GPT2Tokenizer
from .tokenization_chatglm import ChatGLMTokenizer
from .chinese_sp_tokenizer import GLMChineseTokenizer
from .chatglm2_tokenization import ChatGLM2Tokenizer
from .tokenization_llama import LlamaTokenizer
from .tokenization_qwen import QWenTokenizer
from .tokenization_hf import AutoTokenizerFromHF


def build_tokenizer(args):
    """Initialize tokenizer."""
    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.tokenizer_type),
              flush=True)

    # Select and instantiate the tokenizer.
    if args.tokenizer_type == 'BertWordPieceLowerCase':
        assert args.vocab_file is not None
        tokenizer = _BertWordPieceTokenizer(vocab_file=args.vocab_file,
                                            lower_case=True,
                                            vocab_extra_ids=args.vocab_extra_ids)
    elif args.tokenizer_type == 'BertWordPieceCase':
        assert args.vocab_file is not None
        tokenizer = _BertWordPieceTokenizer(vocab_file=args.vocab_file,
                                            lower_case=False,
                                            vocab_extra_ids=args.vocab_extra_ids)
    elif args.tokenizer_type == 'GPT2BPETokenizer':
        assert args.vocab_file is not None
        assert args.merge_file is not None
        tokenizer = _GPT2BPETokenizer(args.vocab_file, args.merge_file)
    elif args.tokenizer_type == 'AQUILABPETokenizer':
        assert args.hf_tokenizer_path is not None
        tokenizer = _AQUILABPETokenizer(args.hf_tokenizer_path)
    elif args.tokenizer_type == 'SentencePieceTokenizer':
        assert args.tokenizer_model is not None
        tokenizer = _SentencePieceTokenizer(args.tokenizer_model, vocab_extra_ids=args.vocab_extra_ids)
    elif args.tokenizer_type == 'GPTSentencePieceTokenizer':
        assert args.tokenizer_model is not None
        tokenizer = _GPTSentencePieceTokenizer(args.tokenizer_model)
    elif args.tokenizer_type == 'LLaMASentencePieceTokenizer':
        assert args.tokenizer_model is not None
        tokenizer = _LLaMASentencePieceTokenizer(args.tokenizer_model)
    elif args.tokenizer_type == 'GalacticaHFTokenizer':
        assert args.hf_tokenizer_path is not None
        tokenizer = _GalacticaHFTokenizer(args.hf_tokenizer_path)
    elif args.tokenizer_type == "IceTokenizer":
        from .ice_tokenizer import _IceTokenizer
        tokenizer = _IceTokenizer()
    elif args.tokenizer_type == "ChatGLMTokenizer":
        assert args.tokenizer_model is not None

        # prevent transformers from logging info and warnings on each rank
        import transformers
        import logging
        if args.rank == 0:
            transformers.utils.logging.set_verbosity(logging.INFO)
        else:
            # shut the warnings on replicas
            transformers.utils.logging.set_verbosity(logging.ERROR)

        tokenizer = ChatGLMTokenizer(vocab_file=args.tokenizer_model, num_image_tokens=0)
    elif args.tokenizer_type == "GLMChineseTokenizer":
        assert args.tokenizer_model is not None
        tokenizer = GLMChineseTokenizer(tokenizer_model=args.tokenizer_model,
                                        add_block_symbols=args.block_lm,
                                        add_task_mask=args.task_mask,
                                        add_decoder_mask=args.block_mask_prob > 0.0 or args.context_mask_ratio > 0.0,
                                        fix_command_token=args.fix_command_token)
    elif args.tokenizer_type == 'OptTokenizer':
        assert args.hf_tokenizer_path is not None or (args.vocab_file is not None and args.merge_file is not None)
        tokenizer = _OptTokenizer(args.hf_tokenizer_path, args.vocab_file, args.merge_file)
    elif args.tokenizer_type == "BloomHFTokenizer":
        assert args.hf_tokenizer_path is not None
        tokenizer = _BloomHFTokenizer(args.hf_tokenizer_path)
    elif args.tokenizer_type == 'NullTokenizer':
        assert args.vocab_size is not None
        tokenizer = _NullTokenizer(args.vocab_size)
    elif args.tokenizer_type == "ChatGLM2Tokenizer":
        assert args.tokenizer_model is not None

        # prevent transformers from logging info and warnings on each rank
        import transformers
        import logging
        if args.rank == 0:
            transformers.utils.logging.set_verbosity(logging.INFO)
        else:
            # shut the warnings on replicas
            transformers.utils.logging.set_verbosity(logging.ERROR)

        tokenizer = ChatGLM2Tokenizer(vocab_file=args.tokenizer_model)
    elif args.tokenizer_type == "LLaMASFTTokenizerHF":
        assert args.tokenizer_model is not None
        assert args.seq_length is not None
        seq_length = args.seq_length
        tokenizer = LlamaTokenizer(vocab_file=args.tokenizer_model, model_max_length=seq_length,
                                   padding_side="right", pad_token="[PAD]")
    elif args.tokenizer_type == "HFTokenizer":
        assert args.hf_tokenizer_path is not None
        assert args.seq_length is not None
        seq_length = args.seq_length
        tokenizer = AutoTokenizerFromHF(name_or_path=args.hf_tokenizer_path,
                                        use_fast_tokenizer=args.use_fast_tokenizer,
                                        padding_side=args.padding_side,
                                        model_max_length=args.seq_length,
                                        split_special_tokens=args.split_special_tokens)
    elif args.tokenizer_type == 'QwenTokenizer':
        assert args.tokenizer_model is not None
        assert args.seq_length is not None
        tokenizer = QWenTokenizer(vocab_file=args.tokenizer_model, model_max_length=args.seq_length,
                                  padding_side='right', use_fast=False)
        tokenizer.pad_token_id = tokenizer.pad_id
        tokenizer.eos_token_id = tokenizer.eod_id
    elif args.tokenizer_type == 'BaichuanTokenizer':
        assert args.tokenizer_model is not None
        assert args.seq_length is not None
        from .tokenization_baichuan import BaichuanTokenizer
        tokenizer = BaichuanTokenizer(vocab_file=args.tokenizer_model, model_max_length=args.seq_length,
                                      padding_side='right', use_fast=False, pad_token="[PAD]")
    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(args.tokenizer_type))

    # Add vocab size.
    args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size,
                                                      args)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args):
    if hasattr(args, "pad_vocab_size_to") and args.pad_vocab_size_to is not None:
        """Apply the requested rules to change the size of the vocabulary"""
        if args.pad_vocab_size_to  < orig_vocab_size:
            raise ValueError(
                f"You asked to pad the vocabulary to {args.pad_vocab_size_to} when the initial vocabulary size is "
                f"{orig_vocab_size}. You can only pad to a higher value."
            )

        if args.make_vocab_size_divisible_by is not None and (args.pad_vocab_size_to % args.make_vocab_size_divisible_by) != 0:
            raise ValueError(f"{args.pad_vocab_size_to} is not divisible by {args.make_vocab_size_divisible_by}")

        after = args.pad_vocab_size_to
    else:
        """Pad vocab size so it is divisible by model parallel size and still having GPU friendly size."""
        after = orig_vocab_size
        multiple = args.make_vocab_size_divisible_by * \
            args.tensor_model_parallel_size
        while (after % multiple) != 0:
            after += 1
    if args.rank == 0:
        print(' > padded vocab (size: {}) with {} dummy tokens '
              '(new size: {})'.format(
            orig_vocab_size, after - orig_vocab_size, after), flush=True)
    return after


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError('detokenizer is not implemented for {} '
                                  'tokenizer'.format(self.name))

    @property
    def cls(self):
        raise NotImplementedError('CLS is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def sep(self):
        raise NotImplementedError('SEP is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def pad(self):
        raise NotImplementedError('PAD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def eod(self):
        raise NotImplementedError('EOD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def mask(self):
        raise NotImplementedError('MASK is not provided for {} '
                                  'tokenizer'.format(self.name))


class _BertWordPieceTokenizer(AbstractTokenizer):
    """Original BERT wordpiece tokenizer."""

    def __init__(self, vocab_file, lower_case=True, vocab_extra_ids=0):
        if lower_case:
            name = 'BERT Lower Case'
        else:
            name = 'BERT Upper Case'
        super().__init__(name)
        self.tokenizer = FullBertTokenizer(vocab_file, do_lower_case=lower_case)
        self.cls_id = self.tokenizer.vocab['[CLS]']
        self.sep_id = self.tokenizer.vocab['[SEP]']
        self.pad_id = self.tokenizer.vocab['[PAD]']
        self.mask_id = self.tokenizer.vocab['[MASK]']
        self._additional_special_tokens = []

        # (dsachan) Add BOS and EOS tokens
        SPECIAL_TOKENS = {'eos_token': '[EOS]',
                          'bos_token': '[BOS]'}
        self._bos_token = '[BOS]'
        self.add_token(self._bos_token)
        self._bos_token_id = self.vocab.get(self._bos_token)

        self._eos_token = '[EOS]'
        self.add_token(self._eos_token)
        self._eos_token_id = self.vocab.get(self._eos_token)

        # (dsachan) Add additional special tokens
        # These can be used as sentinel tokens in T5 model inputs
        additional_special_tokens = []
        additional_special_tokens.extend(
            ["<extra_id_{}>".format(i) for i in range(vocab_extra_ids)])
        self.add_additional_special_tokens(additional_special_tokens)

    def add_token(self, token):
        if token not in self.vocab:
            self.inv_vocab[self.vocab_size] = token
            # self.vocab_size comes from len(vocab)
            # and it will increase as we add elements
            self.vocab[token] = self.vocab_size

    def add_additional_special_tokens(self, tokens_list):
        setattr(self, "additional_special_tokens", tokens_list)
        for value in tokens_list:
            self.add_token(value)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size()

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def inv_vocab(self):
        return self.tokenizer.inv_vocab

    def tokenize(self, text):
        text_tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(text_tokens)

    def decode(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return self.tokenizer.convert_tokens_to_string(tokens)

    def decode_token_ids(self, token_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        exclude_list = ['[PAD]', '[CLS]']
        non_pads = [t for t in tokens if t not in exclude_list]

        result = ""
        for s in non_pads:
            if s.startswith("##"):
                result += s[2:]
            else:
                result += " " + s

        return result

    @property
    def cls(self):
        return self.cls_id

    @property
    def sep(self):
        return self.sep_id

    @property
    def pad(self):
        return self.pad_id

    @property
    def mask(self):
        return self.mask_id

    @property
    def bos_token(self):
        """ Beginning of sentence token id """
        return self._bos_token

    @property
    def eos_token(self):
        """ End of sentence token id """
        return self._eos_token

    @property
    def additional_special_tokens(self):
        """ All the additional special tokens you may want to use (list of strings)."""
        return self._additional_special_tokens

    @property
    def bos_token_id(self):
        """ Id of the beginning of sentence token in the vocabulary."""
        return self._bos_token_id

    @property
    def eos_token_id(self):
        """ Id of the end of sentence token in the vocabulary."""
        return self._eos_token_id

    @property
    def additional_special_tokens_ids(self):
        """ Ids of all the additional special tokens in the vocabulary (list of integers)."""
        return [self.vocab.get(token) for token in self._additional_special_tokens]

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value


class _AQUILABPETokenizer(AbstractTokenizer):
    def __init__(self,
            tokenizer_path,
            bos_token='<sop>',
            eos_token='<eop>',
            end_token='</s>',
            ):
        name = 'Aquila BPE'
        super().__init__(name)

        from .aquila_tokenizer.uni_tokenizer.tokenizer import AQUILATokenizer
        self.tokenizer = AQUILATokenizer.from_pretrained("aquilachat-7b", cache_dir=tokenizer_path)

        self.bos_token = bos_token
        self.eod_id = self.tokenizer.encoder('<|endoftext|>')

    @property
    def bos_token_id(self):
        """bos_token_id"""
        return self.bos_token

    @property
    def vocab_size(self):
        """vocab_size"""
        return self.tokenizer.vocab_size

    @property
    def vocab(self):
        """vocab"""
        return self.tokenizer.get_vocab()

    @property
    def inv_vocab(self):
        """inv_vocab"""
        return self.tokenizer.decoder

    def tokenize(self, text):
        """tokenize"""
        return self.tokenizer.encode(text)


    @property
    def eod(self):
        """eod"""
        return self.eod_id


    def get_command_id(self, name):
        """get command token corresponding to `name`"""
        return self.command_name_map[name].Id

    def encode_plus(  # for Seq2seq
            self,
            source_text: str,
            second_text=None,
            target_text=None,
            truncation=True,
            max_length=None,
            padding=True,
    ):
        """encode_plus"""
        return self.tokenizer.encode_plus(source_text)


class _GPT2BPETokenizer(AbstractTokenizer):
    """Original GPT2 BPE tokenizer."""

    def __init__(self, vocab_file, merge_file):
        """__init__"""
        name = 'GPT2 BPE'
        super().__init__(name)

        self.tokenizer = GPT2Tokenizer(vocab_file, merge_file, errors='replace',
                                       special_tokens=[], max_len=None)
        self.eod_id = self.tokenizer.encoder['<|endoftext|>']

    @property
    def vocab_size(self):
        """vocab_size"""
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        """vocab"""
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        """inv_vocab"""
        return self.tokenizer.decoder

    def tokenize(self, text):
        """tokenize"""
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        """detokenize"""
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        """eod"""
        return self.eod_id


class _SentencePieceTokenizer(AbstractTokenizer):
    """SentencePieceTokenizer-Megatron wrapper"""

    def __init__(self, model_file, vocab_extra_ids=0):
        """__init__"""
        name = 'SentencePieceTokenizer'
        super().__init__(name)

        import sentencepiece
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file)
        self._initalize(vocab_extra_ids)

    def _populate_vocab(self):
        """_populate_vocab"""
        self._vocab = {}
        self._inv_vocab = {}

        for i in range(len(self.tokenizer)):
            t = self.tokenizer.id_to_piece(i)
            self._inv_vocab[i] = t
            self._vocab[t] = i

    def _initalize(self, vocab_extra_ids):
        """_initalize"""
        self._populate_vocab()
        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []

        def _add_special_token(t):
            if t not in self._vocab:
                next_id = len(self._vocab)
                self._vocab[t] = next_id
                self._inv_vocab[next_id] = t
            self._special_tokens[t] = self._vocab[t]
            self._inv_special_tokens[self._vocab[t]] = t

        _add_special_token('<CLS>')
        self._cls_id = self._vocab['<CLS>']
        _add_special_token('<SEP>')
        self._sep_id = self._vocab['<SEP>']
        _add_special_token('<EOD>')
        self._eod_id = self._vocab['<EOD>']
        _add_special_token('<MASK>')
        self._mask_id = self._vocab['<MASK>']

        pad_id = self.tokenizer.pad_id()
        try:
            pad_token = self.tokenizer.id_to_piece(pad_id)
        except IndexError:
            pad_token = '<PAD>'
        _add_special_token(pad_token)
        self._pad_id = self._vocab[pad_token]

        bos_id = self.tokenizer.bos_id()
        try:
            bos_token = self.tokenizer.id_to_piece(bos_id)
        except IndexError:
            bos_token = '<BOS>'
        _add_special_token(bos_token)
        self._bos_id = self._vocab[bos_token]

        eos_id = self.tokenizer.eos_id()
        try:
            eos_token = self.tokenizer.id_to_piece(eos_id)
        except IndexError:
            eos_token = '<EOS>'
        _add_special_token(eos_token)
        self._eos_id = self._vocab[eos_token]

        for i in range(vocab_extra_ids):
            t = "<extra_id_{}>".format(i)
            _add_special_token(t)
            self._t5_tokens += [t]

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    @property
    def decoder(self):
        return self._inv_vocab

    @property
    def encoder(self):
        return self._vocab

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L89
    def tokenize(self, text):
        ids = []
        idx = 0

        while 1:
            indices = {}
            for token in self._special_tokens:
                try:
                    indices[token] = text[idx:].index(token)
                except ValueError:
                    continue
            if len(indices) == 0:
                break

            next_token = min(indices, key=indices.get)
            next_idx = idx + indices[next_token]

            ids.extend(self.tokenizer.encode_as_ids(text[idx:next_idx]))
            ids.append(self._special_tokens[next_token])
            idx = next_idx + len(next_token)

        ids.extend(self.tokenizer.encode_as_ids(text[idx:]))
        return ids

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L125
    def detokenize(self, ids):
        text = ""
        last_i = 0

        for i, id in enumerate(ids):
            if id in self._inv_special_tokens:
                text += self.tokenizer.decode_ids(ids[last_i:i]) + " "
                text += self._inv_special_tokens[id] + " "
                last_i = i + 1

        text += self.tokenizer.decode_ids(ids[last_i:])
        return text

    @property
    def cls(self):
        return self._cls_id

    @property
    def sep(self):
        return self._sep_id

    @property
    def pad(self):
        return self._pad_id

    @property
    def bos_token_id(self):
        return self._bos_id

    @property
    def bos(self):
        return self._bos_id

    @property
    def eod(self):
        return self._eod_id

    @property
    def eos_token_id(self):
        return self._eos_id

    @property
    def eos(self):
        return self._eos_id

    @property
    def mask(self):
        return self._mask_id

    @property
    def additional_special_tokens_ids(self):
        return [self.vocab[k] for k in self._t5_tokens]


class _GPTSentencePieceTokenizer(_SentencePieceTokenizer):
    """SentencePieceTokenizer-Megatron wrapper"""

    def __init__(self, model_file, ):
        super().__init__(model_file, vocab_extra_ids=0)

    def _initalize(self, vocab_extra_ids):
        self._populate_vocab()

        self._pad_id = self.tokenizer.pad_id()
        self._bos_id = self.tokenizer.bos_id()
        self._eos_id = self.tokenizer.eos_id()

    def tokenize(self, text):
        return self.tokenizer.encode_as_ids(text)

    def detokenize(self, ids):
        return self.tokenizer.decode_ids(ids)

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self._eos_id

    @property
    def additional_special_tokens_ids(self):
        return None


class _NullTokenizer:
    def __init__(self, vocab_size):
        vocab_size = int(vocab_size)
        self._eos_id = vocab_size
        self.vocab_size = vocab_size + 1

    def tokenize(self, text):
        return [int(x) for x in text.split(' ')]

    def detokenize(self, ids):
        text = [str(x) for x in ids]
        return ' '.join(text)

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self._eos_id

    @property
    def additional_special_tokens_ids(self):
        return None


class _LLaMASentencePieceTokenizer(_SentencePieceTokenizer):
    """SentencePieceTokenizer-Megatron wrapper"""

    def __init__(self, model_file, add_bos_token=True, add_eos_token=False):
        super().__init__(model_file, vocab_extra_ids=0)
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

    def _initalize(self, vocab_extra_ids):
        self._populate_vocab()

        self._pad_id = self.tokenizer.pad_id()
        self._bos_id = self.tokenizer.bos_id()
        self._eos_id = self.tokenizer.eos_id()

    def tokenize(self, text):
        token_ids = self.tokenizer.encode_as_ids(text)

        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids + eos_token_id
        return output

    def detokenize(self, ids):
        return self.tokenizer.decode_ids(ids)

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self._eos_id

    @property
    def additional_special_tokens_ids(self):
        return None


class _GalacticaHFTokenizer(AbstractTokenizer):
    """_GalacticaHFTokenizer tokenizer."""

    def __init__(self, tokenizer_path):
        name = 'GalacticaHFTokenizer'
        super().__init__(name)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._set_tokenizer()

    def _set_tokenizer(self):
        """configures the tokenizer for galactica."""
        # from https://github.com/paperswithcode/galai/blob/main/galai/model.py#L143
        # setup padding
        self.tokenizer.pad_token_id = 1
        self.tokenizer.pad_token = "<pad>"
        self.tokenizer.padding_side = "left"

        # setup truncation
        self.tokenizer.truncation_side = "left"

        # setup special tokens
        self.tokenizer.bos_token_id = 0
        self.tokenizer.bos_token = "<s>"

        self.tokenizer.eos_token_id = 2
        self.tokenizer.eos_token = "</s>"

        self.tokenizer.unk_token = "<unk>"
        self.tokenizer.unk_token_id = 3

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def vocab(self):
        return self.tokenizer.vocab()

    def _escape_custom_split_sequence(self, text):
        """ Applies custom splitting to the text for GALILEO's tokenization"""

        # from https://github.com/paperswithcode/galai/blob/main/galai/utils.py#L43
        def _insert_split_marker(m: re.Match):
            """
            Applies split marker based on a regex match of special tokens such as
            [START_DNA].
            """
            SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"
            start_token, _, sequence, end_token = m.groups()
            sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
            return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"

        CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")
        return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)

    def tokenize(self, input_text):
        text = self._escape_custom_split_sequence(input_text)
        if not text:
            warnings.warn("Found an empty input text. Changing to end-of-document token instead.",
                          UserWarning
                          )
            text = self.tokenizer.eos_token
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.tokenizer.eos_token_id

    @property
    def pad(self):
        return self.tokenizer.pad_token_id

    def inv_vocab(self):
        pass


class _OptTokenizer(AbstractTokenizer):
    """_OptTokenizer tokenizer."""

    def __init__(self, tokenizer_path, vocab_file, merges_file):
        name = 'OptTokenizer'
        super().__init__(name)

        # from https://github.com/facebookresearch/metaseq/blob/main/metaseq/tasks/streaming_language_modeling.py#173
        from tokenizers import ByteLevelBPETokenizer, Tokenizer
        if tokenizer_path is not None:
            self.tokenizer = Tokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = ByteLevelBPETokenizer.from_file(vocab_file, merges_file)

        assert self.tokenizer.id_to_token(0) in {"<BOS>", "<s>"}
        assert self.tokenizer.id_to_token(1) in {"<PAD>", "<pad>"}
        assert self.tokenizer.id_to_token(2) in {"<EOS>", "</s>"}
        assert self.tokenizer.id_to_token(3) in {"<UNK>", "<unk>"}
        self.tokenizer.pad_token_id = 1
        self.tokenizer.eos_token_id = 2

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @property
    def vocab(self):
        return self.tokenizer.vocab()

    def tokenize(self, input_text):
        return self.tokenizer.encode(input_text).ids

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.tokenizer.eos_token_id

    @property
    def pad(self):
        return self.tokenizer.pad_token_id

    def inv_vocab(self):
        pass


class _AutoTokenizer(AbstractTokenizer):
    """AutoTokenizer for Hf Pretrained model loading."""

    def __init__(self, name, tokenizer_path):
        super().__init__(name)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.encoder = self.tokenizer.get_vocab()
        self.decoder = {v: k for k, v in self.encoder.items()}

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.tokenizer.eos_token_id


class _BloomHFTokenizer(_AutoTokenizer):
    """_BloomHFTokenizer tokenizer."""

    def __init__(self, tokenizer_path):
        super().__init__('BloomHFTokenizer', tokenizer_path)
