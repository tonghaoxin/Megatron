#!/usr/bin/env python3 
"""auto tokenizer"""

from typing import Dict, List, Union

from transformers import AutoTokenizer

from megatron.tokenizer.megatron_tokenizer import MegatronTokenizer


class AutoTokenizerFromHF(MegatronTokenizer):
    """
    这里我们采用封装 HF Tokenizer 并继承 MegatronTokenizer 的方式实现, 这种做法的目的：
    1、使用体验和用户使用 HF 一致，方便使用 HuggingFace Tokenizer 提供的功能，后续如果有需要，可继续扩展接口；
    2、可满足 MegatronTokenizer 的接口定义, 避免出现与 Megatron 接口不兼容问题;
    """
    def __init__(self,
                 name_or_path: str,
                 use_fast_tokenizer: bool,
                 padding_side: str,
                 model_max_length: int,
                 split_special_tokens: bool,
                 **kwargs,
    ):
        super().__init__(name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path,
            use_fast=use_fast_tokenizer,
            padding_side=padding_side,
            split_special_tokens=split_special_tokens,
            model_max_length=model_max_length,
            trust_remote_code=True,
            **kwargs,
        )

    def tokenize(self, text: str):
        """tokenize text"""
        return self.tokenizer.encode(text)

    def detokenize(
        self,
        token_ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ):
        """Convert embedding ids to text"""
        return self.tokenizer.decode(token_ids=token_ids,
                                     skip_special_tokens=skip_special_tokens,
                                     clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                                     **kwargs)

    def add_special_tokens(
        self,
        special_tokens_dict,
        replace_additional_special_tokens: bool = True,
    ):
        """add special tokens"""
        return self.tokenizer.add_special_tokens(special_tokens_dict, replace_additional_special_tokens)

    @property
    def vocab(self):
        """get vocab"""
        return self.tokenizer.get_vocab()

    @property
    def inv_vocab(self):
        """get inverse vocab"""
        return {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self):
        """The vocabulary size"""
        return len(self.tokenizer)

    @property
    def cls(self):
        """The CLS token id"""
        return self.tokenizer.cls_token_id

    @property
    def sep(self):
        """The SEP token id"""
        return self.tokenizer.sep_token_id

    @property
    def pad(self):
        """The PAD token id"""
        return self.tokenizer.pad_token_id

    @property
    def eod(self):
        """The EOD token id"""
        return self.eos

    @property
    def bos(self):
        """The BOS token id"""
        return self.tokenizer.bos_token_id

    @property
    def eos(self):
        """The EOS token id"""
        return self.tokenizer.eos_token_id

    @property
    def mask(self):
        """The MASK token id"""
        return self.tokenizer.mask_token_id

    @property
    def chat_template(self):
        """The chat template"""
        return self.tokenizer.chat_template

    @chat_template.setter
    def chat_template(self, new_template):
        """set chat template"""
        # need transformers > 4.34
        self.tokenizer.chat_template = new_template

    @cls.setter
    def cls(self, value):
        """set CLS token id"""
        self.tokenizer.cls_token_id = value

    @sep.setter
    def sep(self, value):
        """set sep token id"""
        self.tokenizer.sep_token_id = value

    @pad.setter
    def pad(self, value):
        """set PAD token id"""
        self.tokenizer.pad_token_id = value

    @bos.setter
    def bos(self, value):
        """set BOS token id"""
        self.tokenizer.bos_token_id = value
        
    @eos.setter
    def eos(self, value):
        """set EOS token id"""
        self.tokenizer.eos_token_id = value
        
    @mask.setter
    def mask(self, value):
        """set MASK token id"""
        self.tokenizer.mask_token_id = value