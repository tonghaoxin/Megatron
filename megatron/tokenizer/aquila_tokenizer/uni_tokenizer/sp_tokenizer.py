# Copyright © 2022 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for using and training tokenizers (char, wordpiece, sentencepiece)"""

import logging

logger = logging.getLogger(__name__)
import sentencepiece as spm


class SentencePieceTokenizer(object):
    """SentencePieceTokenizer"""
    def __init__(self, model_path):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(model_path)
        # vocab = self.get_vocab()

    @property
    def vocab_size(self):
        """Returns the vocabulary size."""
        return self.sp_model.GetPieceSize()

    def get_vocab(self):
        """Returns the vocabulary as a dict from index to token. """
        vocab = {
            self.convert_id_to_token(i): i
            for i in range(self.vocab_size)
        }
        # vocab.update(self.added_tokens_encoder)
        return vocab

    def tokenize(self, text):
        """Tokenizes a string."""
        return self.sp_model.EncodeAsPieces(text)

    def convert_tokens_to_ids(self, tokens):
        """ Converts a token list to an id list. """
        return [self.sp_model.PieceToId(token) for token in tokens]

    def convert_token_to_id(self, token):
        """Converts a single token string to an id. """
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, idx):
        """Converts a single index to token string. """
        return self.sp_model.IdToPiece(int(idx))

    def convert_ids_to_tokens(self, idxs):
        """Converts a index list to token string. """
        return [self.sp_model.IdToPiece(int(idx)) for idx in idxs]

    def convert_tokens_to_string(self, tokens, all_command_token={}):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in all_command_token:
                out_string += self.sp_model.decode_pieces(
                    current_sub_tokens) + token + " "
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode_pieces(current_sub_tokens)
        return out_string.strip()
