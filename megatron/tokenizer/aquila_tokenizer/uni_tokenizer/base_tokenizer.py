"""BaseTokenizer"""
import os
from .properties import VOCAB_FILE, MERGES_FILE, SP_MODEL_FILE, \
        VOCAB_JSON_FILE, TOKENIZER_JSON_FILE, SPECIAL_TOKENS_MAP
import warnings


class BaseTokenizer(object):
    """Base class for all tokenizers."""
    @classmethod
    def from_pretrained(cls,
                        tokenizer_model_name,
                        cache_dir=None,
                        *inputs,
                        **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.

        Args:
            tokenizer_model_name (`str`):
                Name of the model associated with the tokenizer
            cache_dir (`str`):
                The directory that contains the vocab files, or will receive the downloaded vocab files
        """
        tokenizer_class = ""
        # search the cache directory for certain files
        if os.path.exists(cache_dir):
            files = os.listdir(cache_dir)
            if SP_MODEL_FILE in files:
                tokenizer_class = "sp"
            elif MERGES_FILE in files:
                tokenizer_class = "bpe"
            elif VOCAB_FILE in files:
                tokenizer_class = "wp"
        if tokenizer_class == "":
            #raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "Error: no tokenizer files")
            print("Error: no tokenizer files")
            exit()
        resolved_vocab_json_file = os.path.join(
            cache_dir, VOCAB_JSON_FILE) if VOCAB_JSON_FILE in files else None
        resolved_vocab_file = os.path.join(cache_dir, VOCAB_FILE)
        resolved_merges_file = os.path.join(cache_dir, MERGES_FILE)
        resolved_sp_file = os.path.join(cache_dir, SP_MODEL_FILE)
        special_tokens_map = os.path.join(cache_dir, SPECIAL_TOKENS_MAP)
        resolved_tokenizer_json_file = os.path.join(cache_dir, TOKENIZER_JSON_FILE)
        if tokenizer_class == "wp":
            return cls(vocab_file=resolved_vocab_file,
                       tokenizer_class=tokenizer_class,
                       tokenizer_model_name=tokenizer_model_name,
                       special_tokens_map=special_tokens_map,
                       cache_dir=cache_dir,
                       *inputs,
                       **kwargs)
        elif tokenizer_class == "bpe":
            return cls(vocab_file=resolved_vocab_json_file,
                       merges_file=resolved_merges_file,
                       tokenizer_class=tokenizer_class,
                       tokenizer_model_name=tokenizer_model_name,
                       special_tokens_map=special_tokens_map,
                       cache_dir=cache_dir,
                       *inputs,
                       **kwargs)
        elif tokenizer_class == "sp":
            return cls(vocab_file=resolved_vocab_json_file,
                       sp_model_file=resolved_sp_file,
                       tokenizer_class=tokenizer_class,
                       tokenizer_model_name=tokenizer_model_name,
                       tokenizer_json_file=resolved_tokenizer_json_file,
                       special_tokens_map=special_tokens_map,
                       cache_dir=cache_dir,
                       *inputs,
                       **kwargs)
        else:
            raise NotImplementedError(
                "Cannot find a tokenizer class that matches the files settings in the directory or ModelHub"
            )

    def __init__(self,
                 vocab_file=None,
                 merges_file=None,
                 sp_model_file=None,
                 tokenizer_json_file=None,
                 tokenizer_class=None,
                 tokenizer_model_name=None,
                 special_tokens_map=None,
                 cache_dir=None,
                 *inputs,
                 **kwargs):
        """init function"""
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.sp_model_file = sp_model_file
        self.tokenizer_class = tokenizer_class
        self.tokenizer_model_name = tokenizer_model_name
        self.tokenizer_json_file = tokenizer_json_file
        self.special_tokens_map = special_tokens_map
        self.cache_dir = cache_dir
        self.deprecation_warnings = ({})
