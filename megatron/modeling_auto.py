#!/usr/bin/env python3
# coding=utf-8
""" auto model """

import importlib
from .training import get_model

MODEL_MAPPING = {
    "llama": {
        "model_class": "LLaMAModel",
        "pretrain_class": "pretrain_llama",
        "train_step_class": "LlamaTrainStep",
        "train_step_kwargs": {}
    },
    "glm": {
        "model_class": "GLMModel",
        "pretrain_class": "pretrain_glm",
        "train_step_class": "GLMTrainStep",
        "train_step_kwargs": {}
    },
    "chatglm": {
        "model_class": "GLMModel",
        "pretrain_class": "pretrain_chatglm",
        "train_step_class": "ChatGLMTrainStep",
        "train_step_kwargs": {}
    },
    "qwen": {
        "model_class": "QwenModel",
        "pretrain_class": "pretrain_qwen",
        "train_step_class": "QwenTrainStep",
        "train_step_kwargs": {}
    },
    "baichuan": {
        "model_class": "baichuanModel",
        "pretrain_class": "pretrain_baichuan",
        "train_step_class": "BaichuanTrainStep",
        "train_step_kwargs": {}
    }
}

def get_model_provider_func(ModelClass):
    """ get the model provider function """
    def func(pre_process=True, post_process=True, add_encoder=True, add_decoder=True):
        model = ModelClass(
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )
        return model

    return func

class AutoModel:
    """ AutoModel """
    _model_mapping = MODEL_MAPPING

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(model_name)`"
        )

    @classmethod
    def from_pretrained(cls, model_name, *model_args, **kwargs):
        """ from_pretrained """
        get_model, model_provider = cls.get_model_provider(model_name)
        return get_model(model_provider)

    @classmethod
    def get_train_step(cls, model_name):
        """ get train step """
        if model_name in cls._model_mapping.keys():
            train_step_class = cls._model_mapping[model_name]["train_step_class"]
            train_step_kwargs = cls._model_mapping[model_name]["train_step_kwargs"]
            return getattr(importlib.import_module("megatron.train"), train_step_class), train_step_kwargs
        raise ValueError(
            f"Model type should be one of {', '.join(cls._model_mapping.keys())}."
        )

    @classmethod
    def get_model_provider(cls, model_name):
        """ get model p """
        if model_name in cls._model_mapping.keys():
            pretrain = cls._model_mapping[model_name]["pretrain_class"]
            model_provider = getattr(importlib.import_module(f"megatron.{pretrain}"), "model_provider")
            return get_model, model_provider
        raise ValueError(
            f"Model type should be one of {', '.join(cls._model_mapping.keys())}."
        )

    @classmethod
    def get_num_layers(cls, config):
        """ num layers """
        for k in ["num_hidden_layers", "num_layers"]:
            if hasattr(config, k):
                return getattr(config, k)
        raise ValueError(f"Unknown config {config}")

    @classmethod
    def get_hidden_size(cls, config):
        """ hidden size """
        for k in ["hidden_size"]:
            if hasattr(config, k):
                return getattr(config, k)
        raise ValueError(f"Unknown config {config}")

    @classmethod
    def get_num_attention_heads(cls, config):
        """" num attention heads """
        for k in ["num_attention_heads"]:
            if hasattr(config, k):
                return getattr(config, k)
        raise ValueError(f"Unknown config {config}")

    @classmethod
    def get_max_position_embeddings(cls, config):
        """ max position embeddings """
        for k in ["max_position_embeddings"]:
            if hasattr(config, k):
                return getattr(config, k)
        return None

    @classmethod
    def get_vocab_size(cls, config):
        """ vocab size """
        for k in ["vocab_size"]:
            if hasattr(config, k):
                return getattr(config, k)
        raise ValueError(f"Unknown config {config}")

