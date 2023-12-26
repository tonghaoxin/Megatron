#!/usr/bin/env python3
# coding=utf-8
""" auto model """

import importlib
from .training import setup_model_and_optimizer
from .core.enums import ModelType


MODEL_MAPPING = {
    "llama": {
        "model_class": "LLaMAModel",
        "model_type": ModelType.encoder_or_decoder,
        "train_step_class": "LlamaTrainStep",
        "train_step_kwargs": {}
    },
    "glm": {
        "model_class": "GLMModel",
        "model_type": ModelType.encoder_or_decoder,
        "train_step_class": "GLMTrainStep",
        "train_step_kwargs": {}
    },
    "chatglm": {
        "model_class": "GLMModel",
        "model_type": ModelType.encoder_or_decoder,
        "train_step_class": "ChatGLMTrainStep",
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
        need_scheduler = kwargs.pop('need_scheduler', True)
        if model_name in cls._model_mapping.keys():
            model_class = cls._model_mapping[model_name]["model_class"]
            model_type = cls._model_mapping[model_name]["model_type"]
            Model = getattr(importlib.import_module("megatron.model"), model_class)
            model_provider_func = get_model_provider_func(Model)
            return setup_model_and_optimizer(model_provider_func, model_type, need_scheduler=need_scheduler)
        raise ValueError(
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    def get_train_step(cls, model_name):
        """ get train step """
        if model_name in cls._model_mapping.keys():
            train_step_class = cls._model_mapping[model_name]["train_step_class"]
            train_step_kwargs = cls._model_mapping[model_name]["train_step_kwargs"]
            return getattr(importlib.import_module("megatron.train"), train_step_class), train_step_kwargs
        raise ValueError(
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )
