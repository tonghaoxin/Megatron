# !/usr/bin/env python3
# This code is originally from https://github.com/bigscience-workshop/Megatron-DeepSpeed
# under the license https://huggingface.co/spaces/bigscience/license

from functools import reduce
from logging import logMultiprocessing
import os
import sys
import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,os.path.pardir)))

from lm_eval.models.gpt2 import GPT2LM
from lm_eval.base import BaseLM
from lm_eval import evaluator, tasks, utils
from lm_eval.base import CacheHook
from tqdm import tqdm
import torch.nn.functional as F

from lm_eval.tasks import ALL_TASKS
from pretrain_gpt import model_provider
import numpy as np

import torch
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.model import BLOOMModel
from megatron.core import mpu, tensor_parallel
from megatron.checkpointing import load_checkpoint
from megatron.training import setup_model_and_optimizer, get_model
#from megatron.mpu.mappings import gather_from_tensor_model_parallel_region

from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.core.pipeline_parallel import p2p_communication
import pickle
import json

from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model.distributed import DistributedDataParallel as LocalDDP
from megatron.model.module import Float16Module
from megatron.accelerator import get_accelerator

class EvalHarnessAdaptor(GPT2LM):
    def __init__(self, model, tokenizer):
        args = get_args()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.VOCAB_SIZE = tokenizer.vocab_size
        self.EOT_TOKEN_ID = tokenizer.tokenizer.eos_token_id
        self.BOS_TOKEN_ID = tokenizer.tokenizer.bos_token_id

        self._max_length = args.seq_length

        # For ds we split into mini batches and then micro batches to keep pipelining api happy.
        # With Megatron we just go to micro_batches directly
        self._batch_size = args.micro_batch_size * args.micro_bs_multiplier

        self.cache_hook = CacheHook(None)
        self.is_main = args.rank == 0
        self.is_local_main = args.local_rank == 0
        self._device = get_accelerator().current_device()
        self.is_model_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        self.is_pipe_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
        self.is_data_parallel = mpu.get_data_parallel_world_size() > 1
        self.adaptive_seq_len = args.adaptive_seq_len
        if self.is_data_parallel:
            raise NotImplementedError("Data parallelism is currently not supported for evaluation")

        self.is_last_stage = True if not self.is_pipe_parallel else mpu.is_pipeline_last_stage()  # only the last stage of the pipeline model will receive the logits

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device


    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc = [self.EOT_TOKEN_ID]
            else:
                context_enc = [self.BOS_TOKEN_ID] + self.tokenizer_encode(context)

            continuation_enc = self.tokenizer_encode(continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        loglikelihoods = []
        with torch.no_grad():
            for string, in tqdm(requests):
                rolling_token_windows = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                    token_list=self.tokenizer_encode(string),
                    prefix_token=self.EOT_TOKEN_ID,
                    max_seq_len=self.max_length,
                    context_len=1,
                )))

                rolling_token_windows = [(None,) + x for x in rolling_token_windows]

                # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for that
                string_nll = self._loglikelihood_tokens(rolling_token_windows, disable_tqdm=True)

                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

                string_nll = sum(string_nll)
                loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        disable_tqdm = disable_tqdm if self.is_main else True
        res = []
        res_len = 0  # storing the result length for later
        self.model.eval()
        with torch.no_grad():
            def _collate(x):
                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            reord = utils.Reorderer(requests, _collate)
            for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
                inps, contlens, inplens, padding_length = [], [], [], None
                for _, context_enc, continuation_enc in chunk:
                    # when too long to fit in context, truncate from the left
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1):][:-1]
                        , dtype=torch.long).to(self.device)
                    inplen, = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else inplen
                    if not self.adaptive_seq_len:
                        padding_length = self.max_length
                    # pad to length
                    inp = torch.cat([
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device)  # [padding_length - seq]
                    ], dim=0)

                    inps.append(inp.unsqueeze(0))

                    contlens.append(cont)
                    inplens.append(inplen)

                logits = self._model_call(torch.cat(inps, dim=0))
                res_len += len(chunk)
                if logits is not None:
                    multi_logits = F.log_softmax(logits, dim=-1).cpu()  # [batch, seq, vocab]

                    for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(chunk, multi_logits, inps, inplens, contlens):
                        contlen = len(cont_toks)
                        logits = logits[inplen - contlen:inplen].unsqueeze(0)  # [1, seq, vocab]
                        greedy_tokens = logits.argmax(dim=-1)
                        # cont_toks :: [1, seq]
                        cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)
                        max_equal = (greedy_tokens == cont_toks).all()
                        # last_token_slice = logits[:, -1, :].squeeze(0).tolist()

                        logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]
                        answer = (float(logits.sum()), bool(max_equal))
                        # partial caching
                        if cache_key is not None:
                            self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                        res.append(answer)

        if not mpu.is_pipeline_last_stage():
            # @HACK: To make the eval harness happy on threads that don't have access to the results.
            #        We just randomly generate some data.
            res = [(np.random.rand(), np.random.rand()>0.5) for _ in requests]

        return reord.get_original(res)

    def create_model_inputs(self, tokens):
        args = get_args()

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            self.EOT_TOKEN_ID,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)
            #prefix_indices=None,
            #loss_on_targets_only=False)

        return (tokens, position_ids, attention_mask), (tokens, loss_mask)

    def _model_call(self, inps):
        args = get_args()

        if False:
            self.model.set_batch_fn(self.create_model_inputs)
            # round up to multiple of micro_batch_size
            new_size = ((len(inps) + args.micro_batch_size-1)  // args.micro_batch_size) * args.micro_batch_size
            padded = F.pad(inps, (0, 0, 0, new_size-len(inps)), value = 0)
            # dummy data iterator for pipelining.
            data_iterator = list((torch.stack(inp) for inp in utils.chunks(padded, args.micro_batch_size)))
            self.model.micro_batches = len(data_iterator)

            if self.adaptive_seq_len:
                # Allow different shapes than the default seq_len to be communicated across pipes
                # Without this Deepspeed will hang when trying to receive activations
                self.model.reset_activation_shape()

            output = self.model.eval_batch(iter(data_iterator), compute_loss = False, reduce_output = None)


            if output is not None:
                output = torch.cat(output, 0)[:len(inps)]
            else:
                output = None

            # hack #2 for adaptive_seq_len to work as total_loss gets appended to and shapes aren't the same
            if args.adaptive_seq_len:
                self.model.total_loss = None
        else:
            # Since the shape of the micro-batch will change
            # We need set the correct shapes here
            # So that latter pipeline stages knows which shapes to expect.
            # Otherwise we will deadlock.

            args.micro_batch_size = len(inps)
            args.seq_length = len(inps[0])
            args.max_position_embeddings = args.seq_length
            tensor_shape = (args.seq_length, args.micro_batch_size, args.hidden_size)
            dtype = args.params_dtype
            if args.fp32_residual_connection:
                dtype = torch.float

            input_tensor = p2p_communication.recv_forward(tensor_shape, dtype)

            # Forward pass through the model.
            unwrapped_model = unwrap_model(self.model, (torchDDP, LocalDDP, Float16Module))
            unwrapped_model.set_input_tensor(input_tensor)
            output = self.model(*self.create_model_inputs(inps)[0])
            p2p_communication.send_forward(output)

        if mpu.is_pipeline_last_stage():
            return tensor_parallel.gather_from_tensor_model_parallel_region(output)[..., :self.tokenizer.vocab_size]
        else:
            return None

    def tokenizer_encode(self, text):
        """Tokenize text *without* adding special tokens."""
        # Splitting this into its own method in case we need to handle special cases for different tokenizers
        from megatron.tokenizer.gpt2_tokenization import GPT2Tokenizer
        if isinstance(self.tokenizer.tokenizer, GPT2Tokenizer):
            return self.tokenizer.tokenizer.encode(text)
        else:
            return self.tokenizer.tokenizer.encode(text)


from megatron.initialize import initialize_megatron
import megatron

def override_args(args, override_args, skip_keys, skip_if_specified_keys):
    for k, v in vars(override_args).items():
        if k in skip_keys:
            continue
        if k in skip_if_specified_keys and getattr(args, k) is not None:
            continue
        setattr(args, k, v)

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building BLOOM model ...')
    model = BLOOMModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )

    return model

def add_text_generate_args(parser):
    results_path_default = f"results-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    group.add_argument('--task_list', type=str, default = "all", help='Either "all" or comma separated list of tasks.')
    group.add_argument('--results_path', type=str, default = results_path_default, help='Path to where the results will be stored.')
    group.add_argument('--adaptive_seq_len',  default = False, action='store_true',
                       help='Should the sequence length be adapted to the batch during evaluation, if in fp16 the results will be slightly different due to numerical errors but greatly speed up evaluation.')
    group.add_argument('--eval_fp32',  default = False, action='store_true', help='Should the evaluation run in fp32')
    group.add_argument('--intermed_results',  default = False, action='store_true', help='Whether to print & write intermediate results for each task')
    group.add_argument('--bootstrap_iters', type=int, default=100000, help='How many iterations to use for stderr estimation')
    group.add_argument('--micro_bs_multiplier', type=int, default=1, help='Increase the global batch size to remove bubble when pipeline parallel')
    return parser

def tasks_args(parser):

    results_path_default = f"results-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"

    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='Evaluation options')
    group.add_argument('--task_list', type=str, default = "all", help='Either "all" or comma separated list of tasks.')
    group.add_argument('--results_path', type=str, default = results_path_default, help='Path to where the results will be stored.')
    group.add_argument('--adaptive_seq_len',  default = False, action='store_true',
                       help='Should the sequence length be adapted to the batch during evaluation, if in fp16 the results will be slightly different due to numerical errors but greatly speed up evaluation.')
    group.add_argument('--eval_fp32',  default = False, action='store_true', help='Should the evaluation run in fp32')
    group.add_argument('--intermed_results',  default = False, action='store_true', help='Whether to print & write intermediate results for each task')
    group.add_argument('--bootstrap_iters', type=int, default=100000, help='How many iterations to use for stderr estimation')
    group.add_argument('--micro_bs_multiplier', type=int, default=1, help='Increase the global batch size to remove bubble when pipeline parallel')
    return parser

#from megatron.global_vars import _parse_args

def main():
    # parse the megatron args. But wait with initalizing megatron.
    # avoid printing the arguments, since they will later be overridden.
    #args = _parse_args(tasks_args)
    #model = load_ds_checkpoint_and_setup_megatron(args)

    #args = get_args()
    #if args.deepspeed and args.adaptive_seq_len:
    #    # adaptive_seq_len hack #1:
    #    # CL automatically enables reset_activation_shape() which allows us to change input shapes
    #    # and it also reshapes the attenion scores in attention_mask_func
    #    args.curriculum_learning = 1
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'BloomHFTokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()
    load_path = args.load
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None, strict=True)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    task_list = ALL_TASKS if args.task_list == 'all' else args.task_list.split(',')
    task_dict = tasks.get_task_dict(task_list)

    #model.module.activation_checkpoint_interval = 0
    #model._compute_loss = False
    #model.fwd_outputs = []

    tokenizer = get_tokenizer()
    adaptor = EvalHarnessAdaptor(model, tokenizer)

    if args.intermed_results:
        global_results = {"results": {}, "versions": {}}
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        iteration_id = load_path.split("/")[-1].replace("/", "")
        results_path = args.results_path.replace(".json", f"_lm-eval_{iteration_id}_{timestamp}.json")
        # Backup file in case of interruption during writing
        results_path_backup = args.results_path.replace(".json", f"_lm-eval_{iteration_id}_{timestamp}_backup.json")
        for task_name, task in task_dict.items():
            results = evaluator.evaluate(adaptor, {task_name: task}, False, 0, None, bootstrap_iters=args.bootstrap_iters)
            global_results["results"] = {**global_results["results"], **results["results"]}
            global_results["versions"] = {**global_results["versions"], **results["versions"]}
            if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0:
                print(json.dumps(results, indent=2))
                with open(results_path, 'w') as outfile:
                    json.dump(global_results, outfile, indent=4)
                with open(results_path_backup, 'w') as outfile:
                    json.dump(global_results, outfile, indent=4)
    else:
        global_results = evaluator.evaluate(adaptor, task_dict, False, 0, None, bootstrap_iters=args.bootstrap_iters)
        if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0:
            print(json.dumps(global_results, indent=2))
            with open(args.results_path, 'w') as outfile:
                json.dump(global_results, outfile, indent=4)

if __name__ == '__main__':
    main()
