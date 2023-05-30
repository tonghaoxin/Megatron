""" chatglm dataset """
from datasets import load_dataset
from megatron import get_tokenizer
from megatron import get_args

import os


def build_train_valid_test_datasets(
    data_prefix, splits_string, train_valid_test_num_samples, seq_length,
    length_per_sample, aggregated_samples_per_sequence, args
):      
    tokenizer = get_tokenizer()
    # Load dataset
    data_files = {}
    file_dir = data_prefix[0]
    data_files["train"] = os.path.join(file_dir, "train.json")
    extension = "json"    
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir="",
        use_auth_token=False,
    )
    args = get_args()

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names_train = raw_datasets["train"].column_names

    prompt_column = args.prompt_resource_name if args.prompt_resource_name else "content"
    response_column = args.prompt_target_name if args.prompt_resource_name else "summary"
    history_column = args.prompt_history_name if args.prompt_history_name else None
    test_dataset = None
    valid_dataset = None
    train_dataset = raw_datasets["train"]
    prefix = ""
    
    def preprocess_data(examples):
  
        max_seq_length = seq_length
        max_source_length = seq_length // 2
        max_target_length = seq_length // 2
        model_inputs = {
            "input_ids": [],
            "labels": [],
            "loss_mask": [],
        }
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]

                if history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                prompt = prefix + prompt
                a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

                if len(a_ids) > max_source_length - 1:
                    a_ids = a_ids[: max_source_length - 1]

                if len(b_ids) > max_target_length - 2:
                    b_ids = b_ids[: max_target_length - 2]

                input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

                context_length = input_ids.index(tokenizer.bos_token_id)
                mask_position = context_length - 1
                labels = [-100] * context_length + input_ids[mask_position + 1:]
                
                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
                labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

                loss_mask = [(0 if k == -100 else 1) for k in labels ]
                
                model_inputs["loss_mask"].append(loss_mask)
                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)
                

        return model_inputs
    # 将string转换到int
    train_dataset = train_dataset.map(
                preprocess_data,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names_train,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
            )

    train_dataset.set_format(
                type=train_dataset.format["type"], columns=train_dataset.column_names, 
                format_kwargs=train_dataset.format["format_kwargs"]
            )
    return train_dataset, valid_dataset, test_dataset
