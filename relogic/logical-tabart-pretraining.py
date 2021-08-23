# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    set_seed,
)
from relogic.pretrainkit.multitask_trainer import Trainer
from relogic.pretrainkit.datasets.semparse.tabart import DataCollatorForTaBART, TaBARTDataset
from relogic.pretrainkit.datasets.semparse.text2sql import DataCollatorForQuerySchema2SQL, QuerySchema2SQLDataset
from relogic.pretrainkit.scorers.match_sequence import MatchSequenceScorer
from relogic.pretrainkit.models.semparse.logical_tabart import LogicalTaBARTModel
from relogic.pretrainkit.training_args import TrainingArguments

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

is_sagemaker = 'SM_MODEL_DIR' in os.environ

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    pretraining_model: Optional[str] = field(
        default=None, metadata={"help": "What is the model to use for pretraining."}
    )
    load_from_pretrained_ckpt: Optional[str] = field(
        default=None, metadata={"help": "Initialize the model with pretrained checkpoint"}
    )
    pretrained_ckpt_dir: Optional[str] = field(
        default="pretrained_checkpoint", metadata={"help": "Pretrained Checkpoint"}
    )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_names: Optional[str] = field(
        default=None, metadata={"help": "The name of tasks which are separated by ,"}
    )
    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    not_use_text: bool = field(
        default=False, metadata={"help": "To use text in pretraining or not"}
    )
    only_use_text: bool = field(
        default=False, metadata={"help": "To only use text in pretraining or not"}
    )
    cross_lingual: bool = field(
        default=False,
        metadata={"help": "Whether to use Cross-lingual Tabart Training"},
    )
    dump_file_name: str = field(
        default="eval_dump.json",
        metadata={"help": "The file name of evaluation dumping."}
    )

def get_dataset_by_name(pretraining_model, task_name, cross_lingual, tokenizer, file_path, use_text, only_use_text):
    if task_name != "text2sql":
        return TaBARTDataset(tokenizer=tokenizer, file_path=file_path, col_token="<col>",
                         task_name=task_name, use_text=use_text, only_use_text=only_use_text)
    if task_name == "text2sql":
        return QuerySchema2SQLDataset(tokenizer=tokenizer, file_path=file_path, task_name=task_name)


def get_datasets(pretraining_model, args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False):
    file_paths = args.eval_data_file.split(",") if evaluate else args.train_data_file.split(",")
    task_names = args.task_names.split(",")
    datasets = [get_dataset_by_name(pretraining_model, task_name, args.cross_lingual, tokenizer, file_path, not args.not_use_text, args.only_use_text)
                 for task_name, file_path in zip(task_names, file_paths)]
    return datasets

def get_data_collator_by_name(pretraining_model, task_name, cross_lingual, tokenizer):
    if task_name != "text2sql":
        return DataCollatorForTaBART(tokenizer=tokenizer, task=task_name, col_token="<col>")
    if task_name == "text2sql":
        return DataCollatorForQuerySchema2SQL(tokenizer=tokenizer)


def get_data_collators(pretraining_model, args: DataTrainingArguments, tokenizer: PreTrainedTokenizer):
    task_names = args.task_names.split(",")
    collators = [get_data_collator_by_name(pretraining_model, task_name, args.cross_lingual, tokenizer) for task_name in task_names]
    return collators


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if is_sagemaker:
        training_args.do_train = training_args.do_train_str == "True"
        training_args.do_eval = training_args.do_eval_str == "True"
        training_args.evaluate_during_training = training_args.evaluate_during_training_str == "True"
        data_args.train_data_file = ",".join([os.path.join(os.environ['SM_CHANNEL_TRAIN'], item) for item in data_args.train_data_file.split(",")])
        data_args.eval_data_file = ",".join([os.path.join(os.environ['SM_CHANNEL_TRAIN'], item) for item in data_args.eval_data_file.split(",")])
        training_args.output_dir = os.environ['SM_MODEL_DIR']
        model_args.pretrained_ckpt_dir = os.environ.get("SM_CHANNEL_PRETRAINED_CKPT_DIR", None)

    if model_args.pretrained_ckpt_dir is not None and model_args.load_from_pretrained_ckpt is not None:
        model_args.load_from_pretrained_ckpt = os.path.join(model_args.pretrained_ckpt_dir, model_args.load_from_pretrained_ckpt)

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if not is_sagemaker:
        if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    """Initialize models and tokenizer"""
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir, use_fast=False)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=False)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<col>"]})

    model = LogicalTaBARTModel(data_args.task_names)
    model.bert.resize_token_embeddings(len(tokenizer))
    model.bert_for_texttosql.resize_token_embeddings(len(tokenizer))
    model.bert.model.shared.weight = model.bert_for_texttosql.model.shared.weight
    model.bert.model.encoder.embed_tokens.weight = model.bert_for_texttosql.model.encoder.embed_tokens.weight

    if training_args.do_eval and not training_args.do_train:
        model_param = torch.load(os.path.join(model_args.model_name_or_path, "pytorch_model.bin"))
        model.load_state_dict(model_param)
        print("All key matched and load successfully.")

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Get datasets

    train_datasets = get_datasets(model_args.pretraining_model, data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_datasets = get_datasets(model_args.pretraining_model, data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
    # )
    data_collators = get_data_collators(model_args.pretraining_model, data_args, tokenizer=tokenizer)

    eos_id = None
    for data_collator in data_collators:
        if eos_id is None:
            eos_id = data_collator.label_eos_id
        else:
            assert eos_id == data_collator.label_eos_id
    match_sequence_scorer = MatchSequenceScorer(
        eos_id=eos_id, output_path=os.path.join(training_args.output_dir, "eval_dump.json"))
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collators=data_collators,
        train_datasets=train_datasets,
        eval_datasets=eval_datasets,
        compute_metrics=match_sequence_scorer
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
