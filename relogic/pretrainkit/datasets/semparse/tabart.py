from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from tqdm import tqdm
import json
from dataclasses import dataclass
import torch
from relogic.pretrainkit.datasets.utils import pad_and_tensorize_sequence
import random
import numpy as np
import bisect
import itertools

class TaBARTDataset(Dataset):
  """
  This dataset is used for pretraining task on generation-based or retrieval-based
  text-schema pair examples.
  The fields that will be used is `question`, `table_info.header`, `entities`.
  We already make sure that every entity in `entities` will be in `table_info.header`.
  """
  def __init__(self,
               tokenizer: PreTrainedTokenizer,
               file_path: str,
               col_token: str,
               task_name: str,
               use_text: bool,
               only_use_text: bool):
    self.task_name = task_name
    self.examples = []
    total = 0
    valid = 0
    with open(file_path, encoding="utf-8") as f:
      for line in tqdm(f):
        total += 1
        example = json.loads(line)
        text = example["question"]
        if "header" in example["table_info"]:
          schema = example["table_info"]["header"]
        else:
          schema = example["table_info"]["columns"]
        entity_with_value = example.get("with_value_entity", [])
        if use_text:
          question_tokens = [tokenizer.cls_token] + tokenizer.tokenize(text, add_prefix_space=True)
        else:
          question_tokens = []
        # segment_ids = [1] * len(question_tokens) # we want to make col_token as segment 2

        column_spans = []
        column_tokens = []
        start_idx = len(question_tokens)
        for column in schema:
          start_idx += 1 # This is for column separator
          tokenized_column = tokenizer.tokenize(column.lower(), add_prefix_space=True)
          column_tokens.append([col_token] + tokenized_column)
          column_spans.append((start_idx, start_idx + len(tokenized_column)))
          start_idx += len(tokenized_column)

        question_token_ids = tokenizer.convert_tokens_to_ids(question_tokens)
        column_token_ids = []
        for column in column_tokens:
          column_token_ids.append(tokenizer.convert_tokens_to_ids(column))
        # segment_ids.extend([2] * (len(input_ids) - len(segment_ids))) # make the rest of the tokens to be segment 2

        if only_use_text:
          input_ids = question_token_ids + [tokenizer.sep_token_id]
        else:
          input_ids = question_token_ids + list(itertools.chain.from_iterable(column_token_ids)) + [tokenizer.sep_token_id]

        entity_to_value = example.get("entity_to_value", {})


        entities = example.get("entities", {})
        column_labels = [0] * len(schema)
        column_type = [0] * len(schema)
        entity_used = []
        for entity, value in entities.items():
          if entity != "limit" and entity != "*":
            entity_used.append(entity)
            column_labels[schema.index(entity)] = 1
            if entity in entity_with_value:
              column_type[schema.index(entity)] = 2
            else:
              column_type[schema.index(entity)] = 1
        if start_idx > 250:
          continue
        self.examples.append({
          "question_token_ids": question_token_ids,
          "column_token_ids": column_token_ids,
          "input_ids": input_ids,
          "column_spans": column_spans,
          "column_labels": column_labels,
          "column_type": column_type,
          "columns": schema,
          "entity_to_value": entity_to_value,
          "entity_used": entity_used
        })


        valid += 1
        # Create input
    print("Total {} and Valid {}".format(total, valid))
  def __len__(self):
    return len(self.examples)

  def __getitem__(self, i):
    return self.examples[i]


@dataclass
class DataCollatorForTaBART:
  tokenizer: PreTrainedTokenizer
  task: str
  col_token: str
  mlm_probability: float = 0.35
  seed = 3435

  def __post_init__(self):
    self.label_bos_id = self.tokenizer.cls_token_id
    self.label_eos_id = self.tokenizer.sep_token_id
    self.g = torch.Generator()
    self.g.manual_seed(self.seed)

  def generate_mlm_data(self, padded_input_ids_tensor, ignore_segment):
    inputs, labels = self.mask_tokens(padded_input_ids_tensor.clone(), ignore_segment=ignore_segment)
    return {
      "task": "mlm",
      "input_ids": inputs,
      "labels": padded_input_ids_tensor,
      "pad_token_id": self.tokenizer.pad_token_id,
      "label_bos_id": self.tokenizer.bos_token_id,
      "label_eos_id": self.tokenizer.eos_token_id,
      "label_padding_id": self.tokenizer.pad_token_id}

  def generate_col_pred_data(self, examples, padded_input_ids_tensor, use_type=False):
    if use_type:
      column_labels_sequences = [example["column_type"] for example in examples]
      padded_label_ids_tensor = pad_and_tensorize_sequence(column_labels_sequences,
                                                            padding_value=-100)
    else:
      column_labels_sequences = [example["column_labels"] for example in examples]
      padded_label_ids_tensor = pad_and_tensorize_sequence(column_labels_sequences,
                                                         padding_value=-100)
    column_spans_sequences = [example["column_spans"] for example in examples]
    padded_column_spans_tensor = pad_and_tensorize_sequence(column_spans_sequences,
                                                            padding_value=(0, 1))
    return {
      "task": "col_pred" if not use_type else "col_type",
      "input_ids": padded_input_ids_tensor,
      "column_spans": padded_column_spans_tensor,
      "labels": padded_label_ids_tensor,
      "pad_token_id": self.tokenizer.pad_token_id}

  def replace_col_with_value(self, examples, padded_output_ids_tensor):
    question_token_sequences = [example["question_token_ids"] for example in examples]
    column_token_sequences = []
    input_ids_sequences = []
    for example in examples:
      column_ids = []
      for idx, column in enumerate(example["columns"]):
        if column in example["entity_to_value"]:
          if example["column_labels"][idx] > 0:
            # if random.random() < 0.5:
            value = random.sample(example["entity_to_value"][column], 1)[0]
            column_ids.append(self.tokenizer.convert_tokens_to_ids(
              [self.col_token] + self.tokenizer.tokenize(value.lower())))
            # else:
            #   pass
              # Column delete
          else:
            if random.random() < 0.5:
              value = random.sample(example["entity_to_value"][column], 1)[0]
              column_ids.append(self.tokenizer.convert_tokens_to_ids(
                [self.col_token] + self.tokenizer.tokenize(value.lower())))
            else:
              column_ids.append(example["column_token_ids"][idx])
        else:
          column_ids.append(example["column_token_ids"][idx])
      column_ids = list(itertools.chain.from_iterable(column_ids))
      column_token_sequences.append(column_ids + [self.tokenizer.sep_token_id])
    for question, column in zip(question_token_sequences, column_token_sequences):
      input_ids_sequences.append(question + column)
    padded_input_ids_tensor = pad_and_tensorize_sequence(input_ids_sequences,
                                                         padding_value=self.tokenizer.pad_token_id)
    return {
      "task": "col_rev",
      "input_ids": padded_input_ids_tensor,
      "labels": padded_output_ids_tensor,
      "pad_token_id": self.tokenizer.pad_token_id,
      "label_bos_id": self.tokenizer.bos_token_id,
      "label_eos_id": self.tokenizer.eos_token_id,
      "label_padding_id": self.tokenizer.pad_token_id}

  def create_table_pred_data(self, examples):
    input_ids_sequences = []
    labels = []
    for example in examples:
      if random.random() < 0.5:
        entity_used = example["entity_used"]
        entity_to_remove = random.sample(entity_used, 1)
        labels.append([1])
      else:
        entity_to_remove = None
        labels.append([0])

      processed_column_ids = []
      for column, column_id in zip(example["columns"], example["column_token_ids"]):
        if column != entity_to_remove:
          processed_column_ids.extend(column_id)
      input_ids_sequences.append(example["question_token_ids"] + processed_column_ids + [self.tokenizer.sep_token_id])
    padded_input_ids_tensor = pad_and_tensorize_sequence(input_ids_sequences,
                                                         padding_value=self.tokenizer.pad_token_id)
    label_ids_tensor = torch.tensor(labels, dtype=torch.long)
    return {
      "task": "table_pred",
      "input_ids": padded_input_ids_tensor,
      "labels": label_ids_tensor,
      "pad_token_id": self.tokenizer.pad_token_id,
      "label_bos_id": self.tokenizer.bos_token_id,
      "label_eos_id": self.tokenizer.eos_token_id,
      "label_padding_id": self.tokenizer.pad_token_id}


  def __call__(self, examples):
    # Task: ["mlm", "col_rev", "col_pred", "col_type", "table_pred", "miss_col"]
    tasks = self.task.split("+")
    threshold = np.cumsum([1/len(tasks)] * len(tasks))
    ind = bisect.bisect(threshold, random.random())
    task = tasks[ind]

    if task == "mlm":
      input_ids_sequences = [example["input_ids"] for example in examples]
      padded_input_ids_tensor = pad_and_tensorize_sequence(input_ids_sequences,
                                                           padding_value=self.tokenizer.pad_token_id)
      return self.generate_mlm_data(padded_input_ids_tensor, None)

    if task == "col_pred":
      input_ids_sequences = [example["input_ids"] for example in examples]
      padded_input_ids_tensor = pad_and_tensorize_sequence(input_ids_sequences,
                                                           padding_value=self.tokenizer.pad_token_id)
      return self.generate_col_pred_data(examples, padded_input_ids_tensor, False)

    if task == "col_type":
      input_ids_sequences = [example["input_ids"] for example in examples]
      padded_input_ids_tensor = pad_and_tensorize_sequence(input_ids_sequences,
                                                           padding_value=self.tokenizer.pad_token_id)
      return self.generate_col_pred_data(examples, padded_input_ids_tensor, True)

    if task == "col_rev":
      input_ids_sequences = [example["input_ids"] for example in examples]
      padded_output_ids_tensor = pad_and_tensorize_sequence(input_ids_sequences,
                                                           padding_value=self.tokenizer.pad_token_id)
      return self.replace_col_with_value(examples, padded_output_ids_tensor)

    if task == "table_pred":
      return self.create_table_pred_data(examples)


  def old_call(self, examples):
    input_ids_sequences = [example["input_ids"] for example in examples]
    segment_ids_sequences = [example["segment_ids"] for example in examples]
    padded_input_ids_tensor = pad_and_tensorize_sequence(input_ids_sequences,
                                                         padding_value=self.tokenizer.pad_token_id)
    padded_segment_ids_tensor = pad_and_tensorize_sequence(segment_ids_sequences,
                                                           padding_value=0)


    if self.task == "mlm":
      return self.generate_mlm_data(padded_input_ids_tensor, None)
    elif self.task == "cond_mlm":
      if random.random() < 0.5:
        ignore_segment = padded_segment_ids_tensor == 1
      else:
        ignore_segment = padded_segment_ids_tensor == 2
      return self.generate_mlm_data(padded_input_ids_tensor, ignore_segment)
    elif self.task == "col_pred":
      return self.generate_col_pred_data(examples, padded_input_ids_tensor)
    elif self.task == "mlm+col_pred":
      if random.random() < 0.5:
        return self.generate_mlm_data(padded_input_ids_tensor, None)
      else:
        return self.generate_col_pred_data(examples, padded_input_ids_tensor)
    elif self.task == "cond_mlm+col_pred":
      if random.random() < 0.5:
        if random.random() < 0.5:
          ignore_segment = padded_segment_ids_tensor == 1
        else:
          ignore_segment = padded_segment_ids_tensor == 2
        return self.generate_mlm_data(padded_input_ids_tensor, ignore_segment)
      else:
        return self.generate_col_pred_data(examples, padded_input_ids_tensor)


  def mask_tokens(self, inputs, ignore_segment=None):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    If ignore_segment is not None, then this mask method will only mask query or only mask columns, but not both.
    """

    if self.tokenizer.mask_token is None:
      raise ValueError(
        "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
      )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, self.mlm_probability)
    special_tokens_mask = [
      self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

    if ignore_segment is not None:
      # This is for conditional masking
      probability_matrix.masked_fill_(ignore_segment, value=0.0)

    if self.tokenizer._pad_token is not None:
      padding_mask = labels.eq(self.tokenizer.pad_token_id)
      probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels
