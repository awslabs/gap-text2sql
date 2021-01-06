import logging
import os
import pickle
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, List, NewType, Tuple
from tqdm import tqdm


import torch
from torch.utils.data.dataset import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.data.data_collator import DataCollator

logger = logging.getLogger(__name__)

label_mapping =  json.load(open("data/preprocessed_data/bart_parser_pretrain_label_mapping.json"))

def pad_and_tensorize_sequence(sequences, padding_value):
  max_size = max([len(sequence) for sequence in sequences])
  padded_sequences = []
  for sequence in sequences:
    padded_sequence = sequence + [padding_value] * (max_size - len(sequence))
    padded_sequences.append(padded_sequence)
  return torch.tensor(padded_sequences, dtype=torch.long)

class QuerySchema2SQLDataset(Dataset):
  """
  Dataset for pretraining task: query + schema -> SQL
  There is not masking for query and schema.
  """
  def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, local_rank=-1):
    assert os.path.isfile(file_path)
    logger.info("Creating features from dataset file at %s", file_path)

    self.examples = []
    self.keywords = label_mapping["keyword"]
    self.label_eos_id = self.keywords.index(label_mapping["label_eos_token"])
    self.label_bos_id = self.keywords.index(label_mapping["label_bos_token"])
    total, valid = 0, 0
    with open(file_path, encoding="utf-8") as f:
      for line in tqdm(f):
        total += 1
        example = json.loads(line)
        text = example["question"]
        columns = example["columns"] + example["tables"] + example["extra"] + example["negative"][:15]
        columns = [column.lower() for column in columns]

        # column_to_text = example["column_to_text"]
        column_to_text = {}
        for column in columns:
          column_text = column.replace(".", " ").replace("_", " ")
          column_to_text[column] = column_text.lower()
        sql = example["processed_sql"]
        text_tokens = [tokenizer.cls_token] + tokenizer.tokenize(text) + [tokenizer.sep_token]
        column_spans = []
        start_idx = len(text_tokens)
        for column in columns:
          column_tokens = tokenizer.tokenize(column_to_text[column])
          text_tokens.extend(column_tokens)
          text_tokens.append(tokenizer.sep_token)
          end_idx = start_idx + len(column_tokens)
          column_spans.append((start_idx, end_idx))
          start_idx = end_idx + 1
        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)

        if len(input_ids) > 600:
          continue

        label_ids = []
        try:
          for token in sql.split():
            token = token.lower()
            if token in columns:
              label_ids.append(columns.index(token) + len(self.keywords))
            else:
              label_ids.append(self.keywords.index(token))
        except:
          continue
        if len(label_ids) > 300:
          continue
        label_ids = [self.label_bos_id] + label_ids + [self.label_eos_id]

        self.examples.append({
          "idx": example["sql_id"],
          "input_ids": input_ids,
          "column_spans": column_spans,
          "label_ids": label_ids})
        valid += 1
    print("Valid Example {}; Invalid Example {}".format(valid, total-valid))

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, i):
    return self.examples[i]

@dataclass
class DataCollatorForQuerySchema2SQL:
  """
  Data collator used for query + schema -> sql modeling.
  """
  tokenizer: PreTrainedTokenizer
  label_padding_id = label_mapping["keyword"].index(label_mapping["label_padding_token"])
  label_eos_id = label_mapping["keyword"].index(label_mapping["label_eos_token"])
  label_bos_id = label_mapping["keyword"].index(label_mapping["label_bos_token"])
  logging_file = open("index_logging.txt", "w")
  def collate_batch(self, examples) -> Dict[str, torch.Tensor]:
    for example in examples:
      self.logging_file.write(str(example["idx"]) + "\n")
    input_ids_sequences = [example["input_ids"] for example in examples]
    column_spans_sequences = [example["column_spans"] for example in examples]
    label_ids_sequences = [example["label_ids"] for example in examples]
    padded_input_ids_tensor = pad_and_tensorize_sequence(
      input_ids_sequences, padding_value=self.tokenizer.pad_token_id)
    padded_column_spans_tensor = pad_and_tensorize_sequence(
      column_spans_sequences, padding_value=(0, 1))


    label_ids_tensor = pad_and_tensorize_sequence(
      label_ids_sequences, padding_value=self.label_padding_id)
    return {
      "input_ids": padded_input_ids_tensor,
      "column_spans": padded_column_spans_tensor,
      "labels": label_ids_tensor,
      "input_padding_id": self.tokenizer.pad_token_id,
      "label_padding_id": self.label_padding_id,
      "label_eos_id": self.label_eos_id,
      "label_bos_id": self.label_bos_id
    }

