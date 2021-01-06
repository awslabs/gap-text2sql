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
from transformers.tokenization_bart import BartTokenizer
from transformers.tokenization_roberta import RobertaTokenizer
from relogic.pretrainkit.datasets.utils import pad_and_tensorize_sequence
logger = logging.getLogger(__name__)

label_mapping =  json.load(open("data/preprocessed_data/bart_parser_label_mapping_2.json"))

class QuerySchemaRelation2SQLDataset(Dataset):
  """
  Dataset for relation-aware text-to-SQL: query + schema + relation -> SQL
  """
  def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, local_rank=-1):
    self.examples = []
    self.keywords = label_mapping["keyword"]
    self.label_eos_id = self.keywords.index(label_mapping["label_eos_token"])
    self.label_bos_id = self.keywords.index(label_mapping["label_bos_token"])
    add_prefix_space = isinstance(tokenizer, BartTokenizer) or isinstance(tokenizer, RobertaTokenizer)
    total, valid = 0, 0
    with open(file_path, encoding="utf-8") as f:
      for line in tqdm(f):
        total += 1
        example = json.loads(line)
        text = example["normalized_question"]
        columns = example["columns"]
        tables = example["tables"]
        columns_text = example["column_text"]
        tables_text = example["table_text"]
        sql = example["sql"]
        # we need the adjusted token index info.
        token_idx_to_sub_token_start_idx = {}
        text_tokens = [tokenizer.cls_token]
        start_idx = 0 # This is for adjusting the sc_link and cv_link
        for idx, token in enumerate(text.split()):
          sub_tokens = tokenizer.tokenize(token, add_prefix_space=add_prefix_space)
          token_idx_to_sub_token_start_idx[idx] = start_idx
          text_tokens.extend(sub_tokens)
          start_idx += len(sub_tokens)
        text_tokens.append(tokenizer.sep_token)
        question_start, question_end = 1, len(text_tokens) - 1 # exclusive

        column_spans = []
        start_idx = len(text_tokens)
        for column_tokens in columns_text:
          column_str = " ".join(column_tokens)
          column_tokens = tokenizer.tokenize(column_str, add_prefix_space=add_prefix_space)
          text_tokens.extend(column_tokens)
          text_tokens.append(tokenizer.sep_token)
          end_idx = start_idx + len(column_tokens)
          column_spans.append((start_idx, end_idx))
          start_idx = end_idx + 1

        column_start = [column_span[0] for column_span in column_spans]
        column_end = [column_span[1] for column_span in column_spans]

        table_spans = []
        start_idx = len(text_tokens)
        for table_tokens in tables_text:
          table_str = " ".join(table_tokens)
          table_tokens = tokenizer.tokenize(table_str, add_prefix_space=add_prefix_space)
          text_tokens.extend(table_tokens)
          text_tokens.append(tokenizer.sep_token)
          end_idx = start_idx + len(table_tokens)
          table_spans.append((start_idx, end_idx))
          start_idx = end_idx + 1

        table_start = [table_span[0] for table_span in table_spans]
        table_end = [table_span[1] for table_span in table_spans]

        input_ids = tokenizer.convert_tokens_to_ids(text_tokens)

        if len(input_ids) > block_size:
          continue

        label_ids = []
        try:
          for token in sql.split():
            if token in columns:
              label_ids.append(columns.index(token) + len(self.keywords))
            else:
              label_ids.append(self.keywords.index(token))
        except:
          continue

        label_ids = [self.label_bos_id] + label_ids + [self.label_eos_id]

        primary_key = [int(x) for x in example["sc_struct"]["primary_key"]]
        foreign_key = {x.split(",")[0]: int(x.split(",")[1]) for x in example["sc_struct"]["foreign_key"]}
        column_to_table = {"0": None}

        sc_link = {"q_col_match": {}, "q_tab_match": {}}
        for k, v in example["sc_link"]["q_col_match"].items():
          new_k = str(token_idx_to_sub_token_start_idx[int(k.split(",")[0])]) + "," + k.split(",")[1]
          sc_link["q_col_match"][new_k] = v

        for k, v in example["sc_link"]["q_tab_match"].items():
          new_k = str(token_idx_to_sub_token_start_idx[int(k.split(",")[0])]) + "," + k.split(",")[1]
          sc_link["q_tab_match"][new_k] = v

        cv_link = {"num_date_match": {}, "cell_match": {}}
        for k, v in example["cv_link"]["num_date_match"].items():
          new_k = str(token_idx_to_sub_token_start_idx[int(k.split(",")[0])]) + "," + k.split(",")[1]
          cv_link["num_date_match"][new_k] = v
        for k, v in example["cv_link"]["cell_match"].items():
          new_k = str(token_idx_to_sub_token_start_idx[int(k.split(",")[0])]) + "," + k.split(",")[1]
          cv_link["cell_match"][new_k] = v


        for idx, column in enumerate(columns):
          if column == "*":
            continue
          t = column.split(".")[0]
          column_to_table[str(idx)] = tables.index(t)

        foreign_keys_tables = {}
        for k, v in foreign_key.items():
          t_k = str(column_to_table[str(k)])
          t_v = str(column_to_table[str(v)])
          if t_k not in foreign_keys_tables:
            foreign_keys_tables[t_k] = []
          if int(t_v) not in foreign_keys_tables[t_k]:
            foreign_keys_tables[t_k].append(int(t_v))

        self.examples.append({
          "input_ids": input_ids,
          "example_info": {
            "normalized_question": text,
            "columns": columns,
            "tables": tables,
            "tokens": text_tokens,
            "question_start": question_start,
            "question_end": question_end,
            "column_start": torch.LongTensor(column_start),
            "column_end": torch.LongTensor(column_end),
            "table_start": torch.LongTensor(table_start),
            "table_end": torch.LongTensor(table_end),
            "sc_link": sc_link,
            "cv_link": cv_link,
            "primary_keys": primary_key,
            "foreign_keys": foreign_key,
            "column_to_table": column_to_table,
            "foreign_keys_tables": foreign_keys_tables
          },
          "column_spans": column_spans,
          "label_ids": label_ids})
        valid += 1
    print("Valid Example {}; Invalid Example {}".format(valid, total - valid))

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, i):
    return self.examples[i]


@dataclass
class DataCollatorForQuerySchemaRelation2SQL:
  """
  Data collator used for query + schema -> sql modeling.
  """
  tokenizer: PreTrainedTokenizer
  label_padding_id = label_mapping["keyword"].index(label_mapping["label_padding_token"])
  label_eos_id = label_mapping["keyword"].index(label_mapping["label_eos_token"])
  label_bos_id = label_mapping["keyword"].index(label_mapping["label_bos_token"])
  def collate_batch(self, examples) -> Dict[str, torch.Tensor]:

    input_ids_sequences = [example["input_ids"] for example in examples]
    column_spans_sequences = [example["column_spans"] for example in examples]
    label_ids_sequences = [example["label_ids"] for example in examples]
    padded_input_ids_tensor = pad_and_tensorize_sequence(
      input_ids_sequences, padding_value=self.tokenizer.pad_token_id)
    padded_column_spans_tensor = pad_and_tensorize_sequence(
      column_spans_sequences, padding_value=(0, 1))

    example_info_list = []
    for example in examples:
      example_info_list.append(example["example_info"])
    label_ids_tensor = pad_and_tensorize_sequence(
      label_ids_sequences, padding_value=self.label_padding_id)
    return {
      "input_ids": padded_input_ids_tensor,
      "column_spans": padded_column_spans_tensor,
      "labels": label_ids_tensor,
      "example_info_list": example_info_list,
      "input_padding_id": self.tokenizer.pad_token_id,
      "label_padding_id": self.label_padding_id,
      "label_eos_id": self.label_eos_id,
      "label_bos_id": self.label_bos_id
    }

